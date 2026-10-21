"""Finite automatic Pallas staging windows, distinct from explicit HBM Refs."""

from dataclasses import dataclass
from itertools import product
from math import prod

from ..errors import UnsupportedJaxprError
from .index_domain import AnalysisBudget, substitute
from .model import Origin, PallasCallView
from .protocol import Access, Instance
from .remote_coordinates import merge_spans
from .remote_grid import RemoteGridPlan, plan_remote_grid
from .storage import Span
from .tensor_domain import TensorDomain, prove_tensor_domain


@dataclass(frozen=True)
class RemotePort:
    ref: int
    hbm: int
    input: bool
    buffered: bool
    width: int
    size: int
    offsets: tuple[int, ...]
    restore_on_revisit: bool = False
    restore_programs: tuple[bool, ...] = ()
    shape: tuple[int, ...] = ()
    block_shape: tuple[int, ...] = ()
    mappings: tuple[tuple[tuple[Span, Span], ...], ...] = ()

    def segments(self, program: int) -> tuple[tuple[Span, Span], ...]:
        return (
            self.mappings[program]
            if self.mappings
            else ((Span(0, self.window(program).size), self.window(program)),)
        )

    def window(self, program: int) -> Span:
        start = self.offsets[program]
        return Span(start, min(start + self.width, self.size))


@dataclass(frozen=True)
class StagingAccess:
    port: int
    phase: str
    program: int
    command: Access


@dataclass(frozen=True)
class RemoteIOPlan:
    domains: tuple[TensorDomain, ...]
    ports: tuple[RemotePort, ...]
    next_resource: int
    grid: RemoteGridPlan

    def accesses(
        self, program: int, device: int, path: str, *, before: bool
    ) -> tuple[StagingAccess, ...]:
        """Replay native staging effects, including synchronous ANY restoration.

        The locked Buffered/RevisitMode contract restores earlier output state;
        single-buffer native lowering orders restoration before the invocation.
        These local effects add no peer synchronization or first-visit values.
        """
        result = []
        for i, port in enumerate(self.ports):
            restore = (
                before
                and not port.input
                and port.restore_on_revisit
                and port.restore_programs[program]
            )
            if not port.buffered or (port.input != before and not restore):
                continue
            offsets = port.offsets
            adjacent = self.grid.previous[program] if before else self.grid.following[program]
            boundary = adjacent is None or offsets[program] != offsets[adjacent]
            if not boundary:
                continue
            hbm, buffer = Instance(device, port.hbm), Instance(device, port.ref)
            pairs = port.segments(program)
            local = merge_spans(tuple(a for a, _ in pairs))
            global_ = merge_spans(tuple(b for _, b in pairs))
            full = (Span(0, port.width),)
            entries = (
                (
                    ("restore_read" if restore else "input_read", hbm, global_, False),
                    ("restore_write" if restore else "input_write", buffer, full, True),
                )
                if before
                else (("output_read", buffer, local, False), ("output_write", hbm, global_, True))
            )
            for phase, storage, spans, write in entries:
                origin = Origin(f"{path}.device[{device}].program[{program}].{phase}[{i}]")
                result.append(
                    StagingAccess(
                        i,
                        phase,
                        program,
                        Access(storage, Span(spans[0].start, spans[-1].stop), write, origin, spans),
                    )
                )
        return tuple(result)


def plan_remote_io(call: PallasCallView, budget: AnalysisBudget) -> RemoteIOPlan:
    path = call.kernel.origin.path
    grid = plan_remote_grid(call, budget)
    programs = len(grid.points)
    budget.spend(path + ".io_programs", programs * len(call.blocks))
    domains, ports = [], []
    next_resource = max(v.id for v in call.kernel.inputs) + 1
    for i, block in enumerate(call.blocks):
        shape, block_shape = block.array.shape, block.block_shape
        if not shape or any(n <= 0 for n in shape) or len(shape) != len(block_shape):
            raise UnsupportedJaxprError(
                block.origin.path, "remote staging requires positive static windows"
            )
        domain = prove_tensor_domain(block, grid.shape, budget, writer=False)
        size, width = prod(shape), prod(block_shape)
        offsets, mappings = [], []
        for program in range(programs):
            mapping = tuple(substitute(v, grid.coordinates(program)) for v in domain.mapping)
            if any(v.kind != "constant" for v in mapping):
                raise UnsupportedJaxprError(
                    block.origin.path, "remote window mapping must resolve at each program"
                )
            origin = tuple(int(v.value) * n for v, n in zip(mapping, block_shape, strict=True))
            if any(not 0 <= k < n for k, n in zip(origin, shape, strict=True)):
                raise UnsupportedJaxprError(
                    block.origin.path, "remote staging window is outside logical storage"
                )
            offset = sum(k * prod(shape[i + 1 :]) for i, k in enumerate(origin))
            offsets.append(offset)
            extent = tuple(
                min(b, n - k) for k, b, n in zip(origin, block_shape, shape, strict=True)
            )
            # Pair row intervals at identical logical block coordinates. Collapse
            # full common suffixes, so large contiguous blocks cost one segment.
            split = len(shape) - 1
            while (
                split > 0
                and origin[split] == 0
                and extent[split] == shape[split] == block_shape[split]
            ):
                split -= 1
            budget.spend(path + ".staging_segments", prod(extent[:split]))
            segments = []
            for prefix in product(*(range(n) for n in extent[:split])):
                local_start = sum(k * prod(block_shape[i + 1 :]) for i, k in enumerate(prefix))
                global_start = offset + sum(k * prod(shape[i + 1 :]) for i, k in enumerate(prefix))
                length = extent[split] * prod(extent[split + 1 :])
                segments.append(
                    (
                        Span(local_start, local_start + length),
                        Span(global_start, global_start + length),
                    )
                )
            mappings.append(tuple(segments))
        if block.unbuffered and (block_shape != shape or any(offsets)):
            raise UnsupportedJaxprError(
                block.origin.path, "remote ANY storage requires a fixed full HBM Ref"
            )
        mode = block.pipeline_mode
        restore = mode is not None and mode.revisit == "any"
        # JAX's native XLA emitter supports ANY only with synchronous buffering.
        # The separate Python pipeline.Scheduler is not evidence for this path.
        if mode is not None and (
            mode.buffer_count != 1
            or mode.use_lookahead
            or mode.prefetched_count != 0
            or mode.revisit not in (None, "immediate", "any")
            or block.unbuffered
        ):
            raise UnsupportedJaxprError(
                block.origin.path, "remote staging requires a reviewed synchronous pipeline"
            )
        if i >= call.num_inputs:
            windows = set(offsets)
            coverage = merge_spans(
                tuple(
                    span
                    for offset in sorted(windows)
                    for _, span in mappings[offsets.index(offset)]
                )
            )
            if coverage != (Span(0, size),):
                raise UnsupportedJaxprError(block.origin.path, "remote output windows leave a gap")
            visits = sum(
                (previous := grid.previous[j]) is None or offset != offsets[previous]
                for j, offset in enumerate(offsets)
            )
            if not block.unbuffered and not restore and visits != len(windows):
                raise UnsupportedJaxprError(
                    block.origin.path,
                    "nonconsecutive automatic output revisit requires a source writeback-order "
                    "contract",
                )
        ref = call.kernel.inputs[i].id
        hbm = next_resource if not block.unbuffered else ref
        if not block.unbuffered:
            next_resource += 1
        repeats = [False] * programs
        for stream in grid.streams:
            visited: set[int] = set()
            for program in stream:
                repeats[program] = offsets[program] in visited
                visited.add(offsets[program])
        ports.append(
            RemotePort(
                ref,
                hbm,
                i < call.num_inputs,
                not block.unbuffered,
                width,
                size,
                tuple(offsets),
                restore and i >= call.num_inputs,
                tuple(repeats) if restore and i >= call.num_inputs else (),
                shape,
                block_shape,
                tuple(mappings),
            )
        )
        domains.append(domain)
    return RemoteIOPlan(tuple(domains), tuple(ports), next_resource, grid)

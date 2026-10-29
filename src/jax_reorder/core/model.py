"""Immutable source facts shared by graph, address and memory consumers.

This module deliberately has no JAX or symbolic-backend imports. Source handles
belong to the adaptation boundary; consumers receive only ProgramView.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

type Attribute = None | bool | int | str | bytes | tuple[Attribute, ...]


@dataclass(frozen=True)
class Origin:
    path: str


@dataclass(frozen=True)
class DeviceView:
    id: int
    process: int
    platform: str
    kind: str
    coordinates: tuple[int, ...] | None = None
    core_on_chip: int | None = None


@dataclass(frozen=True)
class MeshView:
    axes: tuple[tuple[str, int, str], ...]
    devices: tuple[DeviceView, ...]
    abstract_device: Attribute
    concrete: bool


@dataclass(frozen=True)
class ShardingView:
    mesh: MeshView
    partitions: tuple[tuple[str, ...], ...]
    memory_kind: str | None
    logical_device_ids: tuple[int, ...] | None


@dataclass(frozen=True)
class ShardMapView:
    mesh: MeshView
    input_partitions: tuple[tuple[tuple[str, ...], ...], ...]
    output_partitions: tuple[tuple[tuple[str, ...], ...], ...]
    check_vma: bool
    newly_manual_axes: tuple[str, ...]


@dataclass(frozen=True)
class ArrayType:
    shape: tuple[int, ...]
    dtype: str
    weak_type: bool
    # Complete reviewed sharding/layout/memory/axis descriptors, without source objects.
    metadata: tuple[tuple[str, Attribute], ...]
    sharding: ShardingView | None = None


@dataclass(frozen=True)
class SemaphoreType:
    shape: tuple[int, ...]
    kind: str
    metadata: tuple[tuple[str, Attribute], ...]


@dataclass(frozen=True)
class RefType:
    inner: ArrayType | SemaphoreType
    memory_space: Attribute
    kind: Attribute


type ValueType = ArrayType | RefType


@dataclass(frozen=True)
class ValueDef:
    id: int
    type: ValueType
    dropped: bool = False


@dataclass(frozen=True)
class ValueRef:
    id: int


@dataclass(frozen=True)
class LiteralValue:
    type: ArrayType
    data: bytes


type Operand = ValueRef | LiteralValue


@dataclass(frozen=True)
class ParameterView:
    name: str
    value: Attribute


@dataclass(frozen=True)
class EffectView:
    kind: Literal["read", "write"]
    ref: ValueRef


@dataclass(frozen=True)
class AxisEffectView:
    name: str


@dataclass(frozen=True)
class GlobalEffectView:
    kind: Literal["semaphore", "communication"]


type Effect = EffectView | AxisEffectView | GlobalEffectView


@dataclass(frozen=True)
class SliceIndex:
    start: int
    size: int
    stride: int


@dataclass(frozen=True)
class DynamicSliceIndex:
    start: Operand
    size: int
    stride: int


@dataclass(frozen=True)
class TensorIndex:
    indices: tuple[SliceIndex | DynamicSliceIndex | Operand, ...]
    shape: tuple[int, ...]
    integer_shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class RefTransform:
    kind: Literal["reshape", "transpose", "index"]
    dimensions: tuple[int, ...]
    indices: tuple[SliceIndex | DynamicSliceIndex | int | Operand, ...] = ()


@dataclass(frozen=True)
class MemoryAccess:
    ref: ValueRef
    index: SliceIndex | TensorIndex | Operand
    value: Operand | None = None
    mask: Operand | None = None
    other: Operand | None = None
    scalar: bool = False
    transforms: tuple[RefTransform, ...] = ()


@dataclass(frozen=True)
class DmaView:
    # Individual source events, never pre-paired by the adapter.
    phase: Literal["start", "wait"]
    source: ValueRef
    destination: ValueRef
    semaphore: ValueRef
    device_id_type: str
    priority: int | None = None
    add: bool | None = None
    insert_dummy_device: bool | None = None
    is_wait_send: bool | None = None
    source_slice: SliceIndex | None = None
    destination_slice: SliceIndex | None = None
    # Raw source tree positions, including the intentional swap on wait_send.
    source_semaphore: ValueRef | None = None
    device_coordinates: tuple[Operand, ...] | None = None
    source_tensor_slice: TensorIndex | None = None
    destination_tensor_slice: TensorIndex | None = None
    semaphore_index: TensorIndex | None = None
    source_semaphore_index: TensorIndex | None = None
    source_identity_views: tuple[tuple[int, ...], ...] = ()
    destination_identity_views: tuple[tuple[int, ...], ...] = ()
    source_transforms: tuple[RefTransform, ...] = ()
    destination_transforms: tuple[RefTransform, ...] = ()


@dataclass(frozen=True)
class ScratchView:
    type: RefType
    lifetime: str
    initial_value: int | None
    # A VMEM allocation is local to this kernel invocation. Preserve its raw
    # descriptor above as well as its type after binding the local mesh.
    bound_array: ArrayType | None = None


@dataclass(frozen=True)
class SemaphoreView:
    phase: Literal["signal", "wait", "read"]
    semaphore: ValueRef
    amount: Operand | None = None
    device_coordinates: tuple[Operand, ...] | None = None
    core_index: Operand | None = None
    index: TensorIndex | None = None


@dataclass(frozen=True)
class RegionBinding:
    role: str
    region: RegionView
    arguments: tuple[Operand, ...]
    results: tuple[ValueRef, ...]


@dataclass(frozen=True)
class OperationView:
    id: int
    primitive: str
    operands: tuple[Operand, ...]
    results: tuple[ValueDef, ...]
    parameters: tuple[ParameterView, ...]
    effects: tuple[Effect, ...]
    context: Attribute
    origin: Origin = field(compare=False)
    access: MemoryAccess | None = None
    call: PallasCallView | None = None
    regions: tuple[RegionBinding, ...] = ()
    shard_map: ShardMapView | None = None
    dma: DmaView | None = None
    semaphore: SemaphoreView | None = None
    allocations: tuple[ScratchView, ...] = ()


@dataclass(frozen=True)
class RegionView:
    inputs: tuple[ValueDef, ...]
    constants: tuple[tuple[ValueDef, LiteralValue], ...]
    operations: tuple[OperationView, ...]
    outputs: tuple[Operand, ...]
    effects: tuple[Effect, ...]
    is_high: bool
    origin: Origin = field(compare=False)


@dataclass(frozen=True)
class PipelineModeView:
    buffer_count: int
    use_lookahead: bool
    revisit: str | None
    prefetched_count: int


@dataclass(frozen=True)
class BlockMappingView:
    block_shape: tuple[int, ...]
    array: ArrayType
    ref: RefType
    index_map: RegionView
    origin: Origin = field(compare=False)
    unbuffered: bool = False
    pipeline_mode: PipelineModeView | None = None
    # Array axes indexed by one program coordinate and absent from the local Ref.
    # Their physical window width remains one in block_shape.
    squeezed_axes: tuple[int, ...] = ()


@dataclass(frozen=True)
class GridAxis:
    size: int
    # None means the backend default, not an assertion of parallel execution.
    semantics: str | None = None


@dataclass(frozen=True)
class CompilerView:
    explicit: bool
    backend: str
    parameters: tuple[ParameterView, ...]
    # Exact equality to the reviewed defaults, excluding dimension_semantics.
    default_execution_options: bool
    # Reviewed TensorCore layout/codegen hints preserve logical coordinates and
    # dot dimensions. Exact options remain in parameters and call signatures.
    reviewed_layout_options: bool = False
    # Capacity and runtime bounds-check settings preserve successful values only
    # after the tensor executor proves every memory access independently.
    reviewed_resource_options: bool = False


@dataclass(frozen=True)
class PallasCallView:
    grid: tuple[GridAxis, ...]
    blocks: tuple[BlockMappingView, ...]
    kernel: RegionView
    num_inputs: int
    num_outputs: int
    interpret: bool
    # Complete source contracts; individual rewrite rules may admit less.
    aliases: tuple[tuple[int, int], ...] = ()
    compiler: CompilerView | None = None
    kernel_argument_policy: str = "identity"
    scratch: tuple[ScratchView, ...] = ()
    prefetch: tuple[RefType, ...] = ()
    cost_estimate: tuple[int, int, int, int] | None = None
    # Locked JAX serializes these diagnostic strings into HLO frontend attributes.
    # None and an explicitly empty mapping remain distinct source facts.
    metadata: tuple[tuple[str, str], ...] | None = None
    # Positional index-map arguments: True is a flat leaf, False is None.
    index_map_bindings: tuple[bool, ...] = ()


@dataclass(frozen=True)
class AdaptationContext:
    jax_version: str = "0.11.2"
    jaxlib_version: str = "0.11.2"
    derivative_policy: str = "forward"
    environment: tuple[tuple[str, Attribute], ...] = ()
    pallas_backend: str = "tpu"


@dataclass(frozen=True)
class ProgramView:
    root: RegionView
    context: AdaptationContext


@dataclass(frozen=True)
class SourceHandle:
    """Diagnostic/reconstruction handle, never passed to analysis consumers."""

    source: object = field(compare=False, repr=False)


@dataclass(frozen=True)
class AdaptedProgram:
    view: ProgramView
    source: SourceHandle

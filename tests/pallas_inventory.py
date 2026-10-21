"""Auditable public TPU surface inventory for the locked JAX version.

Categories distinguish executable trace evidence from declarations, tooling,
source rejection and the separate MPMD frontend. No identity admission count is
reported as runtime or transformation coverage.
"""

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from tests.pallas_coverage_cases import CASES


def inventory():
    result = {api: f"trace:{case.name}" for case in CASES for api in case.apis}
    groups = {
        "trace:test_block_dimension_descriptors": (
            "pl.Blocked pl.BlockSpec pl.BoundedSlice pl.Buffered pl.CostEstimate pl.Element "
            "pl.Squeezed pl.pallas_call"
        ),
        "trace:test_multidimensional_index_maps_scalar_prefetch_and_dynamic_grid": (
            "tpu.PrefetchScalarGridSpec pl.GridSpec"
        ),
        "trace:test_official_tpu_operation_families": "pl.debug_check pl.debug_print",
        "trace:test_remote_dma_conditional_waits_and_device_routes_are_structural": (
            "tpu.async_remote_copy tpu.make_async_remote_copy"
        ),
        "upstream-rejection:test_indirect_array_index_is_rejected_by_locked_pallas_frontend": (
            "pl.Indirect"
        ),
        "outside-one-pallas-call:MPMD-frontend": (
            "pl.kernel tpu.emit_pipeline_with_async_prefetch tpu.TensorCoreMesh"
        ),
        "host-helper:no-primitive": "pl.next_power_of_2 pl.strides_from_shape pl.estimate_cost",
        "tooling:configuration-or-lowering": (
            "pl.enable_debug_checks pl.enable_poison_buffers pl.lower_as_mlir "
            "pl.pallas_export_experimental tpu.force_tpu_interpret_mode "
            "tpu.reset_tpu_interpret_mode_state tpu.set_tpu_interpret_mode "
            "tpu.InterpretParams tpu.LoweringException"
        ),
        "tooling:device-metadata": (
            "tpu.ChipVersion tpu.get_tpu_info tpu.get_tpu_info_for_chip tpu.is_tpu_device "
            "tpu.TpuInfo"
        ),
        "declaration:exact-parameter-encoding": (
            "pl.CompilerParams pl.MemoryRef pl.MemorySpace pl.RevisitMode pl.semaphore "
            "pl.DeviceIdType pl.ANY tpu.CompilerParams tpu.CoreType tpu.dma_semaphore "
            "tpu.GridDimensionSemantics tpu.MemorySpace tpu.SemaphoreType tpu.SideEffectType "
            "tpu.BufferedRef tpu.BufferedRefBase tpu.BufferType tpu.PipelineStep tpu.Tiling "
            "tpu.CMEM tpu.SMEM tpu.VMEM tpu.VMEM_SHARED tpu.HBM tpu.SEMAPHORE"
        ),
    }
    for category, names in groups.items():
        result.update((name, category) for name in names.split())
    actual = {
        f"{prefix}.{name}"
        for prefix, module in (("pl", pl), ("tpu", tpu))
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and callable(obj)
        and getattr(obj, "__module__", "").startswith("jax")
    }
    assert actual == result.keys(), (actual - result.keys(), result.keys() - actual)
    return result


if __name__ == "__main__":
    for name, category in sorted(inventory().items()):
        print(f"{name}: {category}")

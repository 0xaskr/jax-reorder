"""CPU 验收配置，以及快速、完整业务与压力任务的固定分组。"""

import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")


def pytest_collection_modifyitems(items):
    business = {"test_business_layers.py", "test_sglang_model_layers.py", "test_full_attention.py"}
    for item in items:
        if item.path.name in business:
            item.add_marker(pytest.mark.business)
        elif item.path.name in {"test_depth.py", "test_stress.py"}:
            item.add_marker(pytest.mark.stress)

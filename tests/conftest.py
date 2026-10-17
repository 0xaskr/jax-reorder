"""pytest 全局配置：固定 CPU 平台、单进程设备数，以及快速/业务/压力任务分组。

本文件在测试收集阶段生效，不包含断言：
- 平台固定为 CPU，避免在带有 GPU/TPU 的机器上切换到未审查的 lowering。
- 用 XLA_FLAGS 强制暴露 2 个 host 设备，使集合通信与 shard_map 测试可以在
  单进程内运行；需要 4 设备的组合由 examples/collective_jaxpr.py --devices 4 单独验证。
- 按文件名把业务层验收标为 ``business``，把深度/规模测试标为 ``stress``，供 CI 分任务运行。

这里只影响本地默认值，不构成对比较算法本身的证明：分组正确不代表测试通过，
也不能说明未列入分组的文件一定可以跳过。
"""

import os

import pytest

# setdefault 表示尊重调用方已显式设置的环境变量，只在缺省时给出 CPU 验收默认值。
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")


def pytest_collection_modifyitems(items):
    """按文件名登记 CI 分组；标记只决定运行批次，不改变测试的断言与前置条件。"""
    business = {"test_business_layers.py", "test_sglang_model_layers.py", "test_full_attention.py"}
    for item in items:
        if item.path.name in business:
            item.add_marker(pytest.mark.business)
        elif item.path.name in {"test_depth.py", "test_stress.py"}:
            item.add_marker(pytest.mark.stress)

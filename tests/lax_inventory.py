"""重新追踪 lax 固定用例并核对原语清单文档；--write 才写回。

清单只记录“测试绑定关系”，执行证据来自完整 pytest 运行，脚本本身不执行全部计算。
默认模式把重新追踪的结果与 docs/development-docs/lax-coverage.md 逐字比较，清单过期
即断言失败，从而阻止文档漂移或漏掉公开原语；--write 只应在审查覆盖变化后使用。
前置条件：conftest 之外的独立入口自行固定 CPU 平台和 2 个 host 设备。
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax  # noqa: E402
from jax._src import core  # noqa: E402
from lax_cases import cases, trace_case  # noqa: E402

from jax_reorder.support import public_primitives  # noqa: E402


def primitives(graph):
    """收集图及其子图参数中出现的全部 primitive，用于核对用例实际追踪结果。"""
    found = set()
    pending = [graph]
    while pending:
        value = pending.pop()
        if isinstance(value, core.Jaxpr):
            for equation in value.eqns:
                found.add(equation.primitive)
                pending.extend(equation.params.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
    return found


def report():
    """生成清单文档全文；每个公开原语都必须能由一个固定用例绑定，否则当场断言失败。"""
    fixtures = cases()
    traced = {c.name: primitives(trace_case(c)) for c in fixtures}
    exported = public_primitives()
    rows = []
    for primitive, names in sorted(exported.items(), key=lambda pair: pair[0].name):
        targets = [c.name for c in fixtures if c.expected == primitive.name]
        assert targets, primitive
        evidence = "CPU 执行、结构比较、独立分支换序"
        if primitive is jax.lax.stage_p:
            evidence = "公开调用被 JAX 追踪消去；不计 IR 执行覆盖"
        elif primitive is jax.lax.ragged_all_to_all_p:
            evidence = "结构处理通过；断言 XLA:CPU 明确拒绝执行"
        else:
            assert any(primitive in traced[t] for t in targets), primitive
        rows.append(
            f"| `{primitive.name}` | {', '.join(f'`{n}`' for n in names)} | "
            f"{', '.join(f'`{t}`' for t in targets)} | {evidence} |"
        )
    extra = set().union(*traced.values()) - exported.keys()
    lines = [
        "# CPU lax 原语清单与验证绑定",
        "",
        f"基线 JAX {jax.__version__}；`jax.lax` 与公开子模块 `jax.lax.linalg` "
        f"共 {len(exported)} 个去重的公开 Primitive 对象。",
        f"当前有 {len(fixtures)} 个原语及参数用例，额外追踪出 {len(extra)} 个辅助原语。",
        "",
        "该清单由 `uv run python tests/lax_inventory.py --write` 重新追踪生成。"
        "它记录测试绑定关系；只有完整 `uv run pytest` 通过才构成 CPU 执行证据。"
        "脚本本身不运行全部计算。测试会阻止清单漂移或漏掉公开原语。",
        "",
        "原语在 CPU 支持的全部合法参数、dtype、shape 都属于目标域；表中的代表值"
        "不构成白名单，也不宣称穷举该域。参数用带类型的完整结构编码，子图递归处理。"
        "[设计、适配与审查](bfs-reordering.md)解释具体规则和边界。",
        "",
        "## 公开原语",
        "",
        "169 个公开原语有 CPU 执行用例；另有 stage 的追踪消去检查和 "
        "ragged_all_to_all 的 CPU 拒绝检查。后两者不计为 IR 执行通过。",
        "",
        "| 实际 primitive | 公开导出 | 固定用例 | 验证方式 |",
        "| --- | --- | --- | --- |",
        *rows,
        "",
        "## 包装函数实际产生的辅助原语",
        "",
        "| primitive | 命中的固定用例 |",
        "| --- | --- |",
    ]
    for p in sorted(extra, key=lambda p: p.name):
        evidence = [name for name, found in traced.items() if p in found]
        lines.append(f"| `{p.name}` | {', '.join(f'`{n}`' for n in evidence)} |")
    lines.extend(
        [
            "",
            "`pbroadcast` 的用例显式断言 CPU 缺少 lowering。`psend`、`precv` 的"
            "独立测试断言 JAX 仅提供 GPU 实现。它们不计为 CPU 执行覆盖。"
            "`dce_sink` 同时检查 `prevent_mlir_dce=False/True`；`psum` 检查 "
            "`check_vma=False/True`，后者实际产生 `psum_invariant`。",
            "",
            "## 固定参数与输入签名",
            "",
            "全部静态参数及数值输入保存在 [lax_cases.py](../../tests/lax_cases.py)。"
            "以下签名对应默认 X64 关闭的原语用例；"
            "[test_semantics.py](../../tests/test_semantics.py)"
            "另在 X64 开启时覆盖 19 种 dtype × 5 种 shape，以及 key dtype 和捕获 key 数组。",
            "",
            "| 用例 | 输入叶子的 dtype / shape |",
            "| --- | --- |",
        ]
    )
    for c in fixtures:
        signatures = [f"`{x.dtype}{tuple(x.shape)}`" for x in jax.tree.leaves(c.args)]
        lines.append(f"| `{c.name}` | {', '.join(signatures) or '无输入'} |")
    return "\n".join(lines) + "\n"


def main():
    """比较或写回清单文档；默认只核对，--write 才更新受审查的覆盖表。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "docs/development-docs/lax-coverage.md"
    content = report()
    if args.write:
        path.write_text(content)
        print(f"Wrote {path}")
    else:
        assert path.read_text() == content, (
            "inventory is stale; regenerate after reviewing coverage"
        )
        print("Inventory matches the locked JAX environment")


if __name__ == "__main__":
    main()

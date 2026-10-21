# 历史：9.1 本轮检查

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

## 9. 验证与讨论记录

### 9.1 本轮检查

- 定向 pytest：`tests/test_package_layout.py`、`tests/test_canonical.py`、`tests/test_semantics.py`
  共 130 passed；`tests/test_correspondence.py`、`tests/test_algebra_structural.py` 共 203 passed。
- 全量 pytest：两次后台运行在 agent 环境中长时间停在早期用例后终止；未完成，不报告为通过。
- `ruff check .` 通过；`ruff format --check .` 为 193 files already formatted；
  `tools/check_docs.py --changed` 通过；`git diff --check` 通过。
- 本轮生产代码未变；新增/修改仅为本文及开发文档索引已有的入口。

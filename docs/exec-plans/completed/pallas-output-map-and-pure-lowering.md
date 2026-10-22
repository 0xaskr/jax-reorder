# 顺序输出映射与纯死计算 lowering

## 目标与完成条件

用户确认输出映射也在当前 kernel 完成后更新：初值绑定首个输入/输出窗口，
当前 kernel 的读写始终使用已绑定窗口，状态更新只用于下一 program 的映射。
补齐串行、有限、可求定地址的状态驱动输出映射及其完整覆盖/窗口生命周期证明。
同时核对数学 lowering 的纯死计算删减；不因地址计算本身是纯函数就丢弃其活跃依赖。

## 设计与范围

- 复用语义层的版本化存储、串行 grid 展开、输出窗口保存/恢复与 writer 覆盖检查。
  输入和输出映射在同一 program 前读取相同的 prefetch 版本，完成 kernel 后才推进。
- 输出地址、每轮历史及全部有序输出必须进入最终证明；临时规划域不能作为成功证据。
  保留已有 ANY/默认 staging 规则，不放宽非法重访、越界、未初始化和 DMA 在途冲突。
- 仍要求有限静态串行 grid 与每步可求定的块地址；真实自动流水线时序仍是显式前提。
- 数学 lowering 可删未被输出或副作用消费的纯计算，完整结构比较保留原图。
  运行时未知地址若影响输出、读写或通信，不能据此删除；不扩展数据 peer 和动态 grid。

## 工作项

- [x] 语义层接入输出映射、窗口生命周期与最终覆盖证据。
- [x] 三个公开入口的正例、状态/目的地址错误反例及拒绝路径。
- [x] 核对并补齐纯死计算在数学 lowering 的处理与活跃依赖反例。
- [x] 独立完整坐标 oracle、相关回归、静态检查与开发文档；归档计划。

## 分工与验证

core、输出测试、代数 lowering 三项并行；主代理复核时序和证据边界、集成与文档。
针对输出 map 的 oracle 使用用户指定的 Python/NumPy 顺序模型，不把原生预取或
普通 interpret 的执行结果当作这一顺序的证明。逐步验证后再运行受影响状态/公开入口回归。
本轮不提交、不推送、不改受保护文档；保留工作区已有改动。

## 进展与结果

- 已确认现有输入 map 每轮在 kernel 前求值，读 prefetch 当前版本；输出 map 尚有明确拒绝。
- 已确认契约只允许数学 lowering 删除纯死计算；影响输出的有界未知地址仍是活跃依赖。
- 独立最小探针确认已有 lowering 能忽略未使用的 `counter & 1`；同一值参与输出或
  `index_map` 时保留依赖。第 2 项补边界回归和短文档，不新增重复 DCE 实现。
- 输出实现采用与静态 grid 共用的增量窗口证明，逐轮记录映射、检查重访，结束时证明
  全输出域覆盖；这样状态映射和原静态路径使用相同的生命周期规则。
- `test_pallas_semantic_slice.py` 全部 60 项通过，含纯死计算/活跃值/活跃地址对照，以及
  未使用但未初始化的读取不可删减；不需要增加代数生产实现。
- 初版状态回归两片分别为 74 passed/8 deselected（grid_state）和 258 passed/24 deselected
  （tensor_memory、dynamic_ranges、dynamic_views、owner_partition、core_execution）。
  输出专项发现 output-only reverse 的完整函数 CAS 未决；语义与局部数学、384 点 oracle
  已通过，原因是未使用窗口范围消去局部模坐标。修复后另做最终验收，不将此批称最终结果。
- 独立二维反向映射全部 768 点及 300 点尾块对照通过；另一路独立二维 768 点与
  350 点尾块对照通过。只说明指定逻辑顺序下的投影正确，不作为原生执行时序证据。

## 最终验收

- 复用已证明窗口与逻辑 shape 的交集简化输出坐标，保留原窗口条件；反向输出的三个
  公开入口全部通过，无需增加代数快路径。修复后的静态窗口专项 21 passed/61 deselected。
- 最终公开入口四模块 198 passed/7 deselected（128.66 秒）；投影三模块
  240 passed/24 deselected（133.93 秒）。两片文件不重叠，共 438 项，无失败、错误或跳过。
  通信标记用例按命令取消选择；此前阶段回归与专项结果不再加进此数。
- 新规则覆盖正序、反序、输入输出同时状态映射、连续累积、ANY 重访、混合输出生命周期，
  并检查状态错误反例、越界/缺口/未知地址/并行拒绝和执行前提篡改。
- 最终源码再次通过二维 768 点及尾块 300 点 oracle；无普通输入的输出 kernel 另以
  全部 384 点对照通过。纯死计算专项还用两个初值各 256 点验证活跃值必须保留。
- 全库 Ruff/格式（401 个 Python 文件）、basedpyright 零错误/警告、tracing equivalent、
  文档本地链接与 `git diff --check` 通过；受保护文档未修改。

最终复现命令（Python 3.12，已同步依赖）：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m pytest -q \
  tests/test_pallas_source_coverage.py tests/test_pallas_local_entry.py \
  tests/test_pallas_local_values.py tests/test_pallas_semantic_slice.py \
  -m 'not pallas_communication'
JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m pytest -q \
  tests/test_pallas_tensor_memory.py tests/test_pallas_padding_algebra.py \
  tests/test_pallas_dynamic_views.py -m 'not pallas_communication'
```

本机原始证据：`/tmp/output-map-public-final.xml`、`/tmp/output-map-projection-final.xml`、
`/tmp/output-map-independent-probes-final.log`；五个实现/测试文件的最终 SHA-256 位于
`/tmp/output-map-final-source-hashes.json`。完整命令输出保留在同名 `.log` 中。

## 保留边界

有界但未知的活跃输入/输出地址仍需要单独的符号地址证明；数据 peer 与动态 grid 未扩展。
纯函数的结果影响输出或副作用时不能删减。未使用但有资源/effect 义务的计算仍需验证。
执行顺序作为公开比较前提保留，不声称实际自动流水线一定采用相同预取顺序。

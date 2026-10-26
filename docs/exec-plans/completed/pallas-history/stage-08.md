# 历史：9.8 公共 Pallas 结构比较与真实开源内核（2026-10-19）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../../design-docs/current/pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.8 公共 Pallas 结构比较与真实开源内核（2026-10-19）

实现基于合并后的 `f8effb03c6a768490ef07e792fb64a3ced3a34d6`。公共
`compare_jaxprs`、`canonicalize`、`compare_functions(..., equivalence="structural")`
及自动模式的结构快路径现可处理 Pallas。结果使用现有 `Graph` 与完整原图双射，
不调用 `normalize_view` 的存储/协议证明，也不把数学上相同当作结构相同。
自动模式遇到结构不同仍进入独立代数流程；未支持的 Pallas 转换抛出
`UnsupportedJaxprError(stage="algebra", failure_kind="unsupported_conversion")`。

[源适配](../../../../src/jax_reorder/core/adapters/pallas/structural.py)按锁定 TPU registry 的
Primitive 身份接入，并补充高层 Ref 原语及 44 项 Pallas/state 参数 schema。
初始化时先加载官方 TPU frontend，避免 pipeline/PRNG 延迟注册造成导入顺序差异；
身份及 schema 准入**不是逐原语语法覆盖率**。自定义同名 Primitive、Python callback、
未知 effect、未知 schema 与不透明描述符仍拒绝。GridMapping/BlockMapping 包括嵌套 index map，
只省略已审计的调试名称/来源位置；网格、变换、编译参数、别名及存储空间均保留。

`run_scoped` 的捕获槽位跟随父操作数共同规范化，临时 tracer 载荷不作为常量数据。
私有 scratch 分配可按一致的资源映射重命名，外部输入/输出角色固定。
Ref 读写、DMA、semaphore、PRNG、debug/check、MXU 等状态操作均保留相对顺序；
JAX 的 `free_ref` 虽报告空 effects，仍按资源失效操作加顺序边。独立复核会检查
父子实参绑定、资源角色、别名及操作顺序，伪造相同 signal 节点的交换不能通过。

[结构测试](../../../../tests/test_pallas_structural.py)中的实际追踪覆盖如下；部分 API 只有结构自比较，
不能据此宣称硬件可执行或所有参数组合已覆盖。

| 类别 | 证据与反例 |
| --- | --- |
| 图与函数接口 | 独立纯计算换序；128/256 分块、逆向 index map、错误操作数返回 different；自动代数拒绝 |
| 数据与区域 | 多维/部分/动态切片、reshape/transpose、dot/reduce；cond、fori、while、scan；动态 grid 与 scalar prefetch |
| 资源 | scratch 和 scoped 分配换序、捕获实参换序；new_ref/freeze/free_ref、读写/累加；合并两个 semaphore 的反例 |
| 通信 | 本地/远端 DMA、条件 wait、请求—回复；计数/索引/设备路由与拆分 signal 的差异；不模拟运行时间 |
| TPU 辅助 API | masked load、bitcast、roll、stochastic round、pack/unpack、access assumption、barrier、显式/隐式 PRNG、MXU FIFO、debug print/check |
| 拒绝与复核 | 未知身份/effect/参数、伪造对应关系；规范化后 IR 检查、scoped kernel 的 CPU interpret 与幂等性 |

**GitHub 内核与 AI 正反例。**按用户要求选择两个具有实际 TPU 内核的活跃项目，
2026-10-19 查询时分别为 307/464 stars；热度仅用于选样，不作为正确性依据。

| 固定来源 | 实际采用的定义与配置 | 变体数 |
| --- | --- | --- |
| [openxla/tokamax，991f283](https://github.com/openxla/tokamax/blob/991f283fb4f441499e848858606abbe373bb86b0/tokamax/_src/ops/deepseek_rope/pallas_mosaic_tpu_kernel.py) | 未改写的 `_rope_body` 及其四个 helper；GPT-J interleaved RoPE，8×128、rotary_dim=64；FP32 原地输出与 int8 量化/scale 两种分支 | 10 |
| [vllm-project/tpu-inference，d85861a](https://github.com/vllm-project/tpu-inference/blob/d85861af245fab2bd9c69e789545e6bcfefbfc3b/tpu_inference/kernels/flash_attention/kernel.py) | 未改写的 `_flash_attention_kernel_single_batch_single_step`；单 batch/head、128 token/head dim、causal、scale=0.5 | 7 |

[来源清单](../../../../tests/fixtures/pallas_kernels/manifest.json)固定 commit、git blob、SHA-256 和原 URL，
保存完整原文件及各项目 Apache-2.0 许可证。加载器仅提取完整函数定义，注入实际 JAX/Pallas 与
原常量，不安装上游服务、不导入调参器、不替换原计算。候选由 AI 本地逐个编写，
位于 [tests/pallas_kernels.py](../../../../tests/pallas_kernels.py)，预期标签固定在源码中。

17 组结果：6 组结构 equivalent，11 组结构 different。三个真正重排的正例确认原方程顺序发生改变；
另三个未使用 `sin` 在 **Pallas tracing 阶段**就被上游 DCE，传给比较器的图已经相同。
这不改变本项目“结构层保留收到的全部节点，纯死代码只允许在代数 lowering 删除”的规则。
11 个 different 中，8 个改变符号、旋转方向、量化/attention scale、causal 边界、归约轴或 V 资源，
两组固定输入均有数值反例；其余 3 个插入参与输出的乘一节点，数值一致但计算图不同。

[开源回归](../../../../tests/test_pallas_upstream.py)用种子 7/23 做 34 对 CPU interpret，
原始结果同时对照独立 NumPy RoPE/量化/attention 实现。结构正例复核原图双射，所有规范化图
通过 JAX IR 检查。数值抽样独立报告，不替代结构证明；这些配置不代表整个 Tokamax/vLLM 库。

**实际剩余边界。**大型 `emit_pipeline` 组合仍会让嵌套区域搜索耗时较长，GLM5 absorbed
Attention/Decoder 的四个配置遇到 8 个歧义 carry 槽位，排列数超过默认 10,000 预算而明确失败。
现有 Wan cross attention 与 EPMoE 原图可结构自比较，但这不算候选通过；CPU 编译与图比较
分开记录，见[完整层记录](../validation/full-attention.md)。没有真实 TPU 编译/运行证据，多主机非本地常量
仍拒绝；尚未逐一审计全部官方语法组合。跨 block/grid 的数学证明、协议证明与 Pallas↔JAX
保持后续工作，不能宣称“任意合法 Pallas 已全覆盖”。

可复现入口：

```bash
uv run python -m tests.pallas_kernels
uv run pytest -q tests/test_pallas_structural.py tests/test_pallas_upstream.py
uv run pytest -q tests/test_pallas_*.py
```

最终分组回归和静态检查结果见[执行记录](../pallas-structural-comparison.md)。

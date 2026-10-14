# GLM-5.2 纯 JAX decode DSA/MLA 的 5+5 测试

## 来源与边界

开源实现是 [SGLang-JAX DSA reference](https://github.com/sgl-project/sglang-jax/blob/c98ced4a7a64e83261f6672937ce41f0095828ba/python/sgl_jax/srt/kernels/dsa/ref.py)，提交固定为 `c98ced4a7a64e83261f6672937ce41f0095828ba`。完整文件保存到[快照](../../tests/fixtures/sglang_jax/python/sgl_jax/srt/kernels/dsa/ref.py.txt)，包含原始注释；[清单](../../tests/fixtures/sglang_jax/manifest.json)校验 SHA-256，[Git tree 摘要](../../tests/fixtures/sglang_jax/upstream-attention-tree.json)独立校验 Git blob。沿用快照目录中的 Apache-2.0 许可证。

reference 只导入 JAX/JNP 和标准库。测试原样执行 `streamindex_topk_ref(one_token_per_seq=True)` 和 `sparse_mla_ref`，前者按每序列一个 query 进行 decode top-k；后者从非恒等物理页表收集被选中的压缩 KV，计算 softmax 与加权 value。原始 approx_max_k、padding 的 -1、有效长度屏蔽和空序列行为全部保留。

[新脚本](../../examples/glm52_decode_dsa_mla.py)将这两个函数与 `Glm5Attention._forward_mqa` 中的 query absorption、value expansion 及输出投影公式组合。基线的线性组合是本仓库显式编写的适配层，不是整个上游 `Glm5Attention` 原样实例化。输入 query 已投影并施加 RoPE，输入 cache 已更新；不包含前端线性层、归一化、RoPE 生成、缓存写入、完整 MoE 或 IndexShare 跨层传播。

默认 `glm` 配置保留 latent=512、rope=64、nope=192、value=256、index_heads=32、index_dim=128，与[模型配置](https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json)及固定上游类一致。为 CPU 验证将 attention heads 缩为 2、输出宽度缩为 16，batch=2，每序列最多 12 token，top-k=4；完整模型为 64 heads、hidden=6144、top-k=2048。另有 `small` 配置供快速回归。只测 FP32 前向，不代表 BF16 舍入路径、完整模型、真实权重或性能。

## 手写版本

全部候选函数直接编写，未从 base 的 AST/jaxpr 自动生成。5 个正例为：

| 函数 | 变化 | 预期证据 |
| --- | --- | --- |
| `query_first` | query absorption 移到独立的索引器之前 | 完整结构等价，原始执行顺序确实改变 |
| `query_broadcast` | query 的 einsum 改广播乘积与求和 | 完整层代数证明 |
| `value_broadcast` | value 的 einsum 改广播乘积与求和 | 完整层代数证明 |
| `fused_output` | 先合并 value/output 权重，再投影 latent | 完整层代数证明；矩阵乘法重新结合 |
| `headwise_output` | 输出投影先逐 head 计算，再对 head 求和 | 完整层代数证明；拆分归约轴 |

5 个反例分别为 `wrong_scale`（错误地按 latent+rope 维度缩放）、`wrong_rope`（丢弃 rotary 分支）、`wrong_selection`（忽略 top-k 而取前 k 项）、`wrong_pages`（忽略物理页表）、`wrong_value`（倒置 value 权重的 head 对应）。每个保留相同输入输出形状，必须有实际浮点输出差异，不能只依靠返回的索引不同。

## 验证方法

- `compare_functions` 检查输入输出 pytree、完整前向图和 CPU 编译。正例必须有结构或代数证明；4 个代数正例另外独立复核证据，保留浮点警告。
- 共同原始 JIT 函数采用比较器已有的精确函数符号规则。其完整图、参数与输入绑定均参与验证；未简化或删除 top-k、mask、分页 gather 和 softmax。
- 错误候选的结构不匹配、代数 `unknown` 仅表示未证明。数学反例由实际数值差异另行给出，不将 `unknown` 改称“不等价证明”。
- 独立 NumPy FP64 参照逐序列/逐头求值，CPU 上逐项核对 top-k 和输出；TPU 的 approx_max_k 允许近似选择，未将 CPU 的 exact top-k 对照扩展到 TPU。
- 默认两个随机种子、非恒等页表、两条不同有效长度，覆盖真实稀疏选择与 -1 padding。回归另测空序列、单 token、页边界、满页，以及改变索引器权重确实改变 attention 输出。
- 代数等价限于理想实数解释；浮点重排、分配或重新结合可能改变舍入、溢出、NaN/Inf 和有符号零，不承诺逐位相同或误差始终很小。

## 运行与结果

```bash
uv run python -m examples.glm52_decode_dsa_mla --json /tmp/glm52.json
uv run pytest -q tests/test_glm52_decode_dsa_mla.py
```

`--profile small` 使用更小形状，`--trials N` 控制数值种子数量。所有 10 个版本默认均运行，任一证明、参照、反例或编译检查失败均以非零状态退出。JSON 分别记录结构/代数结论、浮点警告、随机种子和数值反例。

2026-10-14，Python 3.12 / JAX 0.11.2 / CPU：默认 `glm` 配置、seed=17/18 的独立脚本已全部通过。1 个正例结构等价，4 个正例代数等价且证据复核通过；5 个反例均结构不匹配、代数 unknown，并在两个种子下各有输出反例。正例相对 NumPy 的最大绝对误差为 `2.043e-6`（仅本次采样数据）。

| 反例 | 两个种子中较小的最大输出差异 |
| --- | --- |
| `wrong_scale` | 0.243254 |
| `wrong_rope` | 0.242552 |
| `wrong_selection` | 1.585175 |
| `wrong_pages` | 2.223960 |
| `wrong_value` | 2.153967 |

运行报告保存在 `/tmp/jax-reorder-glm52.json`，不纳入源码。回归和静态检查结果见[完成记录](../exec-plans/completed/glm52-decode-dsa-mla.md)。

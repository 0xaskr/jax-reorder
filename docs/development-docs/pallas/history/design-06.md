# 历史：原语覆盖、参数分类与审计

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

## 6. 原语覆盖、参数分类与审计

建议将覆盖范围固定到项目锁定的 JAX / jaxlib 版本并维护清单，升级时审查差异。清单应包含官方原语、
变换与包装产生的内部原语、嵌套区域，以及 TPU Pallas 的状态、同步和通信原语，不能只扫描
`jax.lax` 的公开导出。用户已明确排除第三方/用户自定义 Primitive、FFI 与 callback；这些入口应明确
拒绝，不列入官方原语覆盖完成度。不能仅因名字相同或抽象 shape 相同就视为同一操作。

每个原语分别记录三个方面，避免把"认识名字"当成"支持语义"：

| 方面 | 必须回答的问题 |
| --- | --- |
| 结构语义 | 完整参数、抽象值、输入输出位置、effects、区域绑定与执行环境能否保真表示？ |
| 数学语义 | 可以展开为哪种公式，适用的 dtype、定义条件和改写规则是什么？未展开时能否保留经过审查的精确算子应用？ |
| 验证证据 | 是否有合法参数变化、独立重排、语义反例与组合测试？追踪、比较、编译和数值执行分别验证了什么？ |

不要求每个原语都先展开成加乘公式才能做结构比较。已识别且语义完整的纯原语，可以在相同输入、参数
和环境下按同一函数应用比较；跨分块使用时还需要其索引、切片或归约语义，不能假设任意原语都逐点执行。
不透明函数的符号残差不能直接作为源程序数学不等价的依据。

effects、随机状态、通信和外部调用不能套用纯函数规则。尚未适配的语义应明确失败，并计为覆盖缺口；
这样的失败不能算作"已完成所有原语支持"。原语覆盖完整也不意味着任意循环、输入相关控制流或任意图
变换都有可完成的数学判定。

inventory 应覆盖三个层次：

1. **身份清单**：普通 JAX、state、Pallas、Mosaic TPU 的真实 Primitive 对象；同名伪造对象拒绝。
2. **参数结构**：参数键、类型、枚举、默认值、嵌套描述符、Ref/区域绑定和 effect 端口角色。
3. **能力状态**：分别记录"能解析""能结构比较""能规范化""能数学转换"，不能因为出现在清单里就
   计为完整支持。

每个 schema 都应有合法变体测试、语义参数变化反例，以及未知字段、未知描述符类型、同名假 primitive
的拒绝测试。除已列字段外，还需审查 `BlockMapping.index_map_out_tree`、`GridMapping` 的索引操作数、
scratch、环境 hooks 以及 Ref transforms。

### 6.1 `pallas_call` 参数分类

| 参数 | 分类与处理 |
| --- | --- |
| `jaxpr` / `out_avals` / `grid_mapping` / `mesh` / `input_output_aliases` | 语义事实，完整解释。 |
| `interpret` | 执行模式。简单例子的 inner jaxpr 相同，不足以支持直接忽略；特殊 InterpretParams 还可能改变计算行为。比较对象和验证运行模式分别记录。 |
| `debug` / `name` | 通常属于诊断和来源信息；不要与 kernel 内实际的调试 effect 混淆。 |
| `cost_estimate` | 成本/执行策略信息，校验 schema 并保留，不当作计算公式。 |
| `metadata` | 通常为诊断信息，但要保留映射、验证保留键及合法性；锁定源码中已有 `mesh_axes` 的特殊处理。 |

锁定版本 `CompilerParams` 的字段建议分组：

| 字段 | 初始处理原则 |
| --- | --- |
| `dimension_semantics` | 调度与顺序语义；默认值也须结合上下文解释。 |
| `collective_id`、`has_side_effects`、`skip_device_barrier`、`allow_collective_id_without_custom_barrier` | 同步、外部 effect 或执行契约；不能直接忽略；隐式 barrier 也须进入分析。 |
| `shape_invariant_numerics` | 数值行为相关，必须保留并审查。 |
| `disable_bounds_checks`、`disable_semaphore_checks` | 影响检查行为；只有相应有效性条件已证明时，才讨论消除差异。 |
| `flags` | 逐键审计，不能把任意配置字典当成无语义。 |
| `allow_input_fusion`、`vmem_limit_bytes`、`internal_scratch_in_bytes`、`serialization_format`、`use_tc_tiling_on_sc`、`needs_layout_passes`、`fuse_transposed_lhs_in_matmul`、`opt_level` | 编译、资源和执行策略字段；初期保留，是否允许差异逐项建立版本化规则，不能仅凭名称断定不影响数值或可执行性。 |

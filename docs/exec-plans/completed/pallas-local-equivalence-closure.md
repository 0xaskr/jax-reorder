# Pallas 设备局部等价比较：缺口修复与循环审查

## 目标与完成条件

按用户要求，修复已确认的六类缺口，再以真实开源 kernel 和独立正反例重新审查；
新发现的问题继续复现、修复、验证。比较对象始终是分布式程序中单个 device 的
Pallas kernel，保留实际使用的设备/轴/实参与通信依赖，不以多卡全局值组装作为
局部判定的额外条件。不扩大到用户排除的 core_map、mpmd_map 或输入输出共享 HBM。

初始基线为 `8190b05`，审查证据位于 `/tmp/jax-reorder-scope-review/review-v2.json`。
纯死计算删减已有九个 Pallas 正例；被覆盖写入的 log 例子没有确定的错误判等结论，
不得恢复此前已撤回的 P1 评级。普通 JAX 的另一条 DCE 路径单独记录，不混入 Pallas 证据。

## 设计与不变量

- 源结构、规范化语义图、数学判定分别提供结果和证据。结构不匹配、拒绝、预算不足、
  CAS 未决与数学不等价严格区分；不能用数值抽样替代数学结论。
- core 负责静态 grid、坐标、状态版本、初始化与协议；数学层消费已证明的局部纯值投影。
  新的设备局部编排必须位于 core 之上，保持 core → algebra → functions 单向依赖。
- 局部输入输出形状和实际使用的轴环境进入契约，不能把完整父函数的全局输出形状
  偷换成局部形状，也不能忽略被 kernel 使用的外层表达式与常量。
- 按实际内存操作选择规范化路径；覆盖写、mask 和未定义值必须在各 rank 下保持同一语义。
- 新增源码准入必须同时建立坐标、值及初始化规则；只放开 adapter 白名单不算完成。
- 条件初始化保留 predicate 下的旧值/未定义事实；互补写可以覆盖，真实读未定义、
  不互补 mask、错误 DMA 时序仍应拒绝。
- 求和域规范化只接受已证明的有界索引映射，保留完整收缩域和原条件；不得为了匹配
  直接去掉偏移、步长、mask 或不透明转换。默认 60 秒不作为掩盖转换开销的手段。

## 工作项

接口决定：沿用现有公共函数，`compare_pallas_kernels(..., equivalence="semantic")`
在完整资源证明后比较局部纯值图；默认仍为局部结构比较。数学及函数入口新增
`scope="device_local"`，默认 `scope="function"` 保持原契约。局部结果显式携带 scope、
局部输入/输出和轴契约以及两侧投影证据；不能凭父图结构相同跳过资源义务。
core 不调用 SymPy，代数编排只把已校验的局部投影交给既有 worker。外层结果包装
不参与局部结论，实际输入来源与端口仍参与；不将外层全局 shape 冒充局部 shape。

一维覆盖写：同一 region 中重复写同一 Ref 必须选择按版本回放的路径，嵌套 region
原有选择规则不变；不能把顺序覆盖的完整窗口送进只接受不相交区间的拼接路径。
lane gather：静态 lane 向量按表达式 DAG 节点计算一次，使用任意精度整数检查每个
节点的 dtype 范围和最终 in-bounds 承诺；不再为每个 lane 重走同一图。预算继续约束
lane 数和唯一节点数，不提高默认值，也不忽略任何索引运算。

完整 paged attention 的性能不变量：Ref 足迹只在 storage 每轴为常量或偏移加独立
单位 local 轴、非单例局部轴全部且唯一出现、范围严格合法时使用精确 Box；stride 和
diagonal 仍走原足迹枚举。tensor bound 范围在进入该节点的 operands 前安装，删除
整 DAG 的重复扫描。`memory._substitute` 复用完全未变的节点，避免广播/视图不断复制
同一大 DAG；替换、边界校验和实际计费仍执行。单侧投影由 6819/6853 降至 2817/2851，
公开 semantic 的真实正例及错误 V×2 候选均在默认 10000 下完成。

- [x] 设备局部入口串联源结构、语义规范化与数学比较；shard_map 正反例不再要求全局组装。
- [x] 修复一维顺序覆盖写的规范化派发，回归多维、部分写、非法资源与初始化路径。
- [x] 补静态 squeezed BlockSpec 以及 stop_gradient、值 slice、reduce_min/and/or 等合法源码。
      用原始 JAX paged attention 继续推进首次准入后的索引、循环、DMA 问题。
- [x] 补远端接收后的数据 mask 条件初始化，覆盖互补写和真实错误反例。
- [x] 补分块求和的索引域重编号，1024 及更大合法实例的正例、加常数/错误权重反例能完成。
- [x] 补真实 attention/RoPE 的数学判定完成率；解决具体瓶颈，不以增大预算作为默认修复。
- [x] 将已有开源候选纳入分层验收矩阵：JAX matmul、FlashAttention、pipeline、paged attention、
      Tokamax 量化 RoPE、vLLM 单步 attention；记录固定来源与原调用/函数抽取边界。
- [x] 交叉审查新增实现，复扫范围内的路径组合；所有确认问题均有修复和回归。
- [x] 完成相关回归、Ruff、格式、类型、tracing、文档链接与差异检查；归档记录。

## 验证方法

复用 `tests/pallas_oracle.py` 和 `tests/pallas_aot.py`。每项规则有正例、实际错误候选、
拒绝路径及必要证据篡改；来源合法性、CPU 完整输出、race 检查和离线 TPU AOT 分列。
只使用现有 Python 3.12 与已有 libtpu 环境；不要求硬件执行或新增依赖。
日志和临时探针写 `/tmp/jax-reorder-local-closure/`，测试保留在现有相关模块。
每轮在最终代码稳定后记录测试 ID，分批去重，保留中途失败和修复后的证据。

## 进展

- 2026-10-22：重新核对当前工作区干净、基线及六类复现，开始分组实施。
- 设备局部入口的普通/shard_map 分块、纯代数、设备编号、远端 peer、父函数输出
  包装及输入来源回归通过。独立复审又修复远端实参重排/重复的分片映射，以及源结构
  shortcut 绕过局部投影契约；12 项入口用例通过。函数/局部广域批次 125 项通过，
  其唯一失败为新增编排模块的导入环，已拆出共用结果类型；架构守卫 4 项复验通过。
- 严格资源证明已成功时，局部数学入口直接使用完整规范化结果；仅在严格路径拒绝后
  才尝试原有 padding 数学义务，不能预先把所有潜在 padding 标为未决。真实 paged
  两种 inline 形式已在公开数学入口得到 normalized 等价，无需不必要的 CAS 查询。
- 独立入口审查确认显式多 core 的局部数学路径完成了协议检查，却丢弃返回的
  `CoreExecutionEvidence` 和进展前提；结果补充两侧 `core_executions`，结构与数学
  路径均保留完整证据及协议 assumptions。不同绑定的结构 shortcut 暂未构造出
  合法错误判等：每个 core 的 native 顺序、唯一输出 writer 和 scratch 初始化仍需证明。

### 远端条件初始化（独立工作项）

设计：数据 masked store 将尚未写入的 lane 保留为条件 `undefined`，只证明 active
incoming 有定义；普通 load、DMA start、自动 staging/输出提交仍逐实际消费域验证。
同一布尔 predicate 及显式 `not` 的互补写可以覆盖，不能猜测不同数据条件之间的关系。
不改变协议的保守足迹、等待拓扑和原始存储身份。正例复用 remote mask fixture，
反例包括不互补写、未初始化读/控制/地址及 DMA 源泄漏/缺 wait；CPU 和 AOT 分开报告。
实现完成：仅在同一个 predicate 下折叠嵌套选择，保留实际消费点的定义义务；
不依赖 CAS 推测不同数据条件之间的关系。实际未定义 load 即使其结果随后被覆盖仍拒绝；
JAX 已在追踪时删除的未使用 load 不在输入 jaxpr 中，不冒称对此作了验证。

远端局部接口：`functionalize_remote_call(..., scope="device_local")` 保留真实全局输入
来源、协议和 staging 证明，按 `(输出端口, device)` 展开完整本地输出，证据的
`local_outputs` 记录同一顺序；不额外要求父函数输出复制一致或组装。默认 `global`
路径保持原契约。复制输出但各 device 值不同的局部正例、错误 peer 和多输出顺序已覆盖。

验证证据位于 `/tmp/jax-reorder-local-closure/remote/`：条件初始化新增 20 项静态
正反例通过；相关远端/ND/重访批次 165 项通过；局部接口另有 4 项通过，完整输入输出
坐标由独立 NumPy 解释器核对。新条件源码的官方 CPU eager/on_wait 加 race 检查 8 项
通过；离线 TPU 2×2×1 AOT 中 5 个源码完成 trace、lower、IR verify、backend compile。
第 6 个 masked-load 源码在当前上游后端 lower 明确不支持，单独记为来源拒绝。
初版二跳 DMA fixture 复用了接收区及 semaphore 代，分别触发真实读写竞态和完成顺序
歧义；最终使用独立接收区和独立 semaphore 对，未放宽生产协议检查。嵌套选择导致的
条件正例 CAS 超时由上述精确同条件折叠修复，没有增大默认超时或预算。
最终相关 mask 批次 112 项通过，文件类型检查为零错误。交叉审查局部入口另发现并由
主线修复两处连接问题：wrapper 输入分片必须按实际调用实参重新绑定（含重复实参），
语义比较不能用相同局部源结构跳过投影契约。对应四项回归已追加到局部入口测试，
该文件共 12 项通过；完整端口数值与证据重放均已验证。批次去重 ID、源文件哈希和
AOT 结果见同目录 `manifest.json`。

### 数学：求和域与真实规模

- 根因：`sum(x.reshape(2,512),axis=(0,1))` 保留 `x[j+512*i]` 的双重有限和；
  ArrayExpr 只接受纯坐标标签，256 项以上不作显式展开，因此完整正例及负例均未决。
- 设计：仅对静态独立矩形域，以严格混合进制系数识别有限索引双射；将相关轴替换成
  一个等长轴，其他读取和标量引用按精确 div/mod 逆映射保留。过程按表达式与轴数计费，
  不按张量元素数枚举，不改输出 shape、原定义条件或内存初始化义务。
- 反例：偏移、非单位步长、不完整覆盖、错误权重、依赖外层轴及嵌套绑定不能被混同；
  大规模负例需要在完整输出上形成严格数学见证，不把泛化 Sum 原子不同当作不等价。
- attention：保留同一个精确 Sum/exp DAG，坐标替换时不做可选的 Piecewise 分配、
  指数符号推理或无必要的绑定复制；真正存在捕获风险时仍重命名。默认限时保持 60 秒。
  大公式负例使用完整输出坐标上的精确 0/1 输入见证，不将数值抽样当作等价证明。
- 闭合有理系数 `exp(有理数)` 使用有预算的稀疏指数环，避免 `exp(-1000000)` 触发
  巨大次数多项式 GCD。共同分母为 D 时，`exp(1/D)` 是超越数，其不同 Laurent
  单项式有理线性无关；交叉乘前证明分母非零。符号指数、非有理指数和不透明函数拒绝。
- 量化 float→integer 保留 dtype 专属的逐元素不透明函数；见证只计算源 dtype 可精确
  表示、且位于目标整数范围的整数值，分数/越界仍未决。Tokamax 反例在 `(4,64)`
  使用全 1 输入得到精确 0/127；原点的 NoPE 通道及 `127/2` 不能作为该 cast 反例。
- 独立审查补两类完整性约束：任何早返回前先预检所有端口的内存初始化；精确见证还
  必须在同一输入下满足两侧所有非空输出、全部坐标的原定义域。普通符号 False 若缺少
  这些全函数前提，只能转精确见证或保留未决。不能用其他输出的 log 无定义输入反驳函数。
- 阶段证据：数学相关首轮 229 项中 228 项通过，1 个旧“十亿项 sum 必须未决”预期
  更新为有预算精确负例并通过；最终专项 64 个去重用例通过（含 5 个真实公共正反例）。
  全域/多输出独立回归 22 项通过；独立审查另核对 36 个求和域、100 个指数环、6 个
  绑定碰撞对照。CPU 独立 NumPy oracle 7 项通过，源码离线 TPU 2×2×1 AOT 11 项
  全部完成 trace/lower/IR verify/backend compile（含 4096 项 sum 正反例），无 TPU 执行。
  最终结果与源哈希写 `/tmp/jax-reorder-local-closure/math/`，不提高默认预算或超时。

### 源码准入与完整 paged attention（源码专项完成）

- 静态 `BlockSpec(None, ...)` 保留原 squeezed 轴元数据，物理窗口仍含宽度为 1 的轴，
  Ref 局部坐标只包含未压缩轴；入口校验二者绑定，不重写源块冒充其他布局。
  补 `stop_gradient`、静态值 slice、`reduce_min/and/or` 的值投影及完整输入定义义务。
  cond 仅允许合法的 weak_type join，dtype/shape/sharding 不放宽；kernel 内默认 JIT
  精确重绑 Ref effects，非默认执行契约仍拒绝。
- 可写 scalar-prefetch 是本次调用的 SMEM 状态，串行 program 保留版本，不能写回
  原 HBM 输入；没有并行所有权证明时拒绝。若 index_map 读取这些 Ref，现有纯静态
  index_map 守卫仍拒绝，不把初始值冻结到全部迭代。该具体来源已通过 TPU AOT，
  属于“合法源码但当前动态地址未证明”，不是编译器拒绝；数据无关 index_map 正例通过。
- 大 Ref 视图足迹仅在每个 storage 轴为常量或 offset + 独立单位 local 轴、完整覆盖
  所有非单例 local 轴、无重复且严格 in-bounds 时直接证明为一个 Box；stride/diagonal
  保留精确逐点 fallback。坐标投影进入 tensor binder 时安装其范围，删除重复整图预扫描；
  无变化的坐标替换保留原 DAG 节点，仍执行替换及检查。没有放大默认预算或降低计费倍率。
- 完整原始 JAX paged attention 固定于 manifest 的 JAX commit
  `32544801e26115ac1794926d027148abf3baf009`，未替换 kernel 算法。
  Q 为 f32 `(1,8,128)`，K/V 为 f32 `(1,8,16,128)`，pages_per_compute_block=8，
  megacore_mode=None；lengths=128、pages=0..7 是显式外层常量。
  inline_seq_dim 两配置比较全部 3 个 kernel 输出，V×2 为反例；未知 lengths/pages
  明确拒绝。原始 page copy 的 K/V→bf16 转换也进入独立 NumPy oracle。
- 单侧分析由 6819/6853 降至 2817/2851；默认 10000 的公共 semantic 正反例通过。
  公共数学入口已越过源码/资源阶段，但首次默认运行仍在 worker 60 秒超时，交数学专项
  继续处理，不能据此把数学项记为已通过。
- 验证目录 `/tmp/jax-reorder-local-closure/source/`：最终共享 DAG 相关回归 239 项通过，
  类型检查零错误；此前源码/视图批次 304 项通过。官方 CPU 完整输出及 race 检查 3 项通过，
  主 AOT 35 项均完成 trace/lower/IR verify/backend compile；另 1 个 mutable-prefetch
  index_map 拒绝源也完成 AOT。均未在 TPU 硬件执行。初版 9 个 AOT 来源几何/stride 拒绝
  与 oracle 漏写 bf16 cast 的失败记录保留，未混入通过数。
- 追加复审确认静态 concatenate 是合法 TPU 源码，现按各输入区间建立精确坐标偏移和
  选择，校验 axis/shape/dtype；初始化只要求当前输出坐标选择的 operand，不加入新
  不透明 CAS 运算。两个 `(8,128)` 张量分别沿轴 0/1 连接，与两段 Ref store 正例和
  错误交换顺序反例均通过公共比较及 CPU 完整输出；6 项专项测试通过。对应 6 个源码
  均完成离线 TPU AOT。lane-gather 向量化另做 10 项新旧解释器差分：6 项精确图一致、
  4 项 overflow/零除/越界拒绝一致；新增与既有布尔/位运算共 14 项回归通过。

- 局部边界复审补齐未使用 mesh 的投影：裸 Pallas call 与外围 shard_map 具有相同局部
  输入时可比较，父输入 `(256,)` / `(512,)` 的差别仍保留在原始来源证据中。只删除未用
  mesh 轴，实际 axis_index 使用的轴名、顺序、大小、种类保留；portable AbstractMesh
  的解释放在 adapter，其他 context/ArrayType 事实不变，remote 全局输入来源及 partition
  不做此投影。默认完整 function 契约仍拒绝不同父输入。公共结构、语义、数学、functions
  正例及错误计算反例、轴范围变化、证据篡改拒绝均覆盖；局部组 19 项及原 scope/remote/
  core 组 108 项通过，类型检查零错误。

- 多输出 tensor 数学构造仅在同一 kernel 调用内共享闭合计算：以有 memo 的完整 DAG
  结构签名、运算参数/类型/输入来源以及所有自由坐标的实际绑定和范围为键。每个输出
  仍独立读取坐标，保留定义域、内存初始化和 padding lowering。单输出不启用新签名
  或计费路径。新增 6 项捕获/参数/来源/读取顺序/义务/预算回归通过；独立审查的
  7 个嵌套绑定与 padding 用例共 87 项检查通过，未改变默认预算与 60 秒限时。

- 缓存预算复核发现首次和复用前各走一次签名 DAG 会使两输出量化 RoPE 回退；现将首次
  签名合成放入已计费的 emit 节点访问，只接受全部已完成子签名，未访问的 lazy 分支仍
  另行计费遍历。真实 wrong_scale 的 signature 访问 1672→836，省 836 次 emit 和
  256 次 reduction 展开，默认总 work 为 9527（禁用缓存为 9796）；6 项专项重验通过。
  并行数学开发期间的 5 用例复验中 RoPE 三项通过，attention 两项 worker 60 秒超时，
  已交数学专项定位；不把这个中间版本记为完整验收。
- 公开局部入口复核复用固定上游原始源码：JAX matmul f32 `(16,256)×(256,256)`、
  block `(8,128)` 的 block_k=128↔256 数学等价，语义图 DIFFERENT；reset_each、
  overwrite、double_state 三个原源码状态变异均得完整数学反例。完整 FlashAttention
  Q `(1,1,128,128)`、K/V `(1,1,256,128)` 的 block_k=128↔256 双入口等价。
  真 pipeline copy i32 `(512,512)`、块 `(128,128)` 的 early_wait 双入口等价，
  wrong_block 数学不等价；missing_input_wait 双入口以真实 DMA hazard 拒绝。
  pipeline matmul f32 `(512,512)` 的 early_wait 在默认 10000 明确预算拒绝，在原
  fixture 已有 100000 预算下双入口等价（数学总 work=43622），不改形状或 worker 限时。
  原测试通过 scope 参数保留 function 覆盖，14 项相关回归通过；大 matmul 的两项公开
  调用结果另有完整 probe 日志。拒绝例与大 pipeline 的前端 canonicalization 较慢，
  保留真实耗时及栈记录，不当作错误等价或硬件性能结果。

### 最终复审与回归进度

- 公开局部 API 的独立跨层复审未发现新的可复现缺陷；核对默认 function 兼容性、
  全部 kernel 输出、实际输入来源、有限设备范围，以及结构/数学两路的资源证据。
- 多输出缓存最终单遍签名版本另行复验 7 个差分案例、87 项检查，全部通过。
- 业务/压力全组完成 364 项通过、2 项失败；失败是中间数学版本引入的 mHC bf16
  定义域回归，经保留小 cast 参数的既有化简修复，两项分别 fresh 复验通过。
  这些历史失败保留在日志，不冒称一次全组全绿。
- attention 新超时已定位为仅绑定变量名不同的定义域误入 128 项 Max 的 Piecewise
  展开；独立探针确认构图约 5.38 秒，绑定名规范化后原定义域与值均一致，域检查约
  0.12 秒。最终数学验收仍需覆盖此修复及完整 paged attention 反例。
- 完整回归发现另一条旧预期仍把同一 program 的顺序完整覆盖写作为非法多写；
  fresh 复现后删除该过时拒绝参数，复用已有同源码的最新版本正例和错误输出反例。
  未初始化、部分初始化、写输入三条真实拒绝保留；4 项 fresh 复验通过。
- 开发期间广域 core 批次完成 3,740 通过、23 失败：16 项是旧进程中的结果类型/adapter
  与新模块混用（core_executions、concatenate），2 项旧断言已更新，1 项 cond 总性
  回归已修复，其余为 attention 两项与 max 两项，均转 fresh 专项处理。完整 JUnit
  及分类保留在 `development-regression-summary.json`，不作最终全绿证据。
- max 两项的精确 extremum 反例本来成立；新完整域检查先化简 quantified Product
  后丢失绑定坐标范围，导致误报未决。现逐一对两侧所有非空输出在 shape bounds 下
  证明 `bounds & original_defined == bounds`，保留原完整域给实际输入见证，任何
  域未知仍不能提升为无条件反例。两个 max、部分域多输出及 cond 共 21 项 fresh
  回归通过；独立最小域对照记录于 `max-regression-review/`。

### 数学最终收口：完整 attention 与严格见证

- 构造保留有限 Sum/Product 与嵌套 Piecewise，坐标替换保持 DAG 共享，避免仅替换坐标
  就隐式分配全部分支。极值仍保留完整操作数；定义域的 alpha 同一性仅重命名绑定，
  不移除任何条件。多输出的地址规范化缓存按完整 Indexed 身份复用。
- 分母规则仅接受已知有限的正实数 exp、正有理数、每个分支均正且有默认分支的
  Piecewise，以及静态非空域上的正项 Sum。仍保留全部 operand 原定义域；空和、零项、
  可正可负项与部分定义的 log/sqrt 不能因此变成无条件合法。这使三角 mask 分母的
  非零性无需按外层坐标展开。独立 12 个公开边界探针及新增 9 项持久回归均通过。
- 精确输入搜索补“仅一个输入端口全 1、其余全 0”候选，搜索顺序仅依赖输入数和是否
  含已注册 cast，不引用 kernel 名、形状或 Q/K/V 角色。两侧所有非空输出、全部坐标的
  原定义域必须先在同一输入下成立；只有可信投影给出有限值证书时，才在随后值求值中
  使用有限零乘法短路。域检查本身不使用这个短路。未知 opaque、非有限常量和无法精确
  表示的 cast 仍拒绝，候选相等不能证明等价，实际访问照常计费。
- 页选择仅做局部布尔互补、NNF 和由整数区间证明的恒真/恒假；先检查 Piecewise guard
  再访问可达值。展开的欧几里得余数只在整数分子、正整数分母与精确匹配系数下恢复，
  额外余项仍计入范围。独立 1,152 组逐点区间对照通过，新增系数/偏移与拒绝回归通过。
- 独立审查确认有限值证书不能替代输入范围证明；零乘法被跳过的子树现另做有缓存的
  bounds-only 检查，Sum/Product 保留词法范围，每个新 Indexed 范围查询独立计费。
  直接越界读取及嵌套 Sum 越界均保留拒绝；四项有效/拒绝回归通过。
- 公开默认预算/60 秒数学六例和新增七项见证守卫最终通过
  （`math/real-six-bounds-final.xml`，13 项/48.67 秒）：vLLM attention identity_mul
  5.252 秒、wrong_scale 5.630 秒；Tokamax 量化 RoPE identity_mul 6.651 秒、wrong_sign
  5.993 秒、wrong_scale 5.565 秒；完整 JAX paged attention V×2 18.646 秒。后者在原
  3,437 剩余数学预算内用 3,256 工作量验证精确输入 `only_input_2`、输出 `1 != 2`，
  保留全部三个输出义务。计时是本地比较时间，不是硬件执行或性能结论。
- mHC bf16 两项最终 fresh 复验通过（`math/business-final.xml`，2 项/67.25 秒），
  包括各自 forward/JVP/VJP；后续 bounds-only 改动不经过该正例路径。数学模块类型
  检查零错误，Ruff/format/diff 检查通过。最终独立审查的 12 个非有限/opaque/部分域/
  原越界反例与 8 个 Sum/Product 同名 shadow、共享 read 不同范围、bound 后 free 坐标
  等词法探针全部通过，无剩余健全性阻塞。数学生产于 2026-10-21 18:10:43 UTC 冻结，
  哈希和唯一测试 ID 见 `math/final-math-manifest.json`；最终受影响回归与静态检查由主线继续。

### 冻结后的统一回归

- 数学生产冻结后，最后独立 20 项越界、有限性、原定义域与 Sum/Product 绑定范围
  检查全部通过。Ruff 检查通过、399 个 Python 文件格式一致、全库 basedpyright
  零错误/警告，tracing 示例报告 equivalent。
- 最终按文件分片：全部代数模块与局部值、padding、functions、包分层为三个不重叠
  分片；公开 Pallas 代数、语义切片和源码/缓存接入另作第四片。每片保存 JUnit 与
  src/tests Python 文件的运行前后 SHA-256，验证期间不改生产或测试。
- 四片分别为 401、355、256、276 项通过，合计 **1,288 个不同测试 ID**，无失败、
  错误或跳过；各片运行前后及片间的 227 个 Python 文件 SHA-256 完全一致。
  pytest 耗时分别为 462.76、581.55、544.99、506.12 秒，不能相加当作并行墙钟时间。
  完整清单及哈希见 `/tmp/jax-reorder-local-closure/final-validation-manifest.json`。
  前述开发期间失败批次保留，不混入这次冻结验收，也不宣称全库单次 pytest 全绿。

同一组 26 个文件可用以下命令重新执行；本次为并行分片运行：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m pytest -q \
  tests/test_algebra*.py tests/test_functions.py tests/test_package_layout.py \
  tests/test_pallas_local_values.py tests/test_pallas_padding_algebra.py \
  tests/test_pallas_algebra.py tests/test_pallas_semantic_slice.py \
  tests/test_pallas_source_coverage.py
```

## 结论与保留边界

六项已确认缺口及后续复审发现的问题均已修复，并补入持久回归。最终公开入口与独立
边界复审没有发现新的可复现错误判定。真实 JAX matmul、FlashAttention、pipeline、
paged attention、Tokamax 量化 RoPE、vLLM attention 均保留固定源码及正反例证据。

这不扩大为任意合法 Pallas 程序都可判定：未证明的动态地址、数据路由/终止、未知
布局/effect 仍拒绝，规模预算和复杂 CAS 仍可能未决。大 pipeline 的默认预算拒绝也
明确保留；`core_map`、`mpmd_map`、输入输出共享 HBM 与全局多设备值组装仍不在本轮范围。
拒绝、超时和图不同均不能单独推出数学不等价。完整边界由
[验证入口](../../design-docs/current/pallas/validation.md)维护；没有新增依赖或 TPU 实机结论。

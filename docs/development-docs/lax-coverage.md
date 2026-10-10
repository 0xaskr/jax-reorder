# CPU lax 原语清单与验证绑定

基线 JAX 0.11.2；`jax.lax` 与公开子模块 `jax.lax.linalg` 共 171 个去重的公开 Primitive 对象。
当前有 182 个原语及参数用例，额外追踪出 11 个辅助原语。

该清单由 `uv run python tests/lax_inventory.py --write` 重新追踪生成。它记录测试绑定关系；只有完整 `uv run pytest` 通过才构成 CPU 执行证据。脚本本身不运行全部计算。测试会阻止清单漂移或漏掉公开原语。

原语在 CPU 支持的全部合法参数、dtype、shape 都属于目标域；表中的代表值不构成白名单，也不宣称穷举该域。参数用带类型的完整结构编码，子图递归处理。[设计、适配与审查](bfs-reordering.md)解释具体规则和边界。

## 公开原语

169 个公开原语有 CPU 执行用例；另有 stage 的追踪消去检查和 ragged_all_to_all 的 CPU 拒绝检查。后两者不计为 IR 执行通过。

| 实际 primitive | 公开导出 | 固定用例 | 验证方式 |
| --- | --- | --- | --- |
| `abs` | `jax.lax.abs_p` | `abs` | CPU 执行、结构比较、独立分支换序 |
| `acos` | `jax.lax.acos_p` | `acos` | CPU 执行、结构比较、独立分支换序 |
| `acosh` | `jax.lax.acosh_p` | `acosh` | CPU 执行、结构比较、独立分支换序 |
| `add` | `jax.lax.add_p` | `add` | CPU 执行、结构比较、独立分支换序 |
| `after_all` | `jax.lax.after_all_p` | `after_all` | CPU 执行、结构比较、独立分支换序 |
| `all_gather` | `jax.lax.all_gather_p` | `all_gather` | CPU 执行、结构比较、独立分支换序 |
| `all_to_all` | `jax.lax.all_to_all_p` | `all_to_all` | CPU 执行、结构比较、独立分支换序 |
| `and` | `jax.lax.and_p` | `and` | CPU 执行、结构比较、独立分支换序 |
| `approx_top_k` | `jax.lax.approx_top_k_p` | `approx_top_k` | CPU 执行、结构比较、独立分支换序 |
| `argmax` | `jax.lax.argmax_p` | `argmax` | CPU 执行、结构比较、独立分支换序 |
| `argmin` | `jax.lax.argmin_p` | `argmin` | CPU 执行、结构比较、独立分支换序 |
| `asin` | `jax.lax.asin_p` | `asin` | CPU 执行、结构比较、独立分支换序 |
| `asinh` | `jax.lax.asinh_p` | `asinh` | CPU 执行、结构比较、独立分支换序 |
| `atan` | `jax.lax.atan_p` | `atan` | CPU 执行、结构比较、独立分支换序 |
| `atan2` | `jax.lax.atan2_p` | `atan2` | CPU 执行、结构比较、独立分支换序 |
| `atanh` | `jax.lax.atanh_p` | `atanh` | CPU 执行、结构比较、独立分支换序 |
| `axis_index` | `jax.lax.axis_index_p` | `axis_index` | CPU 执行、结构比较、独立分支换序 |
| `bessel_i0e` | `jax.lax.bessel_i0e_p` | `bessel_i0e` | CPU 执行、结构比较、独立分支换序 |
| `bessel_i1e` | `jax.lax.bessel_i1e_p` | `bessel_i1e` | CPU 执行、结构比较、独立分支换序 |
| `bitcast_convert_type` | `jax.lax.bitcast_convert_type_p` | `bitcast_convert_type` | CPU 执行、结构比较、独立分支换序 |
| `broadcast_in_dim` | `jax.lax.broadcast_in_dim_p` | `broadcast_in_dim` | CPU 执行、结构比较、独立分支换序 |
| `cbrt` | `jax.lax.cbrt_p` | `cbrt` | CPU 执行、结构比较、独立分支换序 |
| `ceil` | `jax.lax.ceil_p` | `ceil` | CPU 执行、结构比较、独立分支换序 |
| `cholesky` | `jax.lax.linalg.cholesky_p` | `cholesky` | CPU 执行、结构比较、独立分支换序 |
| `cholesky_update` | `jax.lax.linalg.cholesky_update_p` | `cholesky_update` | CPU 执行、结构比较、独立分支换序 |
| `clamp` | `jax.lax.clamp_p` | `clamp` | CPU 执行、结构比较、独立分支换序 |
| `clz` | `jax.lax.clz_p` | `clz` | CPU 执行、结构比较、独立分支换序 |
| `complex` | `jax.lax.complex_p` | `complex` | CPU 执行、结构比较、独立分支换序 |
| `concatenate` | `jax.lax.concatenate_p` | `concatenate` | CPU 执行、结构比较、独立分支换序 |
| `cond` | `jax.lax.cond_p` | `cond` | CPU 执行、结构比较、独立分支换序 |
| `conj` | `jax.lax.conj_p` | `conj` | CPU 执行、结构比较、独立分支换序 |
| `conv_general_dilated` | `jax.lax.conv_general_dilated_p` | `conv_general_dilated` | CPU 执行、结构比较、独立分支换序 |
| `convert_element_type` | `jax.lax.convert_element_type_p` | `convert_element_type` | CPU 执行、结构比较、独立分支换序 |
| `copy` | `jax.lax.copy_p` | `copy` | CPU 执行、结构比较、独立分支换序 |
| `cos` | `jax.lax.cos_p` | `cos` | CPU 执行、结构比较、独立分支换序 |
| `cosh` | `jax.lax.cosh_p` | `cosh` | CPU 执行、结构比较、独立分支换序 |
| `create_token` | `jax.lax.create_token_p` | `create_token` | CPU 执行、结构比较、独立分支换序 |
| `cumlogsumexp` | `jax.lax.cumlogsumexp_p` | `cumlogsumexp` | CPU 执行、结构比较、独立分支换序 |
| `cummax` | `jax.lax.cummax_p` | `cummax` | CPU 执行、结构比较、独立分支换序 |
| `cummin` | `jax.lax.cummin_p` | `cummin` | CPU 执行、结构比较、独立分支换序 |
| `cumprod` | `jax.lax.cumprod_p` | `cumprod` | CPU 执行、结构比较、独立分支换序 |
| `cumsum` | `jax.lax.cumsum_p` | `cumsum` | CPU 执行、结构比较、独立分支换序 |
| `custom_linear_solve` | `jax.lax.linear_solve_p` | `custom_linear_solve` | CPU 执行、结构比较、独立分支换序 |
| `device_put` | `jax.lax.device_put_p` | `device_put` | CPU 执行、结构比较、独立分支换序 |
| `digamma` | `jax.lax.digamma_p` | `digamma` | CPU 执行、结构比较、独立分支换序 |
| `div` | `jax.lax.div_p` | `div` | CPU 执行、结构比较、独立分支换序 |
| `dot_general` | `jax.lax.dot_general_p` | `dot_general` | CPU 执行、结构比较、独立分支换序 |
| `dynamic_slice` | `jax.lax.dynamic_slice_p` | `dynamic_slice` | CPU 执行、结构比较、独立分支换序 |
| `dynamic_update_slice` | `jax.lax.dynamic_update_slice_p` | `dynamic_update_slice` | CPU 执行、结构比较、独立分支换序 |
| `eig` | `jax.lax.linalg.eig_p` | `eig` | CPU 执行、结构比较、独立分支换序 |
| `eigh` | `jax.lax.linalg.eigh_p` | `eigh` | CPU 执行、结构比较、独立分支换序 |
| `eq` | `jax.lax.eq_p` | `eq` | CPU 执行、结构比较、独立分支换序 |
| `eq_to` | `jax.lax.eq_to_p` | `eq_to` | CPU 执行、结构比较、独立分支换序 |
| `erf` | `jax.lax.erf_p` | `erf` | CPU 执行、结构比较、独立分支换序 |
| `erf_inv` | `jax.lax.erf_inv_p` | `erf_inv` | CPU 执行、结构比较、独立分支换序 |
| `erfc` | `jax.lax.erfc_p` | `erfc` | CPU 执行、结构比较、独立分支换序 |
| `exp` | `jax.lax.exp_p` | `exp` | CPU 执行、结构比较、独立分支换序 |
| `exp2` | `jax.lax.exp2_p` | `exp2` | CPU 执行、结构比较、独立分支换序 |
| `expm1` | `jax.lax.expm1_p` | `expm1` | CPU 执行、结构比较、独立分支换序 |
| `fft` | `jax.lax.fft_p` | `fft` | CPU 执行、结构比较、独立分支换序 |
| `floor` | `jax.lax.floor_p` | `floor` | CPU 执行、结构比较、独立分支换序 |
| `gather` | `jax.lax.gather_p` | `gather` | CPU 执行、结构比较、独立分支换序 |
| `ge` | `jax.lax.ge_p` | `ge` | CPU 执行、结构比较、独立分支换序 |
| `gt` | `jax.lax.gt_p` | `gt` | CPU 执行、结构比较、独立分支换序 |
| `hessenberg` | `jax.lax.linalg.hessenberg_p` | `hessenberg` | CPU 执行、结构比较、独立分支换序 |
| `householder_product` | `jax.lax.linalg.householder_product_p` | `householder_product` | CPU 执行、结构比较、独立分支换序 |
| `igamma` | `jax.lax.igamma_p` | `igamma` | CPU 执行、结构比较、独立分支换序 |
| `igamma_grad_a` | `jax.lax.igamma_grad_a_p` | `igamma_grad_a` | CPU 执行、结构比较、独立分支换序 |
| `igammac` | `jax.lax.igammac_p` | `igammac` | CPU 执行、结构比较、独立分支换序 |
| `imag` | `jax.lax.imag_p` | `imag` | CPU 执行、结构比较、独立分支换序 |
| `integer_pow` | `jax.lax.integer_pow_p` | `integer_pow` | CPU 执行、结构比较、独立分支换序 |
| `iota` | `jax.lax.iota_p` | `iota` | CPU 执行、结构比较、独立分支换序 |
| `is_finite` | `jax.lax.is_finite_p` | `is_finite` | CPU 执行、结构比较、独立分支换序 |
| `le` | `jax.lax.le_p` | `le` | CPU 执行、结构比较、独立分支换序 |
| `le_to` | `jax.lax.le_to_p` | `le_to` | CPU 执行、结构比较、独立分支换序 |
| `lgamma` | `jax.lax.lgamma_p` | `lgamma` | CPU 执行、结构比较、独立分支换序 |
| `log` | `jax.lax.log_p` | `log` | CPU 执行、结构比较、独立分支换序 |
| `log1p` | `jax.lax.log1p_p` | `log1p` | CPU 执行、结构比较、独立分支换序 |
| `log2` | `jax.lax.log2_p` | `log2` | CPU 执行、结构比较、独立分支换序 |
| `logistic` | `jax.lax.logistic_p` | `logistic` | CPU 执行、结构比较、独立分支换序 |
| `lt` | `jax.lax.lt_p` | `lt` | CPU 执行、结构比较、独立分支换序 |
| `lt_to` | `jax.lax.lt_to_p` | `lt_to` | CPU 执行、结构比较、独立分支换序 |
| `lu` | `jax.lax.linalg.lu_p` | `lu` | CPU 执行、结构比较、独立分支换序 |
| `lu_pivots_to_permutation` | `jax.lax.linalg.lu_pivots_to_permutation_p` | `lu_pivots_to_permutation` | CPU 执行、结构比较、独立分支换序 |
| `max` | `jax.lax.max_p` | `max` | CPU 执行、结构比较、独立分支换序 |
| `min` | `jax.lax.min_p` | `min` | CPU 执行、结构比较、独立分支换序 |
| `mul` | `jax.lax.mul_p` | `mul` | CPU 执行、结构比较、独立分支换序 |
| `mulhi` | `jax.lax.mulhi_p` | `mulhi` | CPU 执行、结构比较、独立分支换序 |
| `ne` | `jax.lax.ne_p` | `ne` | CPU 执行、结构比较、独立分支换序 |
| `neg` | `jax.lax.neg_p` | `neg` | CPU 执行、结构比较、独立分支换序 |
| `nextafter` | `jax.lax.nextafter_p` | `nextafter` | CPU 执行、结构比较、独立分支换序 |
| `not` | `jax.lax.not_p` | `not` | CPU 执行、结构比较、独立分支换序 |
| `one_minus_square` | `jax.lax.one_minus_square_p` | `one_minus_square` | CPU 执行、结构比较、独立分支换序 |
| `optimization_barrier` | `jax.lax.optimization_barrier_p` | `optimization_barrier` | CPU 执行、结构比较、独立分支换序 |
| `or` | `jax.lax.or_p` | `or` | CPU 执行、结构比较、独立分支换序 |
| `ormqr` | `jax.lax.linalg.ormqr_p` | `ormqr` | CPU 执行、结构比较、独立分支换序 |
| `pad` | `jax.lax.pad_p` | `pad` | CPU 执行、结构比较、独立分支换序 |
| `pmax` | `jax.lax.pmax_p` | `pmax` | CPU 执行、结构比较、独立分支换序 |
| `pmin` | `jax.lax.pmin_p` | `pmin` | CPU 执行、结构比较、独立分支换序 |
| `polygamma` | `jax.lax.polygamma_p` | `polygamma` | CPU 执行、结构比较、独立分支换序 |
| `population_count` | `jax.lax.population_count_p` | `population_count` | CPU 执行、结构比较、独立分支换序 |
| `pow` | `jax.lax.pow_p` | `pow` | CPU 执行、结构比较、独立分支换序 |
| `ppermute` | `jax.lax.ppermute_p` | `ppermute` | CPU 执行、结构比较、独立分支换序 |
| `psum` | `jax.lax.psum_p` | `psum` | CPU 执行、结构比较、独立分支换序 |
| `qr` | `jax.lax.linalg.qr_p` | `qr` | CPU 执行、结构比较、独立分支换序 |
| `ragged_all_to_all` | `jax.lax.ragged_all_to_all_p` | `ragged_all_to_all` | 结构处理通过；断言 XLA:CPU 明确拒绝执行 |
| `real` | `jax.lax.real_p` | `real` | CPU 执行、结构比较、独立分支换序 |
| `reduce` | `jax.lax.reduce_p` | `reduce` | CPU 执行、结构比较、独立分支换序 |
| `reduce_and` | `jax.lax.reduce_and_p` | `reduce_and` | CPU 执行、结构比较、独立分支换序 |
| `reduce_max` | `jax.lax.reduce_max_p` | `reduce_max` | CPU 执行、结构比较、独立分支换序 |
| `reduce_min` | `jax.lax.reduce_min_p` | `reduce_min` | CPU 执行、结构比较、独立分支换序 |
| `reduce_or` | `jax.lax.reduce_or_p` | `reduce_or` | CPU 执行、结构比较、独立分支换序 |
| `reduce_precision` | `jax.lax.reduce_precision_p` | `reduce_precision` | CPU 执行、结构比较、独立分支换序 |
| `reduce_prod` | `jax.lax.reduce_prod_p` | `reduce_prod` | CPU 执行、结构比较、独立分支换序 |
| `reduce_sum` | `jax.lax.reduce_sum_p` | `reduce_sum` | CPU 执行、结构比较、独立分支换序 |
| `reduce_window` | `jax.lax.reduce_window_p` | `reduce_window` | CPU 执行、结构比较、独立分支换序 |
| `reduce_window_max` | `jax.lax.reduce_window_max_p` | `reduce_window_max` | CPU 执行、结构比较、独立分支换序 |
| `reduce_window_min` | `jax.lax.reduce_window_min_p` | `reduce_window_min` | CPU 执行、结构比较、独立分支换序 |
| `reduce_window_sum` | `jax.lax.reduce_window_sum_p` | `reduce_window_sum` | CPU 执行、结构比较、独立分支换序 |
| `reduce_xor` | `jax.lax.reduce_xor_p` | `reduce_xor` | CPU 执行、结构比较、独立分支换序 |
| `regularized_incomplete_beta` | `jax.lax.regularized_incomplete_beta_p` | `regularized_incomplete_beta` | CPU 执行、结构比较、独立分支换序 |
| `rem` | `jax.lax.rem_p` | `rem` | CPU 执行、结构比较、独立分支换序 |
| `reshape` | `jax.lax.reshape_p` | `reshape` | CPU 执行、结构比较、独立分支换序 |
| `rev` | `jax.lax.rev_p` | `rev` | CPU 执行、结构比较、独立分支换序 |
| `rng_bit_generator` | `jax.lax.rng_bit_generator_p` | `rng_bit_generator` | CPU 执行、结构比较、独立分支换序 |
| `rng_uniform` | `jax.lax.rng_uniform_p` | `rng_uniform` | CPU 执行、结构比较、独立分支换序 |
| `round` | `jax.lax.round_p` | `round` | CPU 执行、结构比较、独立分支换序 |
| `rsqrt` | `jax.lax.rsqrt_p` | `rsqrt` | CPU 执行、结构比较、独立分支换序 |
| `scan` | `jax.lax.scan_p` | `scan` | CPU 执行、结构比较、独立分支换序 |
| `scatter` | `jax.lax.scatter_p` | `scatter` | CPU 执行、结构比较、独立分支换序 |
| `scatter-add` | `jax.lax.scatter_add_p` | `scatter-add` | CPU 执行、结构比较、独立分支换序 |
| `scatter-max` | `jax.lax.scatter_max_p` | `scatter-max` | CPU 执行、结构比较、独立分支换序 |
| `scatter-min` | `jax.lax.scatter_min_p` | `scatter-min` | CPU 执行、结构比较、独立分支换序 |
| `scatter-mul` | `jax.lax.scatter_mul_p` | `scatter-mul` | CPU 执行、结构比较、独立分支换序 |
| `scatter-sub` | `jax.lax.scatter_sub_p` | `scatter-sub` | CPU 执行、结构比较、独立分支换序 |
| `schur` | `jax.lax.linalg.schur_p` | `schur` | CPU 执行、结构比较、独立分支换序 |
| `select_and_gather_add` | `jax.lax.select_and_gather_add_p` | `select_and_gather_add` | CPU 执行、结构比较、独立分支换序 |
| `select_and_scatter` | `jax.lax.select_and_scatter_p` | `select_and_scatter` | CPU 执行、结构比较、独立分支换序 |
| `select_and_scatter_add` | `jax.lax.select_and_scatter_add_p` | `select_and_scatter_add` | CPU 执行、结构比较、独立分支换序 |
| `select_n` | `jax.lax.select_n_p` | `select_n` | CPU 执行、结构比较、独立分支换序 |
| `sharding_constraint` | `jax.lax.sharding_constraint_p` | `sharding_constraint` | CPU 执行、结构比较、独立分支换序 |
| `shift_left` | `jax.lax.shift_left_p` | `shift_left` | CPU 执行、结构比较、独立分支换序 |
| `shift_right_arithmetic` | `jax.lax.shift_right_arithmetic_p` | `shift_right_arithmetic` | CPU 执行、结构比较、独立分支换序 |
| `shift_right_logical` | `jax.lax.shift_right_logical_p` | `shift_right_logical` | CPU 执行、结构比较、独立分支换序 |
| `sign` | `jax.lax.sign_p` | `sign` | CPU 执行、结构比较、独立分支换序 |
| `sin` | `jax.lax.sin_p` | `sin` | CPU 执行、结构比较、独立分支换序 |
| `sinh` | `jax.lax.sinh_p` | `sinh` | CPU 执行、结构比较、独立分支换序 |
| `slice` | `jax.lax.slice_p` | `slice` | CPU 执行、结构比较、独立分支换序 |
| `sort` | `jax.lax.sort_p` | `sort` | CPU 执行、结构比较、独立分支换序 |
| `split` | `jax.lax.split_p` | `split` | CPU 执行、结构比较、独立分支换序 |
| `sqrt` | `jax.lax.sqrt_p` | `sqrt` | CPU 执行、结构比较、独立分支换序 |
| `square` | `jax.lax.square_p` | `square` | CPU 执行、结构比较、独立分支换序 |
| `squeeze` | `jax.lax.squeeze_p` | `squeeze` | CPU 执行、结构比较、独立分支换序 |
| `stack` | `jax.lax.stack_p` | `stack` | CPU 执行、结构比较、独立分支换序 |
| `stage` | `jax.lax.stage_p` | `stage` | 公开调用被 JAX 追踪消去；不计 IR 执行覆盖 |
| `stop_gradient` | `jax.lax.stop_gradient_p` | `stop_gradient` | CPU 执行、结构比较、独立分支换序 |
| `sub` | `jax.lax.sub_p` | `sub` | CPU 执行、结构比较、独立分支换序 |
| `svd` | `jax.lax.linalg.svd_p` | `svd` | CPU 执行、结构比较、独立分支换序 |
| `symmetric_product` | `jax.lax.linalg.symmetric_product_p` | `symmetric_product` | CPU 执行、结构比较、独立分支换序 |
| `tan` | `jax.lax.tan_p` | `tan` | CPU 执行、结构比较、独立分支换序 |
| `tanh` | `jax.lax.tanh_p` | `tanh` | CPU 执行、结构比较、独立分支换序 |
| `tile` | `jax.lax.tile_p` | `tile` | CPU 执行、结构比较、独立分支换序 |
| `top_k` | `jax.lax.top_k_p` | `top_k` | CPU 执行、结构比较、独立分支换序 |
| `transpose` | `jax.lax.transpose_p` | `transpose` | CPU 执行、结构比较、独立分支换序 |
| `triangular_solve` | `jax.lax.linalg.triangular_solve_p` | `triangular_solve` | CPU 执行、结构比较、独立分支换序 |
| `tridiagonal` | `jax.lax.linalg.tridiagonal_p` | `tridiagonal` | CPU 执行、结构比较、独立分支换序 |
| `tridiagonal_solve` | `jax.lax.linalg.tridiagonal_solve_p` | `tridiagonal_solve` | CPU 执行、结构比较、独立分支换序 |
| `unstack` | `jax.lax.unstack_p` | `unstack` | CPU 执行、结构比较、独立分支换序 |
| `while` | `jax.lax.while_p` | `while` | CPU 执行、结构比较、独立分支换序 |
| `xor` | `jax.lax.xor_p` | `xor` | CPU 执行、结构比较、独立分支换序 |
| `zeta` | `jax.lax.zeta_p` | `zeta` | CPU 执行、结构比较、独立分支换序 |

## 包装函数实际产生的辅助原语

| primitive | 命中的固定用例 |
| --- | --- |
| `composite` | `composite`, `composite_attributes` |
| `dce_sink` | `dce_sink`, `dce_sink_mlir` |
| `empty` | `empty` |
| `jit` | `fft` |
| `pbroadcast` | `pbroadcast` |
| `platform_index` | `platform_index` |
| `psum_invariant` | `psum_invariant` |
| `ragged_dot_general` | `ragged_dot_general` |
| `reduce_scatter` | `reduce_scatter` |
| `scaled_dot` | `scaled_dot` |
| `shard_map` | `psum`, `pmin`, `pmax`, `axis_index`, `all_gather`, `all_to_all`, `ppermute`, `ragged_all_to_all`, `pbroadcast`, `reduce_scatter`, `psum_invariant` |

`pbroadcast` 的用例显式断言 CPU 缺少 lowering。`psend`、`precv` 的独立测试断言 JAX 仅提供 GPU 实现。它们不计为 CPU 执行覆盖。`dce_sink` 同时检查 `prevent_mlir_dce=False/True`；`psum` 检查 `check_vma=False/True`，后者实际产生 `psum_invariant`。

## 固定参数与输入签名

全部静态参数及数值输入保存在 [lax_cases.py](../../tests/lax_cases.py)。以下签名对应默认 X64 关闭的原语用例；[test_semantics.py](../../tests/test_semantics.py)另在 X64 开启时覆盖 19 种 dtype × 5 种 shape，以及 key dtype 和捕获 key 数组。

| 用例 | 输入叶子的 dtype / shape |
| --- | --- |
| `abs` | `float32(3,)` |
| `acos` | `float32(3,)` |
| `acosh` | `float32(3,)` |
| `asin` | `float32(3,)` |
| `asinh` | `float32(3,)` |
| `atan` | `float32(3,)` |
| `atanh` | `float32(3,)` |
| `bessel_i0e` | `float32(3,)` |
| `bessel_i1e` | `float32(3,)` |
| `cbrt` | `float32(3,)` |
| `ceil` | `float32(3,)` |
| `cos` | `float32(3,)` |
| `cosh` | `float32(3,)` |
| `digamma` | `float32(3,)` |
| `erf` | `float32(3,)` |
| `erf_inv` | `float32(3,)` |
| `erfc` | `float32(3,)` |
| `exp` | `float32(3,)` |
| `exp2` | `float32(3,)` |
| `expm1` | `float32(3,)` |
| `floor` | `float32(3,)` |
| `is_finite` | `float32(3,)` |
| `lgamma` | `float32(3,)` |
| `log` | `float32(3,)` |
| `log1p` | `float32(3,)` |
| `log2` | `float32(3,)` |
| `logistic` | `float32(3,)` |
| `neg` | `float32(3,)` |
| `one_minus_square` | `float32(3,)` |
| `round` | `float32(3,)` |
| `rsqrt` | `float32(3,)` |
| `sign` | `float32(3,)` |
| `sin` | `float32(3,)` |
| `sinh` | `float32(3,)` |
| `sqrt` | `float32(3,)` |
| `square` | `float32(3,)` |
| `tan` | `float32(3,)` |
| `tanh` | `float32(3,)` |
| `add` | `float32(3,)`, `float32(3,)` |
| `atan2` | `float32(3,)`, `float32(3,)` |
| `div` | `float32(3,)`, `float32(3,)` |
| `igamma` | `float32(3,)`, `float32(3,)` |
| `igamma_grad_a` | `float32(3,)`, `float32(3,)` |
| `igammac` | `float32(3,)`, `float32(3,)` |
| `max` | `float32(3,)`, `float32(3,)` |
| `min` | `float32(3,)`, `float32(3,)` |
| `mul` | `float32(3,)`, `float32(3,)` |
| `nextafter` | `float32(3,)`, `float32(3,)` |
| `pow` | `float32(3,)`, `float32(3,)` |
| `rem` | `float32(3,)`, `float32(3,)` |
| `sub` | `float32(3,)`, `float32(3,)` |
| `zeta` | `float32(3,)`, `float32(3,)` |
| `eq` | `float32(3,)`, `float32(3,)` |
| `ge` | `float32(3,)`, `float32(3,)` |
| `gt` | `float32(3,)`, `float32(3,)` |
| `le` | `float32(3,)`, `float32(3,)` |
| `lt` | `float32(3,)`, `float32(3,)` |
| `ne` | `float32(3,)`, `float32(3,)` |
| `clz` | `uint32(3,)` |
| `population_count` | `uint32(3,)` |
| `and` | `int32(3,)`, `int32(3,)` |
| `or` | `int32(3,)`, `int32(3,)` |
| `xor` | `int32(3,)`, `int32(3,)` |
| `not` | `int32(3,)` |
| `mulhi` | `int32(3,)`, `int32(3,)` |
| `shift_left` | `int32(3,)`, `int32(3,)` |
| `shift_right_arithmetic` | `int32(3,)`, `int32(3,)` |
| `shift_right_logical` | `int32(3,)`, `int32(3,)` |
| `conj` | `complex64(3,)` |
| `imag` | `complex64(3,)` |
| `real` | `complex64(3,)` |
| `complex` | `float32(3,)`, `float32(3,)` |
| `regularized_incomplete_beta` | `float32(3,)`, `float32(3,)`, `float32(3,)` |
| `polygamma` | `float32(3,)`, `float32(3,)` |
| `integer_pow` | `float32(3,)` |
| `clamp` | `float32(3,)`, `float32(3,)`, `float32(3,)` |
| `convert_element_type` | `float32(3,)` |
| `bitcast_convert_type` | `float32(3,)` |
| `reduce_precision` | `float32(3,)` |
| `broadcast_in_dim` | `float32(3,)` |
| `reshape` | `float32(3,)` |
| `transpose` | `float32(2, 2)` |
| `squeeze` | `float32(1, 3)` |
| `concatenate` | `float32(3,)`, `float32(3,)` |
| `stack` | `float32(3,)`, `float32(3,)` |
| `unstack` | `float32(2, 2)` |
| `split` | `float32(3,)` |
| `tile` | `float32(3,)` |
| `rev` | `float32(3,)` |
| `slice` | `float32(3,)` |
| `dynamic_slice` | `float32(3,)`, `int32()` |
| `dynamic_update_slice` | `float32(3,)`, `float32(1,)`, `int32()` |
| `pad` | `float32(3,)`, `float32()` |
| `iota` | 无输入 |
| `select_n` | `int32(3,)`, `float32(3,)`, `float32(3,)` |
| `reduce_sum` | `float32(2, 2)` |
| `reduce_prod` | `float32(2, 2)` |
| `reduce_max` | `float32(2, 2)` |
| `reduce_min` | `float32(2, 2)` |
| `reduce_and` | `int32(3,)` |
| `reduce_or` | `int32(3,)` |
| `reduce_xor` | `int32(3,)` |
| `cumlogsumexp` | `float32(3,)` |
| `cummax` | `float32(3,)` |
| `cummin` | `float32(3,)` |
| `cumprod` | `float32(3,)` |
| `cumsum` | `float32(3,)` |
| `argmax` | `float32(3,)` |
| `argmin` | `float32(3,)` |
| `dot_general` | `float32(2, 2)`, `float32(2, 2)` |
| `conv_general_dilated` | `float32(1, 4, 1)`, `float32(2, 1, 2)` |
| `fft` | `complex64(3,)` |
| `top_k` | `float32(3,)` |
| `approx_top_k` | `float32(3,)` |
| `sort` | `float32(3,)`, `int32(3,)` |
| `copy` | `float32(3,)` |
| `stage` | `float32(3,)` |
| `stop_gradient` | `float32(3,)` |
| `optimization_barrier` | `float32(3,)`, `int32(3,)` |
| `rng_bit_generator` | `uint32(4,)` |
| `rng_uniform` | `float32()`, `float32()` |
| `create_token` | `float32(3,)` |
| `after_all` | `float32(3,)` |
| `cond` | `float32(3,)` |
| `scan` | `float32(3,)` |
| `while` | `int32()` |
| `custom_linear_solve` | `float32(3,)` |
| `gather` | `float32(3,)`, `int32(2, 1)` |
| `scatter` | `float32(3,)`, `int32(2, 1)`, `float32(2,)` |
| `scatter-add` | `float32(3,)`, `int32(2, 1)`, `float32(2,)` |
| `scatter-sub` | `float32(3,)`, `int32(2, 1)`, `float32(2,)` |
| `scatter-mul` | `float32(3,)`, `int32(2, 1)`, `float32(2,)` |
| `scatter-min` | `float32(3,)`, `int32(2, 1)`, `float32(2,)` |
| `scatter-max` | `float32(3,)`, `int32(2, 1)`, `float32(2,)` |
| `reduce_window_sum` | `float32(3,)` |
| `reduce_window_max` | `float32(3,)` |
| `reduce_window_min` | `float32(3,)` |
| `reduce_window` | `float32(3,)` |
| `reduce` | `float32(3,)` |
| `cholesky` | `float32(2, 2)` |
| `eig` | `float32(2, 2)` |
| `eigh` | `float32(2, 2)` |
| `hessenberg` | `float32(2, 2)` |
| `lu` | `float32(2, 2)` |
| `qr` | `float32(2, 2)` |
| `schur` | `float32(2, 2)` |
| `svd` | `float32(2, 2)` |
| `triangular_solve` | `float32(2, 2)`, `float32(2, 2)` |
| `tridiagonal` | `float32(2, 2)` |
| `tridiagonal_solve` | `float32(3,)`, `float32(3,)`, `float32(3,)`, `float32(3, 1)` |
| `lu_pivots_to_permutation` | `int32(3,)` |
| `cholesky_update` | `float32(2, 2)`, `float32(2,)` |
| `householder_product` | `float32(2, 2)`, `float32(2,)` |
| `ormqr` | `float32(2, 2)`, `float32(2,)`, `float32(2, 2)` |
| `symmetric_product` | `float32(2, 2)`, `float32(2, 2)` |
| `eq_to` | `float32(3,)`, `float32(3,)` |
| `le_to` | `float32(3,)`, `float32(3,)` |
| `lt_to` | `float32(3,)`, `float32(3,)` |
| `device_put` | `float32(3,)` |
| `sharding_constraint` | `float32(3,)` |
| `select_and_scatter_add` | `float32(2,)`, `float32(3,)` |
| `select_and_gather_add` | `float32(3,)`, `float32(3,)` |
| `select_and_scatter` | `float32(3,)`, `float32(2,)` |
| `psum` | `float32(4, 2)` |
| `pmin` | `float32(4, 2)` |
| `pmax` | `float32(4, 2)` |
| `axis_index` | `float32(4, 2)` |
| `all_gather` | `float32(4, 2)` |
| `all_to_all` | `float32(4, 2)` |
| `ppermute` | `float32(4, 2)` |
| `ragged_all_to_all` | `float32(4, 2)` |
| `pbroadcast` | `float32(4, 2)` |
| `reduce_scatter` | `float32(4, 2)` |
| `empty` | 无输入 |
| `composite` | `float32(3,)` |
| `composite_attributes` | `float32(3,)` |
| `platform_index` | `float32(3,)` |
| `dce_sink` | `float32(3,)` |
| `dce_sink_mlir` | `float32(3,)` |
| `psum_invariant` | `float32(4, 2)` |
| `scaled_dot` | `float32(2, 2)`, `float32(2, 2)` |
| `ragged_dot_general` | `float32(4, 3)`, `float32(2, 3, 2)`, `int32(2,)` |

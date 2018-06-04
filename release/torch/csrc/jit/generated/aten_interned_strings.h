#pragma once

// generated from tools/autograd/templates/aten_interned_strings.h

// ATen symbols correspond exactly to operators defined in ATen.  Every
// symbol here corresponds exactly to an ATen operation which is defined
// in Declarations.yaml; attributes are in one-to-one correspondence with
// their ATen name.

#define FORALL_ATEN_BASE_SYMBOLS(_) \
_(aten, RoiPooling2d_backward) \
_(aten, RoiPooling2d_forward) \
_(aten, __and__) \
_(aten, __and___out) \
_(aten, __iand__) \
_(aten, __ilshift__) \
_(aten, __ior__) \
_(aten, __irshift__) \
_(aten, __ixor__) \
_(aten, __lshift__) \
_(aten, __lshift___out) \
_(aten, __or__) \
_(aten, __or___out) \
_(aten, __rshift__) \
_(aten, __rshift___out) \
_(aten, __xor__) \
_(aten, __xor___out) \
_(aten, _abs) \
_(aten, _abs_out) \
_(aten, _acos) \
_(aten, _acos_out) \
_(aten, _addmv) \
_(aten, _addmv_out) \
_(aten, _addr) \
_(aten, _addr_out) \
_(aten, _arange) \
_(aten, _arange_out) \
_(aten, _argmax) \
_(aten, _argmin) \
_(aten, _asin) \
_(aten, _asin_out) \
_(aten, _atan) \
_(aten, _atan_out) \
_(aten, _cast_Half) \
_(aten, _cast_double) \
_(aten, _cast_float) \
_(aten, _cast_int) \
_(aten, _cast_int16_t) \
_(aten, _cast_int64_t) \
_(aten, _cast_int8_t) \
_(aten, _cast_uint8_t) \
_(aten, _cat) \
_(aten, _cat_out) \
_(aten, _ceil) \
_(aten, _ceil_out) \
_(aten, _convolution) \
_(aten, _convolution_double_backward) \
_(aten, _convolution_nogroup) \
_(aten, _copy_ignoring_overlaps) \
_(aten, _cos) \
_(aten, _cos_out) \
_(aten, _cosh) \
_(aten, _cosh_out) \
_(aten, _cudnn_init_dropout_state) \
_(aten, _cudnn_rnn) \
_(aten, _cudnn_rnn_backward) \
_(aten, _cudnn_rnn_flatten_weight) \
_(aten, _cumprod) \
_(aten, _cumprod_out) \
_(aten, _cumsum) \
_(aten, _cumsum_out) \
_(aten, _dimI) \
_(aten, _dimV) \
_(aten, _dirichlet_grad) \
_(aten, _dirichlet_grad_out) \
_(aten, _dot) \
_(aten, _erf) \
_(aten, _erf_out) \
_(aten, _exp) \
_(aten, _exp_out) \
_(aten, _expm1) \
_(aten, _expm1_out) \
_(aten, _fft_with_size) \
_(aten, _fill) \
_(aten, _floor) \
_(aten, _floor_out) \
_(aten, _ger) \
_(aten, _ger_out) \
_(aten, _gesv_helper) \
_(aten, _gesv_single) \
_(aten, _gesv_single_out) \
_(aten, _indexCopy) \
_(aten, _indices) \
_(aten, _linspace) \
_(aten, _linspace_out) \
_(aten, _log) \
_(aten, _log10) \
_(aten, _log10_out) \
_(aten, _log1p) \
_(aten, _log1p_out) \
_(aten, _log2) \
_(aten, _log2_out) \
_(aten, _log_out) \
_(aten, _logspace) \
_(aten, _logspace_out) \
_(aten, _mm) \
_(aten, _mm_out) \
_(aten, _mv) \
_(aten, _mv_out) \
_(aten, _nnz) \
_(aten, _prod) \
_(aten, _prod_out) \
_(aten, _prodall) \
_(aten, _range) \
_(aten, _range_out) \
_(aten, _round) \
_(aten, _round_out) \
_(aten, _rsqrt) \
_(aten, _rsqrt_out) \
_(aten, _s_where) \
_(aten, _sigmoid) \
_(aten, _sigmoid_backward) \
_(aten, _sigmoid_backward_out) \
_(aten, _sigmoid_forward) \
_(aten, _sigmoid_forward_out) \
_(aten, _sigmoid_out) \
_(aten, _sin) \
_(aten, _sin_out) \
_(aten, _sinh) \
_(aten, _sinh_out) \
_(aten, _sparse_coo_tensor_unsafe) \
_(aten, _sparse_mask) \
_(aten, _sqrt) \
_(aten, _sqrt_out) \
_(aten, _standard_gamma) \
_(aten, _standard_gamma_grad) \
_(aten, _sum) \
_(aten, _sum_cuda_out) \
_(aten, _sum_out) \
_(aten, _sumall) \
_(aten, _tan) \
_(aten, _tan_out) \
_(aten, _tanh) \
_(aten, _tanh_backward) \
_(aten, _tanh_backward_out) \
_(aten, _tanh_forward) \
_(aten, _tanh_forward_out) \
_(aten, _tanh_out) \
_(aten, _th_bernoulli) \
_(aten, _th_bernoulli_out) \
_(aten, _th_prod) \
_(aten, _th_prod_out) \
_(aten, _th_sum) \
_(aten, _th_sum_out) \
_(aten, _th_tanh) \
_(aten, _th_tanh_out) \
_(aten, _trilinear) \
_(aten, _trunc) \
_(aten, _trunc_out) \
_(aten, _unique) \
_(aten, _unsafe_view) \
_(aten, _values) \
_(aten, abs) \
_(aten, abs_out) \
_(aten, acos) \
_(aten, acos_out) \
_(aten, adaptive_avg_pool1d) \
_(aten, adaptive_avg_pool2d) \
_(aten, adaptive_avg_pool2d_backward) \
_(aten, adaptive_avg_pool2d_backward_out) \
_(aten, adaptive_avg_pool2d_forward) \
_(aten, adaptive_avg_pool2d_forward_out) \
_(aten, adaptive_avg_pool2d_out) \
_(aten, adaptive_avg_pool3d) \
_(aten, adaptive_avg_pool3d_backward) \
_(aten, adaptive_avg_pool3d_backward_out) \
_(aten, adaptive_avg_pool3d_forward) \
_(aten, adaptive_avg_pool3d_forward_out) \
_(aten, adaptive_avg_pool3d_out) \
_(aten, adaptive_max_pool1d) \
_(aten, adaptive_max_pool2d) \
_(aten, adaptive_max_pool2d_backward) \
_(aten, adaptive_max_pool2d_backward_out) \
_(aten, adaptive_max_pool2d_forward) \
_(aten, adaptive_max_pool2d_forward_out) \
_(aten, adaptive_max_pool2d_out) \
_(aten, adaptive_max_pool3d) \
_(aten, adaptive_max_pool3d_backward) \
_(aten, adaptive_max_pool3d_backward_out) \
_(aten, adaptive_max_pool3d_forward) \
_(aten, adaptive_max_pool3d_forward_out) \
_(aten, adaptive_max_pool3d_out) \
_(aten, add) \
_(aten, add_out) \
_(aten, addbmm) \
_(aten, addbmm_out) \
_(aten, addcdiv) \
_(aten, addcdiv_out) \
_(aten, addcmul) \
_(aten, addcmul_out) \
_(aten, addmm) \
_(aten, addmm_out) \
_(aten, addmv) \
_(aten, addmv_out) \
_(aten, addr) \
_(aten, addr_out) \
_(aten, alias) \
_(aten, all) \
_(aten, all_out) \
_(aten, allclose) \
_(aten, any) \
_(aten, any_out) \
_(aten, arange) \
_(aten, arange_out) \
_(aten, argmax) \
_(aten, argmin) \
_(aten, as_strided) \
_(aten, as_strided_out) \
_(aten, asin) \
_(aten, asin_out) \
_(aten, atan) \
_(aten, atan2) \
_(aten, atan2_out) \
_(aten, atan_out) \
_(aten, avg_pool2d) \
_(aten, avg_pool2d_backward) \
_(aten, avg_pool2d_backward_out) \
_(aten, avg_pool2d_forward) \
_(aten, avg_pool2d_forward_out) \
_(aten, avg_pool2d_out) \
_(aten, avg_pool3d) \
_(aten, avg_pool3d_backward) \
_(aten, avg_pool3d_backward_out) \
_(aten, avg_pool3d_forward) \
_(aten, avg_pool3d_forward_out) \
_(aten, avg_pool3d_out) \
_(aten, baddbmm) \
_(aten, baddbmm_out) \
_(aten, batch_norm) \
_(aten, bernoulli) \
_(aten, bilinear) \
_(aten, binary_cross_entropy) \
_(aten, binary_cross_entropy_backward) \
_(aten, binary_cross_entropy_backward_out) \
_(aten, binary_cross_entropy_forward) \
_(aten, binary_cross_entropy_forward_out) \
_(aten, binary_cross_entropy_out) \
_(aten, bmm) \
_(aten, bmm_out) \
_(aten, btrifact) \
_(aten, btrifact_out) \
_(aten, btrifact_with_info) \
_(aten, btrifact_with_info_out) \
_(aten, btrisolve) \
_(aten, btrisolve_out) \
_(aten, cat) \
_(aten, cat_out) \
_(aten, cauchy) \
_(aten, ceil) \
_(aten, ceil_out) \
_(aten, chunk) \
_(aten, clamp) \
_(aten, clamp_max) \
_(aten, clamp_max_out) \
_(aten, clamp_min) \
_(aten, clamp_min_out) \
_(aten, clamp_out) \
_(aten, clone) \
_(aten, coalesce) \
_(aten, contiguous) \
_(aten, conv1d) \
_(aten, conv2d) \
_(aten, conv3d) \
_(aten, conv_tbc) \
_(aten, conv_tbc_backward) \
_(aten, conv_transpose1d) \
_(aten, conv_transpose2d) \
_(aten, conv_transpose3d) \
_(aten, convolution) \
_(aten, cos) \
_(aten, cos_out) \
_(aten, cosh) \
_(aten, cosh_out) \
_(aten, cosine_embedding_loss) \
_(aten, cross) \
_(aten, cross_out) \
_(aten, cudnn_affine_grid_generator) \
_(aten, cudnn_affine_grid_generator_backward) \
_(aten, cudnn_batch_norm) \
_(aten, cudnn_batch_norm_backward) \
_(aten, cudnn_convolution) \
_(aten, cudnn_convolution_backward) \
_(aten, cudnn_convolution_backward_bias) \
_(aten, cudnn_convolution_backward_input) \
_(aten, cudnn_convolution_backward_weight) \
_(aten, cudnn_convolution_transpose) \
_(aten, cudnn_convolution_transpose_backward) \
_(aten, cudnn_convolution_transpose_backward_bias) \
_(aten, cudnn_convolution_transpose_backward_input) \
_(aten, cudnn_convolution_transpose_backward_weight) \
_(aten, cudnn_grid_sampler) \
_(aten, cudnn_grid_sampler_backward) \
_(aten, cudnn_is_acceptable) \
_(aten, cumprod) \
_(aten, cumprod_out) \
_(aten, cumsum) \
_(aten, cumsum_out) \
_(aten, data_ptr) \
_(aten, det) \
_(aten, diag) \
_(aten, diag_out) \
_(aten, diagflat) \
_(aten, diagonal) \
_(aten, digamma) \
_(aten, digamma_out) \
_(aten, dim) \
_(aten, dist) \
_(aten, div) \
_(aten, div_out) \
_(aten, dot) \
_(aten, dot_out) \
_(aten, eig) \
_(aten, eig_out) \
_(aten, einsum) \
_(aten, elu) \
_(aten, elu_backward) \
_(aten, elu_backward_out) \
_(aten, elu_forward) \
_(aten, elu_forward_out) \
_(aten, elu_out) \
_(aten, embedding) \
_(aten, embedding_backward) \
_(aten, embedding_bag) \
_(aten, embedding_bag_backward) \
_(aten, embedding_bag_dense_backward) \
_(aten, embedding_bag_sparse_backward) \
_(aten, embedding_dense_backward) \
_(aten, embedding_renorm) \
_(aten, embedding_sparse_backward) \
_(aten, empty) \
_(aten, empty_like) \
_(aten, empty_out) \
_(aten, eq) \
_(aten, eq_out) \
_(aten, equal) \
_(aten, erf) \
_(aten, erf_out) \
_(aten, erfinv) \
_(aten, erfinv_out) \
_(aten, exp) \
_(aten, exp_out) \
_(aten, expand) \
_(aten, expand_as) \
_(aten, expm1) \
_(aten, expm1_out) \
_(aten, exponential) \
_(aten, eye) \
_(aten, eye_out) \
_(aten, fft) \
_(aten, fill) \
_(aten, floor) \
_(aten, floor_out) \
_(aten, fmod) \
_(aten, fmod_out) \
_(aten, frac) \
_(aten, frac_out) \
_(aten, fractional_max_pool2d) \
_(aten, fractional_max_pool2d_backward) \
_(aten, fractional_max_pool2d_backward_out) \
_(aten, fractional_max_pool2d_forward) \
_(aten, fractional_max_pool2d_forward_out) \
_(aten, fractional_max_pool2d_out) \
_(aten, full) \
_(aten, full_like) \
_(aten, full_out) \
_(aten, gather) \
_(aten, gather_out) \
_(aten, ge) \
_(aten, ge_out) \
_(aten, gels) \
_(aten, gels_out) \
_(aten, geometric) \
_(aten, geqrf) \
_(aten, geqrf_out) \
_(aten, ger) \
_(aten, ger_out) \
_(aten, gesv) \
_(aten, gesv_out) \
_(aten, get_device) \
_(aten, glu) \
_(aten, glu_backward) \
_(aten, glu_backward_out) \
_(aten, glu_forward) \
_(aten, glu_forward_out) \
_(aten, glu_out) \
_(aten, group_norm) \
_(aten, gt) \
_(aten, gt_out) \
_(aten, hardshrink) \
_(aten, hardshrink_backward) \
_(aten, hardshrink_backward_out) \
_(aten, hardshrink_forward) \
_(aten, hardshrink_forward_out) \
_(aten, hardshrink_out) \
_(aten, hardtanh) \
_(aten, hardtanh_backward) \
_(aten, hardtanh_backward_out) \
_(aten, hardtanh_forward) \
_(aten, hardtanh_forward_out) \
_(aten, hardtanh_out) \
_(aten, hinge_embedding_loss) \
_(aten, histc) \
_(aten, histc_out) \
_(aten, hspmm) \
_(aten, hspmm_out) \
_(aten, ifft) \
_(aten, index) \
_(aten, index_add) \
_(aten, index_copy) \
_(aten, index_fill) \
_(aten, index_put) \
_(aten, index_select) \
_(aten, index_select_out) \
_(aten, inverse) \
_(aten, inverse_out) \
_(aten, irfft) \
_(aten, is_coalesced) \
_(aten, is_contiguous) \
_(aten, is_cuda) \
_(aten, is_distributed) \
_(aten, is_floating_point) \
_(aten, is_nonzero) \
_(aten, is_same_size) \
_(aten, is_set_to) \
_(aten, is_signed) \
_(aten, is_sparse) \
_(aten, isclose) \
_(aten, kl_div) \
_(aten, kl_div_backward) \
_(aten, kl_div_backward_out) \
_(aten, kl_div_forward) \
_(aten, kl_div_forward_out) \
_(aten, kl_div_out) \
_(aten, kthvalue) \
_(aten, kthvalue_out) \
_(aten, l1_loss) \
_(aten, l1_loss_backward) \
_(aten, l1_loss_backward_out) \
_(aten, l1_loss_forward) \
_(aten, l1_loss_forward_out) \
_(aten, l1_loss_out) \
_(aten, layer_norm) \
_(aten, le) \
_(aten, le_out) \
_(aten, leaky_relu) \
_(aten, leaky_relu_backward) \
_(aten, leaky_relu_backward_out) \
_(aten, leaky_relu_forward) \
_(aten, leaky_relu_forward_out) \
_(aten, leaky_relu_out) \
_(aten, lerp) \
_(aten, lerp_out) \
_(aten, lgamma) \
_(aten, lgamma_out) \
_(aten, linspace) \
_(aten, linspace_out) \
_(aten, log) \
_(aten, log10) \
_(aten, log10_out) \
_(aten, log1p) \
_(aten, log1p_out) \
_(aten, log2) \
_(aten, log2_out) \
_(aten, log_normal) \
_(aten, log_out) \
_(aten, log_sigmoid) \
_(aten, log_sigmoid_backward) \
_(aten, log_sigmoid_backward_out) \
_(aten, log_sigmoid_forward) \
_(aten, log_sigmoid_forward_out) \
_(aten, log_sigmoid_out) \
_(aten, log_softmax) \
_(aten, log_softmax_backward_data) \
_(aten, logdet) \
_(aten, logspace) \
_(aten, logspace_out) \
_(aten, logsumexp) \
_(aten, logsumexp_out) \
_(aten, lt) \
_(aten, lt_out) \
_(aten, margin_ranking_loss) \
_(aten, masked_fill) \
_(aten, masked_scatter) \
_(aten, masked_select) \
_(aten, masked_select_out) \
_(aten, matmul) \
_(aten, matmul_out) \
_(aten, max) \
_(aten, max_out) \
_(aten, max_pool1d) \
_(aten, max_pool2d) \
_(aten, max_pool2d_backward) \
_(aten, max_pool2d_backward_out) \
_(aten, max_pool2d_forward) \
_(aten, max_pool2d_forward_out) \
_(aten, max_pool2d_out) \
_(aten, max_pool3d) \
_(aten, max_pool3d_backward) \
_(aten, max_pool3d_backward_out) \
_(aten, max_pool3d_forward) \
_(aten, max_pool3d_forward_out) \
_(aten, max_pool3d_out) \
_(aten, max_unpool2d) \
_(aten, max_unpool2d_backward) \
_(aten, max_unpool2d_backward_out) \
_(aten, max_unpool2d_forward) \
_(aten, max_unpool2d_forward_out) \
_(aten, max_unpool2d_out) \
_(aten, max_unpool3d) \
_(aten, max_unpool3d_backward) \
_(aten, max_unpool3d_backward_out) \
_(aten, max_unpool3d_forward) \
_(aten, max_unpool3d_forward_out) \
_(aten, max_unpool3d_out) \
_(aten, max_values) \
_(aten, mean) \
_(aten, mean_out) \
_(aten, median) \
_(aten, median_out) \
_(aten, min) \
_(aten, min_out) \
_(aten, min_values) \
_(aten, mkldnn_convolution) \
_(aten, mkldnn_convolution_backward) \
_(aten, mkldnn_convolution_backward_input) \
_(aten, mkldnn_convolution_backward_weights) \
_(aten, mm) \
_(aten, mm_out) \
_(aten, mode) \
_(aten, mode_out) \
_(aten, mse_loss) \
_(aten, mse_loss_backward) \
_(aten, mse_loss_backward_out) \
_(aten, mse_loss_forward) \
_(aten, mse_loss_forward_out) \
_(aten, mse_loss_out) \
_(aten, mul) \
_(aten, mul_out) \
_(aten, multi_margin_loss) \
_(aten, multi_margin_loss_backward) \
_(aten, multi_margin_loss_backward_out) \
_(aten, multi_margin_loss_forward) \
_(aten, multi_margin_loss_forward_out) \
_(aten, multi_margin_loss_out) \
_(aten, multilabel_margin_loss) \
_(aten, multilabel_margin_loss_backward) \
_(aten, multilabel_margin_loss_backward_out) \
_(aten, multilabel_margin_loss_forward) \
_(aten, multilabel_margin_loss_forward_out) \
_(aten, multilabel_margin_loss_out) \
_(aten, multinomial) \
_(aten, multinomial_out) \
_(aten, mv) \
_(aten, mv_out) \
_(aten, narrow) \
_(aten, ne) \
_(aten, ne_out) \
_(aten, neg) \
_(aten, neg_out) \
_(aten, nll_loss) \
_(aten, nll_loss2d) \
_(aten, nll_loss2d_backward) \
_(aten, nll_loss2d_backward_out) \
_(aten, nll_loss2d_forward) \
_(aten, nll_loss2d_forward_out) \
_(aten, nll_loss2d_out) \
_(aten, nll_loss_backward) \
_(aten, nll_loss_backward_out) \
_(aten, nll_loss_forward) \
_(aten, nll_loss_forward_out) \
_(aten, nll_loss_out) \
_(aten, nonzero) \
_(aten, nonzero_out) \
_(aten, norm) \
_(aten, norm_out) \
_(aten, normal) \
_(aten, normal_out) \
_(aten, numel) \
_(aten, ones) \
_(aten, ones_like) \
_(aten, ones_out) \
_(aten, orgqr) \
_(aten, orgqr_out) \
_(aten, ormqr) \
_(aten, ormqr_out) \
_(aten, pairwise_distance) \
_(aten, permute) \
_(aten, pin_memory) \
_(aten, poisson) \
_(aten, polygamma) \
_(aten, polygamma_out) \
_(aten, potrf) \
_(aten, potrf_out) \
_(aten, potri) \
_(aten, potri_out) \
_(aten, potrs) \
_(aten, potrs_out) \
_(aten, pow) \
_(aten, pow_out) \
_(aten, prelu) \
_(aten, prelu_backward) \
_(aten, prelu_backward_out) \
_(aten, prelu_forward) \
_(aten, prelu_forward_out) \
_(aten, prelu_out) \
_(aten, prod) \
_(aten, prod_out) \
_(aten, pstrf) \
_(aten, pstrf_out) \
_(aten, put) \
_(aten, qr) \
_(aten, qr_out) \
_(aten, rand) \
_(aten, rand_like) \
_(aten, rand_out) \
_(aten, randint) \
_(aten, randint_like) \
_(aten, randint_out) \
_(aten, randn) \
_(aten, randn_like) \
_(aten, randn_out) \
_(aten, random) \
_(aten, randperm) \
_(aten, randperm_out) \
_(aten, range) \
_(aten, range_out) \
_(aten, reciprocal) \
_(aten, reciprocal_out) \
_(aten, reflection_pad1d) \
_(aten, reflection_pad1d_backward) \
_(aten, reflection_pad1d_backward_out) \
_(aten, reflection_pad1d_forward) \
_(aten, reflection_pad1d_forward_out) \
_(aten, reflection_pad1d_out) \
_(aten, reflection_pad2d) \
_(aten, reflection_pad2d_backward) \
_(aten, reflection_pad2d_backward_out) \
_(aten, reflection_pad2d_forward) \
_(aten, reflection_pad2d_forward_out) \
_(aten, reflection_pad2d_out) \
_(aten, relu) \
_(aten, remainder) \
_(aten, remainder_out) \
_(aten, renorm) \
_(aten, renorm_out) \
_(aten, repeat) \
_(aten, replication_pad1d) \
_(aten, replication_pad1d_backward) \
_(aten, replication_pad1d_backward_out) \
_(aten, replication_pad1d_forward) \
_(aten, replication_pad1d_forward_out) \
_(aten, replication_pad1d_out) \
_(aten, replication_pad2d) \
_(aten, replication_pad2d_backward) \
_(aten, replication_pad2d_backward_out) \
_(aten, replication_pad2d_forward) \
_(aten, replication_pad2d_forward_out) \
_(aten, replication_pad2d_out) \
_(aten, replication_pad3d) \
_(aten, replication_pad3d_backward) \
_(aten, replication_pad3d_backward_out) \
_(aten, replication_pad3d_forward) \
_(aten, replication_pad3d_forward_out) \
_(aten, replication_pad3d_out) \
_(aten, reshape) \
_(aten, resize) \
_(aten, resize_as) \
_(aten, rfft) \
_(aten, round) \
_(aten, round_out) \
_(aten, rrelu) \
_(aten, rrelu_with_noise) \
_(aten, rrelu_with_noise_backward) \
_(aten, rrelu_with_noise_backward_out) \
_(aten, rrelu_with_noise_forward) \
_(aten, rrelu_with_noise_forward_out) \
_(aten, rrelu_with_noise_out) \
_(aten, rsqrt) \
_(aten, rsqrt_out) \
_(aten, scatter) \
_(aten, scatter_add) \
_(aten, select) \
_(aten, selu) \
_(aten, set) \
_(aten, sigmoid) \
_(aten, sigmoid_out) \
_(aten, sign) \
_(aten, sign_out) \
_(aten, sin) \
_(aten, sin_out) \
_(aten, sinh) \
_(aten, sinh_out) \
_(aten, size) \
_(aten, sizes) \
_(aten, slice) \
_(aten, slogdet) \
_(aten, smm) \
_(aten, smooth_l1_loss) \
_(aten, smooth_l1_loss_backward) \
_(aten, smooth_l1_loss_backward_out) \
_(aten, smooth_l1_loss_forward) \
_(aten, smooth_l1_loss_forward_out) \
_(aten, smooth_l1_loss_out) \
_(aten, soft_margin_loss) \
_(aten, soft_margin_loss_backward) \
_(aten, soft_margin_loss_backward_out) \
_(aten, soft_margin_loss_forward) \
_(aten, soft_margin_loss_forward_out) \
_(aten, soft_margin_loss_out) \
_(aten, softmax) \
_(aten, softmax_backward_data) \
_(aten, softplus) \
_(aten, softplus_backward) \
_(aten, softplus_backward_out) \
_(aten, softplus_forward) \
_(aten, softplus_forward_out) \
_(aten, softplus_out) \
_(aten, softshrink) \
_(aten, softshrink_backward) \
_(aten, softshrink_backward_out) \
_(aten, softshrink_forward) \
_(aten, softshrink_forward_out) \
_(aten, softshrink_out) \
_(aten, sort) \
_(aten, sort_out) \
_(aten, sparse_coo_tensor) \
_(aten, sparse_raw_resize) \
_(aten, split) \
_(aten, split_with_sizes) \
_(aten, sqrt) \
_(aten, sqrt_out) \
_(aten, squeeze) \
_(aten, sspaddmm) \
_(aten, sspaddmm_out) \
_(aten, stack) \
_(aten, stack_out) \
_(aten, std) \
_(aten, std_out) \
_(aten, stft) \
_(aten, storage_offset) \
_(aten, stride) \
_(aten, strides) \
_(aten, sub) \
_(aten, sub_out) \
_(aten, sum) \
_(aten, sum_out) \
_(aten, svd) \
_(aten, svd_out) \
_(aten, symeig) \
_(aten, symeig_out) \
_(aten, t) \
_(aten, take) \
_(aten, take_out) \
_(aten, tan) \
_(aten, tan_out) \
_(aten, tanh) \
_(aten, tanh_out) \
_(aten, tensor) \
_(aten, thnn_batch_norm) \
_(aten, thnn_batch_norm_backward) \
_(aten, thnn_batch_norm_backward_out) \
_(aten, thnn_batch_norm_forward) \
_(aten, thnn_batch_norm_forward_out) \
_(aten, thnn_batch_norm_out) \
_(aten, thnn_conv2d) \
_(aten, thnn_conv2d_backward) \
_(aten, thnn_conv2d_backward_out) \
_(aten, thnn_conv2d_forward) \
_(aten, thnn_conv2d_forward_out) \
_(aten, thnn_conv2d_out) \
_(aten, thnn_conv3d) \
_(aten, thnn_conv3d_backward) \
_(aten, thnn_conv3d_backward_out) \
_(aten, thnn_conv3d_forward) \
_(aten, thnn_conv3d_forward_out) \
_(aten, thnn_conv3d_out) \
_(aten, thnn_conv_depthwise2d) \
_(aten, thnn_conv_depthwise2d_backward) \
_(aten, thnn_conv_depthwise2d_backward_out) \
_(aten, thnn_conv_depthwise2d_forward) \
_(aten, thnn_conv_depthwise2d_forward_out) \
_(aten, thnn_conv_depthwise2d_out) \
_(aten, thnn_conv_dilated2d) \
_(aten, thnn_conv_dilated2d_backward) \
_(aten, thnn_conv_dilated2d_backward_out) \
_(aten, thnn_conv_dilated2d_forward) \
_(aten, thnn_conv_dilated2d_forward_out) \
_(aten, thnn_conv_dilated2d_out) \
_(aten, thnn_conv_dilated3d) \
_(aten, thnn_conv_dilated3d_backward) \
_(aten, thnn_conv_dilated3d_backward_out) \
_(aten, thnn_conv_dilated3d_forward) \
_(aten, thnn_conv_dilated3d_forward_out) \
_(aten, thnn_conv_dilated3d_out) \
_(aten, thnn_conv_transpose2d) \
_(aten, thnn_conv_transpose2d_backward) \
_(aten, thnn_conv_transpose2d_backward_out) \
_(aten, thnn_conv_transpose2d_forward) \
_(aten, thnn_conv_transpose2d_forward_out) \
_(aten, thnn_conv_transpose2d_out) \
_(aten, thnn_conv_transpose3d) \
_(aten, thnn_conv_transpose3d_backward) \
_(aten, thnn_conv_transpose3d_backward_out) \
_(aten, thnn_conv_transpose3d_forward) \
_(aten, thnn_conv_transpose3d_forward_out) \
_(aten, thnn_conv_transpose3d_out) \
_(aten, threshold) \
_(aten, threshold_backward) \
_(aten, threshold_backward_out) \
_(aten, threshold_forward) \
_(aten, threshold_forward_out) \
_(aten, threshold_out) \
_(aten, to_dense) \
_(aten, topk) \
_(aten, topk_out) \
_(aten, trace) \
_(aten, transpose) \
_(aten, tril) \
_(aten, tril_out) \
_(aten, triplet_margin_loss) \
_(aten, triu) \
_(aten, triu_out) \
_(aten, trtrs) \
_(aten, trtrs_out) \
_(aten, trunc) \
_(aten, trunc_out) \
_(aten, type_as) \
_(aten, unfold) \
_(aten, uniform) \
_(aten, unsqueeze) \
_(aten, upsample_bilinear2d) \
_(aten, upsample_bilinear2d_backward) \
_(aten, upsample_bilinear2d_backward_out) \
_(aten, upsample_bilinear2d_forward) \
_(aten, upsample_bilinear2d_forward_out) \
_(aten, upsample_bilinear2d_out) \
_(aten, upsample_linear1d) \
_(aten, upsample_linear1d_backward) \
_(aten, upsample_linear1d_backward_out) \
_(aten, upsample_linear1d_forward) \
_(aten, upsample_linear1d_forward_out) \
_(aten, upsample_linear1d_out) \
_(aten, upsample_nearest1d) \
_(aten, upsample_nearest1d_backward) \
_(aten, upsample_nearest1d_backward_out) \
_(aten, upsample_nearest1d_forward) \
_(aten, upsample_nearest1d_forward_out) \
_(aten, upsample_nearest1d_out) \
_(aten, upsample_nearest2d) \
_(aten, upsample_nearest2d_backward) \
_(aten, upsample_nearest2d_backward_out) \
_(aten, upsample_nearest2d_forward) \
_(aten, upsample_nearest2d_forward_out) \
_(aten, upsample_nearest2d_out) \
_(aten, upsample_nearest3d) \
_(aten, upsample_nearest3d_backward) \
_(aten, upsample_nearest3d_backward_out) \
_(aten, upsample_nearest3d_forward) \
_(aten, upsample_nearest3d_forward_out) \
_(aten, upsample_nearest3d_out) \
_(aten, upsample_trilinear3d) \
_(aten, upsample_trilinear3d_backward) \
_(aten, upsample_trilinear3d_backward_out) \
_(aten, upsample_trilinear3d_forward) \
_(aten, upsample_trilinear3d_forward_out) \
_(aten, upsample_trilinear3d_out) \
_(aten, var) \
_(aten, var_out) \
_(aten, view) \
_(aten, view_as) \
_(aten, where) \
_(aten, zero) \
_(aten, zeros) \
_(aten, zeros_like) \
_(aten, zeros_out) \
/* nothing */

#define FORALL_ATTR_BASE_SYMBOLS(_) \
_(attr, A) \
_(attr, C) \
_(attr, H) \
_(attr, LU_data) \
_(attr, LU_pivots) \
_(attr, N) \
_(attr, W) \
_(attr, accumulate) \
_(attr, align_corners) \
_(attr, alpha) \
_(attr, anchor) \
_(attr, argmaxes) \
_(attr, atol) \
_(attr, bag_size) \
_(attr, base) \
_(attr, batch1) \
_(attr, batch2) \
_(attr, batch_first) \
_(attr, batch_sizes) \
_(attr, benchmark) \
_(attr, beta) \
_(attr, bias) \
_(attr, bias_defined) \
_(attr, bidirectional) \
_(attr, bins) \
_(attr, buffer) \
_(attr, ceil_mode) \
_(attr, checked_signal_sizes) \
_(attr, chunks) \
_(attr, columns) \
_(attr, complex_input) \
_(attr, complex_output) \
_(attr, condition) \
_(attr, count_include_pad) \
_(attr, cudnn_enable) \
_(attr, cudnn_enabled) \
_(attr, cx) \
_(attr, descending) \
_(attr, deterministic) \
_(attr, diagonal) \
_(attr, dilation) \
_(attr, dim) \
_(attr, dim0) \
_(attr, dim1) \
_(attr, dim2) \
_(attr, dimension) \
_(attr, dims) \
_(attr, dropout) \
_(attr, dropout_seed) \
_(attr, dropout_state) \
_(attr, dtype) \
_(attr, eigenvectors) \
_(attr, end) \
_(attr, eps) \
_(attr, epsilon) \
_(attr, equal_nan) \
_(attr, equation) \
_(attr, expand1) \
_(attr, expand2) \
_(attr, expand3) \
_(attr, exponent) \
_(attr, exponential_average_factor) \
_(attr, fft_size) \
_(attr, fgrad_input) \
_(attr, fill_value) \
_(attr, finput) \
_(attr, frame_length) \
_(attr, from) \
_(attr, gO) \
_(attr, generator) \
_(attr, ggI) \
_(attr, ggW) \
_(attr, ggb) \
_(attr, grad) \
_(attr, gradOutput) \
_(attr, grad_bias) \
_(attr, grad_cy) \
_(attr, grad_hy) \
_(attr, grad_input) \
_(attr, grad_output) \
_(attr, grad_weight) \
_(attr, grid) \
_(attr, groups) \
_(attr, hidden_size) \
_(attr, high) \
_(attr, hop) \
_(attr, hx) \
_(attr, i1) \
_(attr, i2) \
_(attr, i3) \
_(attr, ignore_index) \
_(attr, implicit) \
_(attr, index) \
_(attr, indices) \
_(attr, info) \
_(attr, input) \
_(attr, input1) \
_(attr, input2) \
_(attr, input3) \
_(attr, input_size) \
_(attr, inverse) \
_(attr, is_target) \
_(attr, k) \
_(attr, keepdim) \
_(attr, kernel_size) \
_(attr, lambd) \
_(attr, largest) \
_(attr, left) \
_(attr, length) \
_(attr, low) \
_(attr, lower) \
_(attr, lu) \
_(attr, m) \
_(attr, margin) \
_(attr, mask) \
_(attr, mat) \
_(attr, mat1) \
_(attr, mat2) \
_(attr, max) \
_(attr, max_indices) \
_(attr, max_norm) \
_(attr, max_val) \
_(attr, maximum_indices) \
_(attr, maxnorm) \
_(attr, mean) \
_(attr, median) \
_(attr, min) \
_(attr, min_indices) \
_(attr, min_val) \
_(attr, mode) \
_(attr, momentum) \
_(attr, n) \
_(attr, nDimI) \
_(attr, nDimV) \
_(attr, negative) \
_(attr, negative_slope) \
_(attr, noise) \
_(attr, non_blocking) \
_(attr, norm_type) \
_(attr, normalized) \
_(attr, normalized_shape) \
_(attr, num_groups) \
_(attr, num_layers) \
_(attr, num_samples) \
_(attr, num_weights) \
_(attr, offset) \
_(attr, offset2bag) \
_(attr, offsets) \
_(attr, ones) \
_(attr, onesided) \
_(attr, other) \
_(attr, output) \
_(attr, output_mask) \
_(attr, output_padding) \
_(attr, output_size) \
_(attr, output_sizes) \
_(attr, p) \
_(attr, pad) \
_(attr, pad_end) \
_(attr, padding) \
_(attr, padding_idx) \
_(attr, pivot) \
_(attr, pivots) \
_(attr, pooledHeight) \
_(attr, pooledWidth) \
_(attr, positive) \
_(attr, random_samples) \
_(attr, reduce) \
_(attr, repeats) \
_(attr, replacement) \
_(attr, res1) \
_(attr, res2) \
_(attr, res3) \
_(attr, reserve) \
_(attr, result) \
_(attr, return_inverse) \
_(attr, rois) \
_(attr, rtol) \
_(attr, running_mean) \
_(attr, running_var) \
_(attr, save_mean) \
_(attr, save_std) \
_(attr, save_var) \
_(attr, scale) \
_(attr, scale_factor) \
_(attr, scale_grad_by_freq) \
_(attr, self) \
_(attr, self_size) \
_(attr, self_ty) \
_(attr, shape) \
_(attr, sigma) \
_(attr, signal_ndim) \
_(attr, signal_sizes) \
_(attr, size) \
_(attr, size_average) \
_(attr, solution) \
_(attr, some) \
_(attr, sorted) \
_(attr, source) \
_(attr, sourceStorage) \
_(attr, sparse) \
_(attr, spatialScale) \
_(attr, split_size) \
_(attr, split_sizes) \
_(attr, src) \
_(attr, start) \
_(attr, std) \
_(attr, step) \
_(attr, steps) \
_(attr, storage) \
_(attr, storageOffset) \
_(attr, storage_offset) \
_(attr, stride) \
_(attr, sumdim) \
_(attr, swap) \
_(attr, target) \
_(attr, tensor) \
_(attr, tensor1) \
_(attr, tensor2) \
_(attr, tensors) \
_(attr, the_template) \
_(attr, theta) \
_(attr, threshold) \
_(attr, to) \
_(attr, tol) \
_(attr, total) \
_(attr, total_weight) \
_(attr, train) \
_(attr, training) \
_(attr, transpose) \
_(attr, transposed) \
_(attr, unbiased) \
_(attr, unitriangular) \
_(attr, unroll_dim) \
_(attr, upper) \
_(attr, value) \
_(attr, values) \
_(attr, vec) \
_(attr, vec1) \
_(attr, vec2) \
_(attr, weight) \
_(attr, weight_arr) \
_(attr, weight_buf) \
_(attr, weight_size) \
_(attr, weight_stride0) \
_(attr, window) \
_(attr, x) \
_(attr, x1) \
_(attr, x2) \
/* nothing */

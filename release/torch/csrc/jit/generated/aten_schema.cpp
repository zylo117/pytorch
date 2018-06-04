#include "torch/csrc/jit/aten_schema.h"
#include "torch/csrc/jit/tensor_conversions.h"

namespace torch { namespace jit {

using SchemaMap = std::unordered_map<std::string, std::vector<FunctionSchema>>;


std::vector<FunctionSchema> createOperatorSchemas() {
  using namespace at; // for tensor initialization
  std::vector<FunctionSchema> schemas;

  // [aten_schema encoding]
  // This format tries to minimize the actual amount of code produced here to keep
  // compile times low. A naive encoding of this data directly into constructor
  // literals took over 3 minutes in gcc, while this format takes only 10 seconds.

  // However, it is more complicated because of this issue and described below

  // literals are stored uniqued and interned in these arrays:

  // string literals
  const char* names[] = {
    "storage_offset", 
    "self", 
    "result", 
    "numel", 
    "is_contiguous", 
    "is_set_to", 
    "tensor", 
    "masked_select", 
    "mask", 
    "nonzero", 
    "contiguous", 
    "clone", 
    "view", 
    "size", 
    "index_select", 
    "dim", 
    "index", 
    "take", 
    "unfold", 
    "dimension", 
    "step", 
    "gather", 
    "data_ptr", 
    "equal", 
    "other", 
    "__and__", 
    "__iand__", 
    "__or__", 
    "__ior__", 
    "__xor__", 
    "__ixor__", 
    "__lshift__", 
    "__ilshift__", 
    "__rshift__", 
    "__irshift__", 
    "lt", 
    "gt", 
    "le", 
    "ge", 
    "eq", 
    "ne", 
    "min", 
    "keepdim", 
    "min_indices", 
    "max", 
    "max_indices", 
    "kthvalue", 
    "k", 
    "values", 
    "indices", 
    "mode", 
    "median", 
    "sort", 
    "descending", 
    "topk", 
    "largest", 
    "sorted", 
    "all", 
    "any", 
    "get_device", 
    "_abs", 
    "sigmoid", 
    "_log", 
    "_log10", 
    "_log1p", 
    "_log2", 
    "lgamma", 
    "digamma", 
    "polygamma", 
    "n", 
    "_exp", 
    "_expm1", 
    "_cos", 
    "_acos", 
    "_cosh", 
    "_sin", 
    "_asin", 
    "_sinh", 
    "_tan", 
    "_atan", 
    "_th_tanh", 
    "_erf", 
    "erfinv", 
    "_sqrt", 
    "_rsqrt", 
    "_ceil", 
    "_floor", 
    "_round", 
    "_trunc", 
    "frac", 
    "mean", 
    "var", 
    "unbiased", 
    "std", 
    "norm", 
    "p", 
    "renorm", 
    "maxnorm", 
    "dist", 
    "reciprocal", 
    "neg", 
    "atan2", 
    "pow", 
    "exponent", 
    "base", 
    "lerp", 
    "end", 
    "weight", 
    "histc", 
    "bins", 
    "_sumall", 
    "_th_sum", 
    "_prodall", 
    "_th_prod", 
    "_cumsum", 
    "_cumprod", 
    "sign", 
    "trace", 
    "add", 
    "alpha", 
    "sub", 
    "mul", 
    "div", 
    "fmod", 
    "remainder", 
    "clamp", 
    "clamp_min", 
    "clamp_max", 
    "_dot", 
    "tril", 
    "diagonal", 
    "triu", 
    "cross", 
    "diag", 
    "addmm", 
    "mat1", 
    "mat2", 
    "beta", 
    "_addmv", 
    "mat", 
    "vec", 
    "_addr", 
    "vec1", 
    "vec2", 
    "_ger", 
    "_mv", 
    "_mm", 
    "bmm", 
    "addbmm", 
    "batch1", 
    "batch2", 
    "baddbmm", 
    "addcmul", 
    "tensor1", 
    "tensor2", 
    "value", 
    "addcdiv", 
    "_gesv_single", 
    "A", 
    "solution", 
    "lu", 
    "gels", 
    "res1", 
    "res2", 
    "trtrs", 
    "upper", 
    "transpose", 
    "unitriangular", 
    "symeig", 
    "eigenvectors", 
    "eig", 
    "svd", 
    "some", 
    "res3", 
    "inverse", 
    "output", 
    "potrf", 
    "potrs", 
    "input2", 
    "potri", 
    "pstrf", 
    "tol", 
    "qr", 
    "geqrf", 
    "orgqr", 
    "ormqr", 
    "input3", 
    "left", 
    "btrifact", 
    "pivot", 
    "pivots", 
    "btrifact_with_info", 
    "info", 
    "btrisolve", 
    "LU_data", 
    "LU_pivots", 
    "_dirichlet_grad", 
    "x", 
    "total", 
    "sparse_coo_tensor", 
    "alias", 
    "_sparse_coo_tensor_unsafe", 
    "as_strided", 
    "stride", 
    "_cat", 
    "tensors", 
    "to_dense", 
    "_dimI", 
    "_dimV", 
    "_nnz", 
    "coalesce", 
    "is_coalesced", 
    "_indices", 
    "_values", 
    "hspmm", 
    "binary_cross_entropy", 
    "target", 
    "size_average", 
    "reduce", 
    "binary_cross_entropy_forward", 
    "binary_cross_entropy_backward", 
    "grad_output", 
    "grad_input", 
    "kl_div", 
    "kl_div_forward", 
    "kl_div_backward", 
    "l1_loss", 
    "l1_loss_forward", 
    "l1_loss_backward", 
    "mse_loss", 
    "mse_loss_forward", 
    "mse_loss_backward", 
    "multi_margin_loss", 
    "margin", 
    "multi_margin_loss_forward", 
    "multi_margin_loss_backward", 
    "multilabel_margin_loss", 
    "multilabel_margin_loss_forward", 
    "is_target", 
    "multilabel_margin_loss_backward", 
    "nll_loss", 
    "ignore_index", 
    "nll_loss_forward", 
    "total_weight", 
    "nll_loss_backward", 
    "nll_loss2d", 
    "nll_loss2d_forward", 
    "nll_loss2d_backward", 
    "smooth_l1_loss", 
    "smooth_l1_loss_forward", 
    "smooth_l1_loss_backward", 
    "soft_margin_loss", 
    "soft_margin_loss_forward", 
    "soft_margin_loss_backward", 
    "elu", 
    "scale", 
    "elu_forward", 
    "elu_backward", 
    "glu", 
    "glu_forward", 
    "glu_backward", 
    "hardshrink", 
    "lambd", 
    "hardshrink_forward", 
    "hardshrink_backward", 
    "hardtanh", 
    "min_val", 
    "max_val", 
    "hardtanh_forward", 
    "hardtanh_backward", 
    "leaky_relu", 
    "negative_slope", 
    "leaky_relu_forward", 
    "leaky_relu_backward", 
    "log_sigmoid", 
    "log_sigmoid_forward", 
    "buffer", 
    "log_sigmoid_backward", 
    "prelu", 
    "prelu_forward", 
    "prelu_backward", 
    "output_mask", 
    "grad_weight", 
    "rrelu_with_noise_backward", 
    "noise", 
    "lower", 
    "training", 
    "softplus", 
    "threshold", 
    "softplus_forward", 
    "softplus_backward", 
    "softshrink", 
    "softshrink_forward", 
    "softshrink_backward", 
    "threshold_forward", 
    "threshold_backward", 
    "adaptive_avg_pool2d", 
    "output_size", 
    "adaptive_avg_pool2d_forward", 
    "adaptive_avg_pool2d_backward", 
    "adaptive_avg_pool3d", 
    "adaptive_avg_pool3d_forward", 
    "adaptive_avg_pool3d_backward", 
    "adaptive_max_pool2d", 
    "adaptive_max_pool2d_forward", 
    "adaptive_max_pool2d_backward", 
    "adaptive_max_pool3d", 
    "adaptive_max_pool3d_forward", 
    "adaptive_max_pool3d_backward", 
    "avg_pool2d", 
    "kernel_size", 
    "padding", 
    "ceil_mode", 
    "count_include_pad", 
    "avg_pool2d_forward", 
    "avg_pool2d_backward", 
    "avg_pool3d", 
    "avg_pool3d_forward", 
    "avg_pool3d_backward", 
    "fractional_max_pool2d", 
    "random_samples", 
    "fractional_max_pool2d_forward", 
    "fractional_max_pool2d_backward", 
    "max_pool2d", 
    "dilation", 
    "max_pool2d_forward", 
    "max_pool2d_backward", 
    "max_pool3d", 
    "max_pool3d_forward", 
    "max_pool3d_backward", 
    "max_unpool2d", 
    "max_unpool2d_forward", 
    "max_unpool2d_backward", 
    "max_unpool3d", 
    "max_unpool3d_forward", 
    "max_unpool3d_backward", 
    "reflection_pad1d", 
    "reflection_pad1d_forward", 
    "reflection_pad1d_backward", 
    "reflection_pad2d", 
    "reflection_pad2d_forward", 
    "reflection_pad2d_backward", 
    "replication_pad1d", 
    "replication_pad1d_forward", 
    "replication_pad1d_backward", 
    "replication_pad2d", 
    "replication_pad2d_forward", 
    "replication_pad2d_backward", 
    "replication_pad3d", 
    "replication_pad3d_forward", 
    "replication_pad3d_backward", 
    "upsample_linear1d", 
    "align_corners", 
    "upsample_linear1d_forward", 
    "upsample_linear1d_backward", 
    "input_size", 
    "upsample_bilinear2d", 
    "upsample_bilinear2d_forward", 
    "upsample_bilinear2d_backward", 
    "upsample_trilinear3d", 
    "upsample_trilinear3d_forward", 
    "upsample_trilinear3d_backward", 
    "upsample_nearest1d", 
    "scale_factor", 
    "upsample_nearest1d_forward", 
    "upsample_nearest1d_backward", 
    "upsample_nearest2d", 
    "upsample_nearest2d_forward", 
    "upsample_nearest2d_backward", 
    "upsample_nearest3d", 
    "upsample_nearest3d_forward", 
    "upsample_nearest3d_backward", 
    "_sigmoid", 
    "_sigmoid_forward", 
    "_sigmoid_backward", 
    "_tanh", 
    "_tanh_forward", 
    "_tanh_backward", 
    "thnn_batch_norm", 
    "bias", 
    "running_mean", 
    "running_var", 
    "momentum", 
    "eps", 
    "thnn_batch_norm_forward", 
    "save_mean", 
    "save_std", 
    "thnn_batch_norm_backward", 
    "grad_bias", 
    "thnn_conv_transpose2d", 
    "output_padding", 
    "thnn_conv_transpose2d_forward", 
    "columns", 
    "ones", 
    "thnn_conv_transpose2d_backward", 
    "thnn_conv_transpose3d", 
    "thnn_conv_transpose3d_forward", 
    "finput", 
    "fgrad_input", 
    "thnn_conv_transpose3d_backward", 
    "thnn_conv2d", 
    "thnn_conv2d_forward", 
    "thnn_conv2d_backward", 
    "thnn_conv_depthwise2d", 
    "thnn_conv_depthwise2d_forward", 
    "thnn_conv_depthwise2d_backward", 
    "thnn_conv3d", 
    "thnn_conv3d_forward", 
    "thnn_conv3d_backward", 
    "thnn_conv_dilated2d", 
    "thnn_conv_dilated2d_forward", 
    "thnn_conv_dilated2d_backward", 
    "thnn_conv_dilated3d", 
    "thnn_conv_dilated3d_forward", 
    "thnn_conv_dilated3d_backward", 
    "_cast_uint8_t", 
    "non_blocking", 
    "_cast_int8_t", 
    "_cast_double", 
    "_cast_float", 
    "_cast_int", 
    "_cast_int64_t", 
    "_cast_int16_t", 
    "_cast_Half", 
    "_cudnn_rnn_flatten_weight", 
    "weight_arr", 
    "weight_stride0", 
    "hidden_size", 
    "num_layers", 
    "batch_first", 
    "bidirectional", 
    "abs", 
    "acos", 
    "adaptive_avg_pool1d", 
    "adaptive_max_pool1d", 
    "result0", 
    "result1", 
    "allclose", 
    "rtol", 
    "atol", 
    "equal_nan", 
    "addmv", 
    "addr", 
    "argmax", 
    "_argmax", 
    "argmin", 
    "_argmin", 
    "asin", 
    "atan", 
    "batch_norm", 
    "input", 
    "cudnn_enabled", 
    "bernoulli", 
    "bilinear", 
    "input1", 
    "cat", 
    "ceil", 
    "chunk", 
    "chunks", 
    "cudnn_is_acceptable", 
    "convolution", 
    "transposed", 
    "groups", 
    "_convolution", 
    "benchmark", 
    "deterministic", 
    "_convolution_nogroup", 
    "_convolution_double_backward", 
    "ggI", 
    "ggW", 
    "ggb", 
    "gO", 
    "result2", 
    "conv1d", 
    "conv2d", 
    "conv3d", 
    "conv_tbc", 
    "pad", 
    "conv_tbc_backward", 
    "conv_transpose1d", 
    "conv_transpose2d", 
    "conv_transpose3d", 
    "cos", 
    "cosh", 
    "cosine_embedding_loss", 
    "cudnn_affine_grid_generator", 
    "theta", 
    "N", 
    "C", 
    "H", 
    "W", 
    "grid", 
    "cudnn_affine_grid_generator_backward", 
    "grad", 
    "grad_theta", 
    "cudnn_batch_norm", 
    "exponential_average_factor", 
    "epsilon", 
    "cudnn_batch_norm_backward", 
    "save_var", 
    "cudnn_convolution", 
    "cudnn_convolution_backward_input", 
    "self_size", 
    "cudnn_convolution_backward", 
    "cudnn_convolution_backward_bias", 
    "cudnn_convolution_backward_weight", 
    "weight_size", 
    "cudnn_convolution_transpose", 
    "cudnn_convolution_transpose_backward", 
    "cudnn_convolution_transpose_backward_bias", 
    "cudnn_convolution_transpose_backward_input", 
    "cudnn_convolution_transpose_backward_weight", 
    "cudnn_grid_sampler", 
    "cudnn_grid_sampler_backward", 
    "grad_self", 
    "grad_grid", 
    "cumsum", 
    "cumprod", 
    "det", 
    "diagflat", 
    "offset", 
    "dim1", 
    "dim2", 
    "dot", 
    "embedding", 
    "padding_idx", 
    "scale_grad_by_freq", 
    "sparse", 
    "embedding_backward", 
    "num_weights", 
    "embedding_dense_backward", 
    "embedding_sparse_backward", 
    "embedding_bag", 
    "offsets", 
    "result3", 
    "embedding_bag_backward", 
    "offset2bag", 
    "bag_size", 
    "maximum_indices", 
    "embedding_bag_sparse_backward", 
    "embedding_bag_dense_backward", 
    "empty_like", 
    "erf", 
    "exp", 
    "expm1", 
    "expand", 
    "implicit", 
    "expand_as", 
    "floor", 
    "full_like", 
    "fill_value", 
    "hinge_embedding_loss", 
    "ger", 
    "gesv", 
    "_gesv_helper", 
    "group_norm", 
    "num_groups", 
    "fft", 
    "signal_ndim", 
    "normalized", 
    "ifft", 
    "rfft", 
    "onesided", 
    "irfft", 
    "signal_sizes", 
    "_fft_with_size", 
    "complex_input", 
    "complex_output", 
    "checked_signal_sizes", 
    "output_sizes", 
    "isclose", 
    "is_cuda", 
    "is_distributed", 
    "is_floating_point", 
    "is_nonzero", 
    "is_same_size", 
    "is_signed", 
    "is_sparse", 
    "layer_norm", 
    "normalized_shape", 
    "cudnn_enable", 
    "log", 
    "log10", 
    "log1p", 
    "log2", 
    "logdet", 
    "log_softmax", 
    "log_softmax_backward_data", 
    "logsumexp", 
    "margin_ranking_loss", 
    "matmul", 
    "max_values", 
    "max_pool1d", 
    "min_values", 
    "mkldnn_convolution", 
    "mkldnn_convolution_backward_input", 
    "bias_defined", 
    "mkldnn_convolution_backward_weights", 
    "mkldnn_convolution_backward", 
    "mm", 
    "mv", 
    "narrow", 
    "start", 
    "length", 
    "ones_like", 
    "pairwise_distance", 
    "x1", 
    "x2", 
    "permute", 
    "dims", 
    "pin_memory", 
    "rand_like", 
    "randint_like", 
    "high", 
    "low", 
    "randn_like", 
    "repeat", 
    "repeats", 
    "reshape", 
    "shape", 
    "RoiPooling2d_forward", 
    "rois", 
    "pooledHeight", 
    "pooledWidth", 
    "spatialScale", 
    "RoiPooling2d_backward", 
    "gradOutput", 
    "argmaxes", 
    "round", 
    "relu", 
    "rsqrt", 
    "select", 
    "selu", 
    "sin", 
    "sinh", 
    "slice", 
    "slogdet", 
    "smm", 
    "softmax", 
    "softmax_backward_data", 
    "split", 
    "split_size", 
    "split_with_sizes", 
    "split_sizes", 
    "squeeze", 
    "sspaddmm", 
    "stack", 
    "stft", 
    "frame_length", 
    "hop", 
    "fft_size", 
    "window", 
    "pad_end", 
    "sum", 
    "_sum", 
    "sqrt", 
    "prod", 
    "_prod", 
    "t", 
    "tan", 
    "tanh", 
    "dim0", 
    "_trilinear", 
    "i1", 
    "i2", 
    "i3", 
    "expand1", 
    "expand2", 
    "expand3", 
    "sumdim", 
    "unroll_dim", 
    "triplet_margin_loss", 
    "anchor", 
    "positive", 
    "negative", 
    "swap", 
    "trunc", 
    "type_as", 
    "_unique", 
    "return_inverse", 
    "_unsafe_view", 
    "unsqueeze", 
    "view_as", 
    "where", 
    "condition", 
    "_s_where", 
    "zeros_like", 
    "_standard_gamma_grad", 
    "sizes", 
    "strides",
  };

  // Types
  TypePtr types[] = {
    DynamicType::get(), 
    ListType::ofTensors(),
  };

  // default argument values for all ops, represented as using tensors via as_tensor
  at::optional<at::Tensor> tensors[] = {
    at::nullopt, 
    as_tensor(bool(false)), 
    as_tensor(int64_t(-1)), 
    as_tensor(bool(true)), 
    as_tensor(Scalar(2)), 
    as_tensor(int64_t(100)), 
    as_tensor(Scalar(0)), 
    as_tensor(Scalar(1)), 
    as_tensor(int64_t(0)), 
    as_tensor(Scalar(-1)), 
    at::Tensor(), 
    as_tensor(int64_t(-100)), 
    as_tensor(Scalar(0.5)), 
    as_tensor(Scalar(0.01)), 
    as_tensor(std::array<bool,2>({{true, true}})), 
    as_tensor(Scalar(20)), 
    as_tensor(IntList({})), 
    as_tensor(IntList(0)), 
    as_tensor(IntList(1)), 
    as_tensor(std::array<bool,3>({{true, true, true}})), 
    as_tensor(double(1e-05)), 
    as_tensor(double(1e-08)), 
    as_tensor(int64_t(1)), 
    as_tensor(double(0.0)), 
    as_tensor(double(1.0)), 
    as_tensor(double(2)), 
    as_tensor(double(1e-06)), 
    as_tensor(int64_t(9223372036854775807)),
  };

  // the attribute kind tag for any arguments that have optional attribute encodings
  // in the IR.
  at::optional<AttributeInfo> attributes[] = {
    at::nullopt, 
    AttributeInfo{ AttributeKind::is, at::nullopt }, 
    AttributeInfo{ AttributeKind::i, at::nullopt }, 
    AttributeInfo{ AttributeKind::t, at::nullopt }, 
    AttributeInfo{ AttributeKind::is, 2 }, 
    AttributeInfo{ AttributeKind::is, 3 }, 
    AttributeInfo{ AttributeKind::is, 4 }, 
    AttributeInfo{ AttributeKind::is, 6 }, 
    AttributeInfo{ AttributeKind::is, 1 }, 
    AttributeInfo{ AttributeKind::is, 5 }, 
    AttributeInfo{ AttributeKind::f, at::nullopt },
  };

  // for compound objects, it uses 1 integer per argument to the object's constructor
  // which is an index into one of the above tables
  using ArgumentCtor = uint32_t[4];
  ArgumentCtor arguments[] = {
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 6, 0, 0, 0 }, // Argument("tensor", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 8, 0, 0, 0 }, // Argument("mask", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 13, 0, 0, 1 }, // Argument("size", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 16, 0, 0, 0 }, // Argument("index", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 16, 0, 0, 0 }, // Argument("index", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 19, 0, 0, 2 }, // Argument("dimension", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 13, 0, 0, 2 }, // Argument("size", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 20, 0, 0, 2 }, // Argument("step", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 16, 0, 0, 0 }, // Argument("index", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 41, 0, 0, 0 }, // Argument("min", at::nullopt, at::nullopt, DynamicType::get()) 
    { 43, 0, 0, 0 }, // Argument("min_indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 44, 0, 0, 0 }, // Argument("max", at::nullopt, at::nullopt, DynamicType::get()) 
    { 45, 0, 0, 0 }, // Argument("max_indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 47, 0, 0, 2 }, // Argument("k", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 15, 0, 2, 2 }, // Argument("dim", as_tensor(int64_t(-1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 48, 0, 0, 0 }, // Argument("values", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 2, 2 }, // Argument("dim", as_tensor(int64_t(-1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 48, 0, 0, 0 }, // Argument("values", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 48, 0, 0, 0 }, // Argument("values", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 2, 2 }, // Argument("dim", as_tensor(int64_t(-1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 53, 0, 1, 2 }, // Argument("descending", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 48, 0, 0, 0 }, // Argument("values", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 47, 0, 0, 2 }, // Argument("k", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 15, 0, 2, 2 }, // Argument("dim", as_tensor(int64_t(-1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 55, 0, 3, 2 }, // Argument("largest", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 56, 0, 3, 2 }, // Argument("sorted", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 48, 0, 0, 0 }, // Argument("values", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 69, 0, 0, 2 }, // Argument("n", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 92, 0, 3, 2 }, // Argument("unbiased", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 92, 0, 3, 2 }, // Argument("unbiased", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 92, 0, 3, 2 }, // Argument("unbiased", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 92, 0, 3, 2 }, // Argument("unbiased", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 95, 0, 0, 3 }, // Argument("p", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 95, 0, 4, 3 }, // Argument("p", as_tensor(Scalar(2)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 95, 0, 0, 3 }, // Argument("p", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 97, 0, 0, 3 }, // Argument("maxnorm", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 95, 0, 4, 3 }, // Argument("p", as_tensor(Scalar(2)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 103, 0, 0, 3 }, // Argument("exponent", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 103, 0, 0, 0 }, // Argument("exponent", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 104, 0, 0, 3 }, // Argument("base", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 106, 0, 0, 0 }, // Argument("end", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 3 }, // Argument("weight", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 109, 0, 5, 2 }, // Argument("bins", as_tensor(int64_t(100)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 41, 0, 6, 3 }, // Argument("min", as_tensor(Scalar(0)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 44, 0, 6, 3 }, // Argument("max", as_tensor(Scalar(0)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 119, 0, 7, 3 }, // Argument("alpha", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 119, 0, 7, 3 }, // Argument("alpha", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 119, 0, 7, 3 }, // Argument("alpha", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 119, 0, 7, 3 }, // Argument("alpha", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 3 }, // Argument("other", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 41, 0, 0, 3 }, // Argument("min", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 44, 0, 0, 3 }, // Argument("max", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 41, 0, 0, 3 }, // Argument("min", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 44, 0, 0, 3 }, // Argument("max", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 6, 0, 0, 0 }, // Argument("tensor", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 130, 0, 8, 2 }, // Argument("diagonal", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 130, 0, 8, 2 }, // Argument("diagonal", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 2, 2 }, // Argument("dim", as_tensor(int64_t(-1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 130, 0, 8, 2 }, // Argument("diagonal", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 135, 0, 0, 0 }, // Argument("mat1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 136, 0, 0, 0 }, // Argument("mat2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 137, 0, 7, 3 }, // Argument("beta", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 119, 0, 7, 3 }, // Argument("alpha", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 139, 0, 0, 0 }, // Argument("mat", at::nullopt, at::nullopt, DynamicType::get()) 
    { 140, 0, 0, 0 }, // Argument("vec", at::nullopt, at::nullopt, DynamicType::get()) 
    { 137, 0, 7, 3 }, // Argument("beta", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 119, 0, 7, 3 }, // Argument("alpha", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 142, 0, 0, 0 }, // Argument("vec1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 143, 0, 0, 0 }, // Argument("vec2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 137, 0, 7, 3 }, // Argument("beta", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 119, 0, 7, 3 }, // Argument("alpha", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 143, 0, 0, 0 }, // Argument("vec2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 140, 0, 0, 0 }, // Argument("vec", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 136, 0, 0, 0 }, // Argument("mat2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 136, 0, 0, 0 }, // Argument("mat2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 149, 0, 0, 0 }, // Argument("batch1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 150, 0, 0, 0 }, // Argument("batch2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 137, 0, 7, 3 }, // Argument("beta", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 119, 0, 7, 3 }, // Argument("alpha", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 149, 0, 0, 0 }, // Argument("batch1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 150, 0, 0, 0 }, // Argument("batch2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 137, 0, 7, 3 }, // Argument("beta", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 119, 0, 7, 3 }, // Argument("alpha", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 153, 0, 0, 0 }, // Argument("tensor1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 154, 0, 0, 0 }, // Argument("tensor2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 155, 0, 7, 3 }, // Argument("value", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 153, 0, 0, 0 }, // Argument("tensor1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 154, 0, 0, 0 }, // Argument("tensor2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 155, 0, 7, 3 }, // Argument("value", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 158, 0, 0, 0 }, // Argument("A", at::nullopt, at::nullopt, DynamicType::get()) 
    { 159, 0, 0, 0 }, // Argument("solution", at::nullopt, at::nullopt, DynamicType::get()) 
    { 160, 0, 0, 0 }, // Argument("lu", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 158, 0, 0, 0 }, // Argument("A", at::nullopt, at::nullopt, DynamicType::get()) 
    { 162, 0, 0, 0 }, // Argument("res1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 163, 0, 0, 0 }, // Argument("res2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 158, 0, 0, 0 }, // Argument("A", at::nullopt, at::nullopt, DynamicType::get()) 
    { 165, 0, 3, 2 }, // Argument("upper", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 166, 0, 1, 2 }, // Argument("transpose", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 167, 0, 1, 2 }, // Argument("unitriangular", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 162, 0, 0, 0 }, // Argument("res1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 163, 0, 0, 0 }, // Argument("res2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 169, 0, 1, 2 }, // Argument("eigenvectors", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 165, 0, 3, 2 }, // Argument("upper", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 162, 0, 0, 0 }, // Argument("res1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 163, 0, 0, 0 }, // Argument("res2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 169, 0, 1, 2 }, // Argument("eigenvectors", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 162, 0, 0, 0 }, // Argument("res1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 163, 0, 0, 0 }, // Argument("res2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 172, 0, 3, 2 }, // Argument("some", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 162, 0, 0, 0 }, // Argument("res1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 163, 0, 0, 0 }, // Argument("res2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 173, 0, 0, 0 }, // Argument("res3", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 165, 0, 3, 2 }, // Argument("upper", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 178, 0, 0, 0 }, // Argument("input2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 165, 0, 3, 2 }, // Argument("upper", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 165, 0, 3, 2 }, // Argument("upper", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 165, 0, 3, 2 }, // Argument("upper", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 181, 0, 9, 3 }, // Argument("tol", as_tensor(Scalar(-1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 162, 0, 0, 0 }, // Argument("res1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 163, 0, 0, 0 }, // Argument("res2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 162, 0, 0, 0 }, // Argument("res1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 163, 0, 0, 0 }, // Argument("res2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 162, 0, 0, 0 }, // Argument("res1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 163, 0, 0, 0 }, // Argument("res2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 178, 0, 0, 0 }, // Argument("input2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 178, 0, 0, 0 }, // Argument("input2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 186, 0, 0, 0 }, // Argument("input3", at::nullopt, at::nullopt, DynamicType::get()) 
    { 187, 0, 3, 2 }, // Argument("left", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 166, 0, 1, 2 }, // Argument("transpose", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 189, 0, 3, 2 }, // Argument("pivot", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 190, 0, 0, 0 }, // Argument("pivots", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 189, 0, 3, 2 }, // Argument("pivot", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 190, 0, 0, 0 }, // Argument("pivots", at::nullopt, at::nullopt, DynamicType::get()) 
    { 192, 0, 0, 0 }, // Argument("info", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 194, 0, 0, 0 }, // Argument("LU_data", at::nullopt, at::nullopt, DynamicType::get()) 
    { 195, 0, 0, 0 }, // Argument("LU_pivots", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 197, 0, 0, 0 }, // Argument("x", at::nullopt, at::nullopt, DynamicType::get()) 
    { 119, 0, 0, 0 }, // Argument("alpha", at::nullopt, at::nullopt, DynamicType::get()) 
    { 198, 0, 0, 0 }, // Argument("total", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 48, 0, 0, 0 }, // Argument("values", at::nullopt, at::nullopt, DynamicType::get()) 
    { 13, 0, 0, 1 }, // Argument("size", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 48, 0, 0, 0 }, // Argument("values", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 48, 0, 0, 0 }, // Argument("values", at::nullopt, at::nullopt, DynamicType::get()) 
    { 13, 0, 0, 1 }, // Argument("size", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 13, 0, 0, 1 }, // Argument("size", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 0, 0, 2, 2 }, // Argument("storage_offset", as_tensor(int64_t(-1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 205, 1, 0, 0 }, // Argument("tensors", at::nullopt, at::nullopt, ListType::ofTensors()) 
    { 15, 0, 8, 2 }, // Argument("dim", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 135, 0, 0, 0 }, // Argument("mat1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 136, 0, 0, 0 }, // Argument("mat2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 10, 0 }, // Argument("weight", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 217, 0, 3, 2 }, // Argument("size_average", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 3, 2 }, // Argument("reduce", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 3, 2 }, // Argument("size_average", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 3, 2 }, // Argument("reduce", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 3, 2 }, // Argument("size_average", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 3, 2 }, // Argument("reduce", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 3, 2 }, // Argument("size_average", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 3, 2 }, // Argument("reduce", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 95, 0, 7, 3 }, // Argument("p", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 233, 0, 7, 3 }, // Argument("margin", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 107, 0, 10, 0 }, // Argument("weight", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 217, 0, 3, 2 }, // Argument("size_average", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 3, 2 }, // Argument("reduce", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 95, 0, 0, 3 }, // Argument("p", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 233, 0, 0, 3 }, // Argument("margin", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 95, 0, 0, 3 }, // Argument("p", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 233, 0, 0, 3 }, // Argument("margin", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 3, 2 }, // Argument("size_average", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 3, 2 }, // Argument("reduce", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 238, 0, 0, 0 }, // Argument("is_target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 238, 0, 0, 0 }, // Argument("is_target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 10, 0 }, // Argument("weight", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 217, 0, 3, 2 }, // Argument("size_average", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 241, 0, 11, 2 }, // Argument("ignore_index", as_tensor(int64_t(-100)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 3, 2 }, // Argument("reduce", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 241, 0, 0, 2 }, // Argument("ignore_index", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 243, 0, 0, 0 }, // Argument("total_weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 241, 0, 0, 2 }, // Argument("ignore_index", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 243, 0, 0, 0 }, // Argument("total_weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 10, 0 }, // Argument("weight", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 217, 0, 3, 2 }, // Argument("size_average", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 241, 0, 11, 2 }, // Argument("ignore_index", as_tensor(int64_t(-100)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 3, 2 }, // Argument("reduce", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 241, 0, 0, 2 }, // Argument("ignore_index", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 243, 0, 0, 0 }, // Argument("total_weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 241, 0, 0, 2 }, // Argument("ignore_index", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 243, 0, 0, 0 }, // Argument("total_weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 3, 2 }, // Argument("size_average", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 3, 2 }, // Argument("reduce", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 3, 2 }, // Argument("size_average", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 3, 2 }, // Argument("reduce", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 217, 0, 0, 2 }, // Argument("size_average", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 0, 2 }, // Argument("reduce", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 119, 0, 7, 3 }, // Argument("alpha", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 255, 0, 7, 3 }, // Argument("scale", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 119, 0, 0, 3 }, // Argument("alpha", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 255, 0, 0, 3 }, // Argument("scale", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 119, 0, 0, 3 }, // Argument("alpha", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 255, 0, 0, 3 }, // Argument("scale", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 2, 2 }, // Argument("dim", as_tensor(int64_t(-1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 262, 0, 12, 3 }, // Argument("lambd", as_tensor(Scalar(0.5)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 262, 0, 0, 3 }, // Argument("lambd", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 262, 0, 0, 3 }, // Argument("lambd", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 266, 0, 9, 3 }, // Argument("min_val", as_tensor(Scalar(-1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 267, 0, 7, 3 }, // Argument("max_val", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 266, 0, 0, 3 }, // Argument("min_val", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 267, 0, 0, 3 }, // Argument("max_val", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 266, 0, 0, 3 }, // Argument("min_val", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 267, 0, 0, 3 }, // Argument("max_val", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 271, 0, 13, 3 }, // Argument("negative_slope", as_tensor(Scalar(0.01)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 271, 0, 0, 3 }, // Argument("negative_slope", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 271, 0, 0, 3 }, // Argument("negative_slope", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 276, 0, 0, 0 }, // Argument("buffer", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 276, 0, 0, 0 }, // Argument("buffer", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 281, 0, 14, 1 }, // Argument("output_mask", as_tensor(std::array<bool,2>({{true, true}})), AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 282, 0, 0, 0 }, // Argument("grad_weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 284, 0, 0, 0 }, // Argument("noise", at::nullopt, at::nullopt, DynamicType::get()) 
    { 285, 0, 0, 3 }, // Argument("lower", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 165, 0, 0, 3 }, // Argument("upper", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 286, 0, 0, 2 }, // Argument("training", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 137, 0, 7, 3 }, // Argument("beta", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 288, 0, 15, 3 }, // Argument("threshold", as_tensor(Scalar(20)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 137, 0, 0, 3 }, // Argument("beta", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 288, 0, 0, 3 }, // Argument("threshold", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 137, 0, 0, 3 }, // Argument("beta", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 288, 0, 0, 3 }, // Argument("threshold", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 262, 0, 12, 3 }, // Argument("lambd", as_tensor(Scalar(0.5)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 262, 0, 0, 3 }, // Argument("lambd", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 262, 0, 0, 3 }, // Argument("lambd", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 288, 0, 0, 3 }, // Argument("threshold", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 155, 0, 0, 3 }, // Argument("value", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 288, 0, 0, 3 }, // Argument("threshold", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 155, 0, 0, 3 }, // Argument("value", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 288, 0, 0, 3 }, // Argument("threshold", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 155, 0, 0, 3 }, // Argument("value", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 4 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 4 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 5 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 5 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 4 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 4 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 5 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 5 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 203, 0, 16, 4 }, // Argument("stride", as_tensor(IntList({})), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 17, 4 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 312, 0, 1, 2 }, // Argument("ceil_mode", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 313, 0, 1, 2 }, // Argument("count_include_pad", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 203, 0, 0, 4 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 312, 0, 0, 2 }, // Argument("ceil_mode", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 313, 0, 0, 2 }, // Argument("count_include_pad", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 203, 0, 0, 4 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 312, 0, 0, 2 }, // Argument("ceil_mode", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 313, 0, 0, 2 }, // Argument("count_include_pad", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 203, 0, 16, 5 }, // Argument("stride", as_tensor(IntList({})), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 17, 5 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 312, 0, 1, 2 }, // Argument("ceil_mode", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 313, 0, 1, 2 }, // Argument("count_include_pad", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 203, 0, 0, 5 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 0, 5 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 312, 0, 0, 2 }, // Argument("ceil_mode", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 313, 0, 0, 2 }, // Argument("count_include_pad", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 203, 0, 0, 5 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 0, 5 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 312, 0, 0, 2 }, // Argument("ceil_mode", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 313, 0, 0, 2 }, // Argument("count_include_pad", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 297, 0, 0, 4 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 320, 0, 0, 0 }, // Argument("random_samples", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 297, 0, 0, 4 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 320, 0, 0, 0 }, // Argument("random_samples", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 297, 0, 0, 4 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 203, 0, 16, 4 }, // Argument("stride", as_tensor(IntList({})), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 17, 4 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 324, 0, 18, 4 }, // Argument("dilation", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 312, 0, 1, 2 }, // Argument("ceil_mode", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 203, 0, 0, 4 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 324, 0, 0, 4 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 312, 0, 0, 2 }, // Argument("ceil_mode", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 203, 0, 0, 4 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 324, 0, 0, 4 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 312, 0, 0, 2 }, // Argument("ceil_mode", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 203, 0, 16, 5 }, // Argument("stride", as_tensor(IntList({})), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 17, 5 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 324, 0, 18, 5 }, // Argument("dilation", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 312, 0, 1, 2 }, // Argument("ceil_mode", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 203, 0, 0, 5 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 0, 5 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 324, 0, 0, 5 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 312, 0, 0, 2 }, // Argument("ceil_mode", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 203, 0, 0, 5 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 0, 5 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 324, 0, 0, 5 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 312, 0, 0, 2 }, // Argument("ceil_mode", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 4 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 4 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 4 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 5 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 203, 0, 0, 5 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 0, 5 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 5 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 203, 0, 0, 5 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 0, 5 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 5 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 203, 0, 0, 5 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 0, 5 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 6 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 4 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 6 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 4 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 6 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 4 }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 6 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 4 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 6 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 4 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 6 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 4 }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 7 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 6 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 7 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 6 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 7 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 6 }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 8 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 352, 0, 0, 2 }, // Argument("align_corners", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 8 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 352, 0, 0, 2 }, // Argument("align_corners", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 8 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 355, 0, 0, 5 }, // Argument("input_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 352, 0, 0, 2 }, // Argument("align_corners", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 4 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 352, 0, 0, 2 }, // Argument("align_corners", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 4 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 352, 0, 0, 2 }, // Argument("align_corners", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 4 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 355, 0, 0, 6 }, // Argument("input_size", at::nullopt, AttributeInfo{ AttributeKind::is, 4 }, DynamicType::get()) 
    { 352, 0, 0, 2 }, // Argument("align_corners", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 5 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 352, 0, 0, 2 }, // Argument("align_corners", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 5 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 352, 0, 0, 2 }, // Argument("align_corners", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 5 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 355, 0, 0, 9 }, // Argument("input_size", at::nullopt, AttributeInfo{ AttributeKind::is, 5 }, DynamicType::get()) 
    { 352, 0, 0, 2 }, // Argument("align_corners", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 363, 0, 0, 2 }, // Argument("scale_factor", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 363, 0, 0, 2 }, // Argument("scale_factor", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 363, 0, 0, 2 }, // Argument("scale_factor", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 363, 0, 0, 2 }, // Argument("scale_factor", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 363, 0, 0, 2 }, // Argument("scale_factor", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 363, 0, 0, 2 }, // Argument("scale_factor", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 363, 0, 0, 2 }, // Argument("scale_factor", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 363, 0, 0, 2 }, // Argument("scale_factor", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 363, 0, 0, 2 }, // Argument("scale_factor", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 380, 0, 0, 0 }, // Argument("running_mean", at::nullopt, at::nullopt, DynamicType::get()) 
    { 381, 0, 0, 0 }, // Argument("running_var", at::nullopt, at::nullopt, DynamicType::get()) 
    { 286, 0, 0, 2 }, // Argument("training", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 382, 0, 0, 10 }, // Argument("momentum", at::nullopt, AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 383, 0, 0, 10 }, // Argument("eps", at::nullopt, AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 380, 0, 0, 0 }, // Argument("running_mean", at::nullopt, at::nullopt, DynamicType::get()) 
    { 381, 0, 0, 0 }, // Argument("running_var", at::nullopt, at::nullopt, DynamicType::get()) 
    { 286, 0, 0, 2 }, // Argument("training", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 382, 0, 0, 10 }, // Argument("momentum", at::nullopt, AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 383, 0, 0, 10 }, // Argument("eps", at::nullopt, AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 385, 0, 0, 0 }, // Argument("save_mean", at::nullopt, at::nullopt, DynamicType::get()) 
    { 386, 0, 0, 0 }, // Argument("save_std", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 380, 0, 0, 0 }, // Argument("running_mean", at::nullopt, at::nullopt, DynamicType::get()) 
    { 381, 0, 0, 0 }, // Argument("running_var", at::nullopt, at::nullopt, DynamicType::get()) 
    { 286, 0, 0, 2 }, // Argument("training", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 383, 0, 0, 10 }, // Argument("eps", at::nullopt, AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 385, 0, 0, 0 }, // Argument("save_mean", at::nullopt, at::nullopt, DynamicType::get()) 
    { 386, 0, 0, 0 }, // Argument("save_std", at::nullopt, at::nullopt, DynamicType::get()) 
    { 281, 0, 19, 1 }, // Argument("output_mask", as_tensor(std::array<bool,3>({{true, true, true}})), AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 282, 0, 0, 0 }, // Argument("grad_weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 388, 0, 0, 0 }, // Argument("grad_bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 203, 0, 18, 4 }, // Argument("stride", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 17, 4 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 390, 0, 17, 4 }, // Argument("output_padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 324, 0, 18, 4 }, // Argument("dilation", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 203, 0, 0, 4 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 390, 0, 0, 4 }, // Argument("output_padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 324, 0, 0, 4 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 392, 0, 0, 0 }, // Argument("columns", at::nullopt, at::nullopt, DynamicType::get()) 
    { 393, 0, 0, 0 }, // Argument("ones", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 203, 0, 0, 4 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 390, 0, 0, 4 }, // Argument("output_padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 324, 0, 0, 4 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 392, 0, 0, 0 }, // Argument("columns", at::nullopt, at::nullopt, DynamicType::get()) 
    { 393, 0, 0, 0 }, // Argument("ones", at::nullopt, at::nullopt, DynamicType::get()) 
    { 281, 0, 19, 1 }, // Argument("output_mask", as_tensor(std::array<bool,3>({{true, true, true}})), AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 282, 0, 0, 0 }, // Argument("grad_weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 388, 0, 0, 0 }, // Argument("grad_bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 203, 0, 18, 5 }, // Argument("stride", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 17, 5 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 390, 0, 17, 5 }, // Argument("output_padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 324, 0, 18, 5 }, // Argument("dilation", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 203, 0, 0, 5 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 0, 5 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 390, 0, 0, 5 }, // Argument("output_padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 324, 0, 0, 5 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 397, 0, 0, 0 }, // Argument("finput", at::nullopt, at::nullopt, DynamicType::get()) 
    { 398, 0, 0, 0 }, // Argument("fgrad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 203, 0, 0, 5 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 0, 5 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 390, 0, 0, 5 }, // Argument("output_padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 324, 0, 0, 5 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 397, 0, 0, 0 }, // Argument("finput", at::nullopt, at::nullopt, DynamicType::get()) 
    { 398, 0, 0, 0 }, // Argument("fgrad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 281, 0, 19, 1 }, // Argument("output_mask", as_tensor(std::array<bool,3>({{true, true, true}})), AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 282, 0, 0, 0 }, // Argument("grad_weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 388, 0, 0, 0 }, // Argument("grad_bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 203, 0, 18, 4 }, // Argument("stride", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 17, 4 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 203, 0, 0, 4 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 397, 0, 0, 0 }, // Argument("finput", at::nullopt, at::nullopt, DynamicType::get()) 
    { 398, 0, 0, 0 }, // Argument("fgrad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 203, 0, 0, 4 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 397, 0, 0, 0 }, // Argument("finput", at::nullopt, at::nullopt, DynamicType::get()) 
    { 398, 0, 0, 0 }, // Argument("fgrad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 281, 0, 19, 1 }, // Argument("output_mask", as_tensor(std::array<bool,3>({{true, true, true}})), AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 282, 0, 0, 0 }, // Argument("grad_weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 388, 0, 0, 0 }, // Argument("grad_bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 203, 0, 18, 4 }, // Argument("stride", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 17, 4 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 324, 0, 18, 4 }, // Argument("dilation", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 203, 0, 0, 4 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 324, 0, 0, 4 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 203, 0, 0, 4 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 324, 0, 0, 4 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 281, 0, 14, 1 }, // Argument("output_mask", as_tensor(std::array<bool,2>({{true, true}})), AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 282, 0, 0, 0 }, // Argument("grad_weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 203, 0, 18, 5 }, // Argument("stride", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 17, 5 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 203, 0, 0, 5 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 0, 5 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 397, 0, 0, 0 }, // Argument("finput", at::nullopt, at::nullopt, DynamicType::get()) 
    { 398, 0, 0, 0 }, // Argument("fgrad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 203, 0, 0, 5 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 0, 5 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 397, 0, 0, 0 }, // Argument("finput", at::nullopt, at::nullopt, DynamicType::get()) 
    { 398, 0, 0, 0 }, // Argument("fgrad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 281, 0, 19, 1 }, // Argument("output_mask", as_tensor(std::array<bool,3>({{true, true, true}})), AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 282, 0, 0, 0 }, // Argument("grad_weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 388, 0, 0, 0 }, // Argument("grad_bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 203, 0, 18, 4 }, // Argument("stride", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 17, 4 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 324, 0, 18, 4 }, // Argument("dilation", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 203, 0, 0, 4 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 324, 0, 0, 4 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 392, 0, 0, 0 }, // Argument("columns", at::nullopt, at::nullopt, DynamicType::get()) 
    { 393, 0, 0, 0 }, // Argument("ones", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 4 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 203, 0, 0, 4 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 0, 4 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 324, 0, 0, 4 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 392, 0, 0, 0 }, // Argument("columns", at::nullopt, at::nullopt, DynamicType::get()) 
    { 393, 0, 0, 0 }, // Argument("ones", at::nullopt, at::nullopt, DynamicType::get()) 
    { 281, 0, 19, 1 }, // Argument("output_mask", as_tensor(std::array<bool,3>({{true, true, true}})), AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 282, 0, 0, 0 }, // Argument("grad_weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 388, 0, 0, 0 }, // Argument("grad_bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 203, 0, 18, 5 }, // Argument("stride", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 17, 5 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 324, 0, 18, 5 }, // Argument("dilation", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 203, 0, 0, 5 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 0, 5 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 324, 0, 0, 5 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 392, 0, 0, 0 }, // Argument("columns", at::nullopt, at::nullopt, DynamicType::get()) 
    { 393, 0, 0, 0 }, // Argument("ones", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 5 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 203, 0, 0, 5 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 0, 5 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 324, 0, 0, 5 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 392, 0, 0, 0 }, // Argument("columns", at::nullopt, at::nullopt, DynamicType::get()) 
    { 393, 0, 0, 0 }, // Argument("ones", at::nullopt, at::nullopt, DynamicType::get()) 
    { 281, 0, 19, 1 }, // Argument("output_mask", as_tensor(std::array<bool,3>({{true, true, true}})), AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 222, 0, 0, 0 }, // Argument("grad_input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 282, 0, 0, 0 }, // Argument("grad_weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 388, 0, 0, 0 }, // Argument("grad_bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 416, 0, 1, 2 }, // Argument("non_blocking", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 416, 0, 1, 2 }, // Argument("non_blocking", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 416, 0, 1, 2 }, // Argument("non_blocking", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 416, 0, 1, 2 }, // Argument("non_blocking", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 416, 0, 1, 2 }, // Argument("non_blocking", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 416, 0, 1, 2 }, // Argument("non_blocking", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 416, 0, 1, 2 }, // Argument("non_blocking", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 416, 0, 1, 2 }, // Argument("non_blocking", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 425, 1, 0, 0 }, // Argument("weight_arr", at::nullopt, at::nullopt, ListType::ofTensors()) 
    { 426, 0, 0, 2 }, // Argument("weight_stride0", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 355, 0, 0, 2 }, // Argument("input_size", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 50, 0, 0, 2 }, // Argument("mode", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 427, 0, 0, 2 }, // Argument("hidden_size", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 428, 0, 0, 2 }, // Argument("num_layers", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 429, 0, 0, 2 }, // Argument("batch_first", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 430, 0, 0, 2 }, // Argument("bidirectional", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 8 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 297, 0, 0, 8 }, // Argument("output_size", at::nullopt, AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 438, 0, 20, 10 }, // Argument("rtol", as_tensor(double(1e-05)), AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 439, 0, 21, 10 }, // Argument("atol", as_tensor(double(1e-08)), AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 440, 0, 1, 2 }, // Argument("equal_nan", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 139, 0, 0, 0 }, // Argument("mat", at::nullopt, at::nullopt, DynamicType::get()) 
    { 140, 0, 0, 0 }, // Argument("vec", at::nullopt, at::nullopt, DynamicType::get()) 
    { 137, 0, 7, 3 }, // Argument("beta", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 119, 0, 7, 3 }, // Argument("alpha", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 142, 0, 0, 0 }, // Argument("vec1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 143, 0, 0, 0 }, // Argument("vec2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 137, 0, 7, 3 }, // Argument("beta", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 119, 0, 7, 3 }, // Argument("alpha", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 380, 0, 0, 0 }, // Argument("running_mean", at::nullopt, at::nullopt, DynamicType::get()) 
    { 381, 0, 0, 0 }, // Argument("running_var", at::nullopt, at::nullopt, DynamicType::get()) 
    { 286, 0, 0, 2 }, // Argument("training", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 382, 0, 0, 10 }, // Argument("momentum", at::nullopt, AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 383, 0, 0, 10 }, // Argument("eps", at::nullopt, AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 451, 0, 0, 2 }, // Argument("cudnn_enabled", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 454, 0, 0, 0 }, // Argument("input1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 178, 0, 0, 0 }, // Argument("input2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 205, 1, 0, 0 }, // Argument("tensors", at::nullopt, at::nullopt, ListType::ofTensors()) 
    { 15, 0, 8, 2 }, // Argument("dim", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 458, 0, 0, 2 }, // Argument("chunks", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 15, 0, 8, 2 }, // Argument("dim", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 461, 0, 0, 2 }, // Argument("transposed", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 390, 0, 0, 1 }, // Argument("output_padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 462, 0, 0, 2 }, // Argument("groups", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 461, 0, 0, 2 }, // Argument("transposed", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 390, 0, 0, 1 }, // Argument("output_padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 462, 0, 0, 2 }, // Argument("groups", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 464, 0, 0, 2 }, // Argument("benchmark", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 465, 0, 0, 2 }, // Argument("deterministic", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 451, 0, 0, 2 }, // Argument("cudnn_enabled", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 461, 0, 0, 2 }, // Argument("transposed", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 390, 0, 0, 1 }, // Argument("output_padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 468, 0, 0, 0 }, // Argument("ggI", at::nullopt, at::nullopt, DynamicType::get()) 
    { 469, 0, 0, 0 }, // Argument("ggW", at::nullopt, at::nullopt, DynamicType::get()) 
    { 470, 0, 0, 0 }, // Argument("ggb", at::nullopt, at::nullopt, DynamicType::get()) 
    { 471, 0, 0, 0 }, // Argument("gO", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 461, 0, 0, 2 }, // Argument("transposed", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 390, 0, 0, 1 }, // Argument("output_padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 462, 0, 0, 2 }, // Argument("groups", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 464, 0, 0, 2 }, // Argument("benchmark", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 465, 0, 0, 2 }, // Argument("deterministic", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 451, 0, 0, 2 }, // Argument("cudnn_enabled", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 281, 0, 0, 1 }, // Argument("output_mask", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 472, 0, 0, 0 }, // Argument("result2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 203, 0, 18, 8 }, // Argument("stride", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 311, 0, 17, 8 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 324, 0, 18, 8 }, // Argument("dilation", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 462, 0, 22, 2 }, // Argument("groups", as_tensor(int64_t(1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 203, 0, 18, 4 }, // Argument("stride", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 17, 4 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 324, 0, 18, 4 }, // Argument("dilation", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 462, 0, 22, 2 }, // Argument("groups", as_tensor(int64_t(1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 203, 0, 18, 5 }, // Argument("stride", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 17, 5 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 324, 0, 18, 5 }, // Argument("dilation", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 462, 0, 22, 2 }, // Argument("groups", as_tensor(int64_t(1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 477, 0, 0, 2 }, // Argument("pad", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 477, 0, 0, 2 }, // Argument("pad", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 472, 0, 0, 0 }, // Argument("result2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 203, 0, 18, 8 }, // Argument("stride", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 311, 0, 17, 8 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 390, 0, 17, 8 }, // Argument("output_padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 462, 0, 22, 2 }, // Argument("groups", as_tensor(int64_t(1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 324, 0, 18, 8 }, // Argument("dilation", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 203, 0, 18, 4 }, // Argument("stride", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 311, 0, 17, 4 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 390, 0, 17, 4 }, // Argument("output_padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 462, 0, 22, 2 }, // Argument("groups", as_tensor(int64_t(1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 324, 0, 18, 4 }, // Argument("dilation", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 2 }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 203, 0, 18, 5 }, // Argument("stride", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 311, 0, 17, 5 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 390, 0, 17, 5 }, // Argument("output_padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 462, 0, 22, 2 }, // Argument("groups", as_tensor(int64_t(1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 324, 0, 18, 5 }, // Argument("dilation", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 3 }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 454, 0, 0, 0 }, // Argument("input1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 178, 0, 0, 0 }, // Argument("input2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 233, 0, 23, 10 }, // Argument("margin", as_tensor(double(0.0)), AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 217, 0, 3, 2 }, // Argument("size_average", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 3, 2 }, // Argument("reduce", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 486, 0, 0, 0 }, // Argument("theta", at::nullopt, at::nullopt, DynamicType::get()) 
    { 487, 0, 0, 2 }, // Argument("N", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 488, 0, 0, 2 }, // Argument("C", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 489, 0, 0, 2 }, // Argument("H", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 490, 0, 0, 2 }, // Argument("W", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 491, 0, 0, 0 }, // Argument("grid", at::nullopt, at::nullopt, DynamicType::get()) 
    { 493, 0, 0, 0 }, // Argument("grad", at::nullopt, at::nullopt, DynamicType::get()) 
    { 487, 0, 0, 2 }, // Argument("N", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 488, 0, 0, 2 }, // Argument("C", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 489, 0, 0, 2 }, // Argument("H", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 490, 0, 0, 2 }, // Argument("W", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 494, 0, 0, 0 }, // Argument("grad_theta", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 380, 0, 0, 0 }, // Argument("running_mean", at::nullopt, at::nullopt, DynamicType::get()) 
    { 381, 0, 0, 0 }, // Argument("running_var", at::nullopt, at::nullopt, DynamicType::get()) 
    { 286, 0, 0, 2 }, // Argument("training", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 496, 0, 0, 10 }, // Argument("exponential_average_factor", at::nullopt, AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 497, 0, 0, 10 }, // Argument("epsilon", at::nullopt, AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 472, 0, 0, 0 }, // Argument("result2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 380, 0, 0, 0 }, // Argument("running_mean", at::nullopt, at::nullopt, DynamicType::get()) 
    { 381, 0, 0, 0 }, // Argument("running_var", at::nullopt, at::nullopt, DynamicType::get()) 
    { 385, 0, 0, 0 }, // Argument("save_mean", at::nullopt, at::nullopt, DynamicType::get()) 
    { 499, 0, 0, 0 }, // Argument("save_var", at::nullopt, at::nullopt, DynamicType::get()) 
    { 497, 0, 0, 10 }, // Argument("epsilon", at::nullopt, AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 472, 0, 0, 0 }, // Argument("result2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 462, 0, 0, 2 }, // Argument("groups", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 464, 0, 0, 2 }, // Argument("benchmark", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 465, 0, 0, 2 }, // Argument("deterministic", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 502, 0, 0, 1 }, // Argument("self_size", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 462, 0, 0, 2 }, // Argument("groups", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 464, 0, 0, 2 }, // Argument("benchmark", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 465, 0, 0, 2 }, // Argument("deterministic", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 462, 0, 0, 2 }, // Argument("groups", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 464, 0, 0, 2 }, // Argument("benchmark", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 465, 0, 0, 2 }, // Argument("deterministic", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 281, 0, 0, 1 }, // Argument("output_mask", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 472, 0, 0, 0 }, // Argument("result2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 506, 0, 0, 1 }, // Argument("weight_size", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 462, 0, 0, 2 }, // Argument("groups", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 464, 0, 0, 2 }, // Argument("benchmark", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 465, 0, 0, 2 }, // Argument("deterministic", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 390, 0, 0, 1 }, // Argument("output_padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 462, 0, 0, 2 }, // Argument("groups", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 464, 0, 0, 2 }, // Argument("benchmark", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 465, 0, 0, 2 }, // Argument("deterministic", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 390, 0, 0, 1 }, // Argument("output_padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 462, 0, 0, 2 }, // Argument("groups", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 464, 0, 0, 2 }, // Argument("benchmark", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 465, 0, 0, 2 }, // Argument("deterministic", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 281, 0, 0, 1 }, // Argument("output_mask", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 472, 0, 0, 0 }, // Argument("result2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 462, 0, 0, 2 }, // Argument("groups", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 464, 0, 0, 2 }, // Argument("benchmark", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 465, 0, 0, 2 }, // Argument("deterministic", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 506, 0, 0, 1 }, // Argument("weight_size", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 462, 0, 0, 2 }, // Argument("groups", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 464, 0, 0, 2 }, // Argument("benchmark", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 465, 0, 0, 2 }, // Argument("deterministic", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 491, 0, 0, 0 }, // Argument("grid", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 491, 0, 0, 0 }, // Argument("grid", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 514, 0, 0, 0 }, // Argument("grad_self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 515, 0, 0, 0 }, // Argument("grad_grid", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 520, 0, 8, 2 }, // Argument("offset", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 520, 0, 8, 2 }, // Argument("offset", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 521, 0, 8, 2 }, // Argument("dim1", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 522, 0, 22, 2 }, // Argument("dim2", as_tensor(int64_t(1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 6, 0, 0, 0 }, // Argument("tensor", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 525, 0, 2, 2 }, // Argument("padding_idx", as_tensor(int64_t(-1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 526, 0, 1, 2 }, // Argument("scale_grad_by_freq", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 527, 0, 1, 2 }, // Argument("sparse", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 493, 0, 0, 0 }, // Argument("grad", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 529, 0, 0, 2 }, // Argument("num_weights", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 525, 0, 0, 2 }, // Argument("padding_idx", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 526, 0, 0, 2 }, // Argument("scale_grad_by_freq", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 527, 0, 0, 2 }, // Argument("sparse", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 493, 0, 0, 0 }, // Argument("grad", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 529, 0, 0, 2 }, // Argument("num_weights", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 525, 0, 0, 2 }, // Argument("padding_idx", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 526, 0, 0, 2 }, // Argument("scale_grad_by_freq", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 493, 0, 0, 0 }, // Argument("grad", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 529, 0, 0, 2 }, // Argument("num_weights", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 525, 0, 0, 2 }, // Argument("padding_idx", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 526, 0, 0, 2 }, // Argument("scale_grad_by_freq", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 533, 0, 0, 0 }, // Argument("offsets", at::nullopt, at::nullopt, DynamicType::get()) 
    { 526, 0, 1, 2 }, // Argument("scale_grad_by_freq", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 50, 0, 8, 2 }, // Argument("mode", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 527, 0, 1, 2 }, // Argument("sparse", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 472, 0, 0, 0 }, // Argument("result2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 534, 0, 0, 0 }, // Argument("result3", at::nullopt, at::nullopt, DynamicType::get()) 
    { 493, 0, 0, 0 }, // Argument("grad", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 533, 0, 0, 0 }, // Argument("offsets", at::nullopt, at::nullopt, DynamicType::get()) 
    { 536, 0, 0, 0 }, // Argument("offset2bag", at::nullopt, at::nullopt, DynamicType::get()) 
    { 537, 0, 0, 0 }, // Argument("bag_size", at::nullopt, at::nullopt, DynamicType::get()) 
    { 538, 0, 0, 0 }, // Argument("maximum_indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 529, 0, 0, 2 }, // Argument("num_weights", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 526, 0, 0, 2 }, // Argument("scale_grad_by_freq", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 50, 0, 0, 2 }, // Argument("mode", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 527, 0, 0, 2 }, // Argument("sparse", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 493, 0, 0, 0 }, // Argument("grad", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 533, 0, 0, 0 }, // Argument("offsets", at::nullopt, at::nullopt, DynamicType::get()) 
    { 536, 0, 0, 0 }, // Argument("offset2bag", at::nullopt, at::nullopt, DynamicType::get()) 
    { 537, 0, 0, 0 }, // Argument("bag_size", at::nullopt, at::nullopt, DynamicType::get()) 
    { 529, 0, 0, 2 }, // Argument("num_weights", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 526, 0, 0, 2 }, // Argument("scale_grad_by_freq", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 50, 0, 0, 2 }, // Argument("mode", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 493, 0, 0, 0 }, // Argument("grad", at::nullopt, at::nullopt, DynamicType::get()) 
    { 49, 0, 0, 0 }, // Argument("indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 533, 0, 0, 0 }, // Argument("offsets", at::nullopt, at::nullopt, DynamicType::get()) 
    { 536, 0, 0, 0 }, // Argument("offset2bag", at::nullopt, at::nullopt, DynamicType::get()) 
    { 537, 0, 0, 0 }, // Argument("bag_size", at::nullopt, at::nullopt, DynamicType::get()) 
    { 538, 0, 0, 0 }, // Argument("maximum_indices", at::nullopt, at::nullopt, DynamicType::get()) 
    { 529, 0, 0, 2 }, // Argument("num_weights", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 526, 0, 0, 2 }, // Argument("scale_grad_by_freq", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 50, 0, 0, 2 }, // Argument("mode", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 13, 0, 0, 1 }, // Argument("size", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 546, 0, 1, 2 }, // Argument("implicit", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 550, 0, 0, 3 }, // Argument("fill_value", at::nullopt, AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 233, 0, 24, 10 }, // Argument("margin", as_tensor(double(1.0)), AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 217, 0, 3, 2 }, // Argument("size_average", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 3, 2 }, // Argument("reduce", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 143, 0, 0, 0 }, // Argument("vec2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 158, 0, 0, 0 }, // Argument("A", at::nullopt, at::nullopt, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 158, 0, 0, 0 }, // Argument("A", at::nullopt, at::nullopt, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 556, 0, 0, 2 }, // Argument("num_groups", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 107, 0, 10, 0 }, // Argument("weight", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 383, 0, 20, 10 }, // Argument("eps", as_tensor(double(1e-05)), AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 451, 0, 3, 2 }, // Argument("cudnn_enabled", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 558, 0, 0, 2 }, // Argument("signal_ndim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 559, 0, 1, 2 }, // Argument("normalized", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 558, 0, 0, 2 }, // Argument("signal_ndim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 559, 0, 1, 2 }, // Argument("normalized", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 558, 0, 0, 2 }, // Argument("signal_ndim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 559, 0, 1, 2 }, // Argument("normalized", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 562, 0, 3, 2 }, // Argument("onesided", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 558, 0, 0, 2 }, // Argument("signal_ndim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 559, 0, 1, 2 }, // Argument("normalized", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 562, 0, 3, 2 }, // Argument("onesided", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 564, 0, 16, 1 }, // Argument("signal_sizes", as_tensor(IntList({})), AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 558, 0, 0, 2 }, // Argument("signal_ndim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 566, 0, 0, 2 }, // Argument("complex_input", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 567, 0, 0, 2 }, // Argument("complex_output", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 174, 0, 0, 2 }, // Argument("inverse", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 568, 0, 0, 1 }, // Argument("checked_signal_sizes", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 559, 0, 0, 2 }, // Argument("normalized", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 562, 0, 0, 2 }, // Argument("onesided", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 569, 0, 0, 1 }, // Argument("output_sizes", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 438, 0, 20, 10 }, // Argument("rtol", as_tensor(double(1e-05)), AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 439, 0, 21, 10 }, // Argument("atol", as_tensor(double(1e-08)), AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 440, 0, 1, 2 }, // Argument("equal_nan", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 579, 0, 0, 1 }, // Argument("normalized_shape", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 107, 0, 10, 0 }, // Argument("weight", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 379, 0, 10, 0 }, // Argument("bias", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 383, 0, 20, 10 }, // Argument("eps", as_tensor(double(1e-05)), AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 580, 0, 3, 2 }, // Argument("cudnn_enable", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 454, 0, 0, 0 }, // Argument("input1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 178, 0, 0, 0 }, // Argument("input2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 216, 0, 0, 0 }, // Argument("target", at::nullopt, at::nullopt, DynamicType::get()) 
    { 233, 0, 23, 10 }, // Argument("margin", as_tensor(double(0.0)), AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 217, 0, 3, 2 }, // Argument("size_average", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 3, 2 }, // Argument("reduce", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 310, 0, 0, 8 }, // Argument("kernel_size", at::nullopt, AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 203, 0, 16, 8 }, // Argument("stride", as_tensor(IntList({})), AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 311, 0, 17, 8 }, // Argument("padding", as_tensor(IntList(0)), AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 324, 0, 18, 8 }, // Argument("dilation", as_tensor(IntList(1)), AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 312, 0, 1, 2 }, // Argument("ceil_mode", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 379, 0, 0, 0 }, // Argument("bias", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 502, 0, 0, 1 }, // Argument("self_size", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 596, 0, 0, 2 }, // Argument("bias_defined", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 506, 0, 0, 1 }, // Argument("weight_size", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 596, 0, 0, 2 }, // Argument("bias_defined", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 107, 0, 0, 0 }, // Argument("weight", at::nullopt, at::nullopt, DynamicType::get()) 
    { 311, 0, 0, 1 }, // Argument("padding", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 203, 0, 0, 1 }, // Argument("stride", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 324, 0, 0, 1 }, // Argument("dilation", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 281, 0, 0, 1 }, // Argument("output_mask", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 472, 0, 0, 0 }, // Argument("result2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 136, 0, 0, 0 }, // Argument("mat2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 140, 0, 0, 0 }, // Argument("vec", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 602, 0, 0, 2 }, // Argument("start", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 603, 0, 0, 2 }, // Argument("length", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 606, 0, 0, 0 }, // Argument("x1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 607, 0, 0, 0 }, // Argument("x2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 95, 0, 25, 10 }, // Argument("p", as_tensor(double(2)), AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 383, 0, 26, 10 }, // Argument("eps", as_tensor(double(1e-06)), AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 609, 0, 0, 1 }, // Argument("dims", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 613, 0, 0, 2 }, // Argument("high", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 614, 0, 0, 2 }, // Argument("low", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 613, 0, 0, 2 }, // Argument("high", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 617, 0, 0, 1 }, // Argument("repeats", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 619, 0, 0, 1 }, // Argument("shape", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 621, 0, 0, 0 }, // Argument("rois", at::nullopt, at::nullopt, DynamicType::get()) 
    { 622, 0, 0, 2 }, // Argument("pooledHeight", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 623, 0, 0, 2 }, // Argument("pooledWidth", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 624, 0, 0, 10 }, // Argument("spatialScale", at::nullopt, AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 450, 0, 0, 0 }, // Argument("input", at::nullopt, at::nullopt, DynamicType::get()) 
    { 621, 0, 0, 0 }, // Argument("rois", at::nullopt, at::nullopt, DynamicType::get()) 
    { 622, 0, 0, 2 }, // Argument("pooledHeight", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 623, 0, 0, 2 }, // Argument("pooledWidth", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 624, 0, 0, 10 }, // Argument("spatialScale", at::nullopt, AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 626, 0, 0, 0 }, // Argument("gradOutput", at::nullopt, at::nullopt, DynamicType::get()) 
    { 627, 0, 0, 0 }, // Argument("argmaxes", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 16, 0, 0, 2 }, // Argument("index", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 8, 2 }, // Argument("dim", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 602, 0, 8, 2 }, // Argument("start", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 106, 0, 27, 2 }, // Argument("end", as_tensor(int64_t(9223372036854775807)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 20, 0, 22, 2 }, // Argument("step", as_tensor(int64_t(1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 136, 0, 0, 0 }, // Argument("mat2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 221, 0, 0, 0 }, // Argument("grad_output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 641, 0, 0, 2 }, // Argument("split_size", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 15, 0, 8, 2 }, // Argument("dim", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 643, 0, 0, 1 }, // Argument("split_sizes", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 15, 0, 8, 2 }, // Argument("dim", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 135, 0, 0, 0 }, // Argument("mat1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 136, 0, 0, 0 }, // Argument("mat2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 137, 0, 7, 3 }, // Argument("beta", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 119, 0, 7, 3 }, // Argument("alpha", as_tensor(Scalar(1)), AttributeInfo{ AttributeKind::t, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 205, 1, 0, 0 }, // Argument("tensors", at::nullopt, at::nullopt, ListType::ofTensors()) 
    { 15, 0, 8, 2 }, // Argument("dim", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 648, 0, 0, 2 }, // Argument("frame_length", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 649, 0, 0, 2 }, // Argument("hop", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 650, 0, 0, 2 }, // Argument("fft_size", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 559, 0, 1, 2 }, // Argument("normalized", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 562, 0, 3, 2 }, // Argument("onesided", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 651, 0, 10, 0 }, // Argument("window", at::Tensor(), at::nullopt, DynamicType::get()) 
    { 652, 0, 8, 2 }, // Argument("pad_end", as_tensor(int64_t(0)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 8 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 8 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::is, 1 }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 42, 0, 1, 2 }, // Argument("keepdim", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 661, 0, 0, 2 }, // Argument("dim0", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 521, 0, 0, 2 }, // Argument("dim1", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 663, 0, 0, 0 }, // Argument("i1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 664, 0, 0, 0 }, // Argument("i2", at::nullopt, at::nullopt, DynamicType::get()) 
    { 665, 0, 0, 0 }, // Argument("i3", at::nullopt, at::nullopt, DynamicType::get()) 
    { 666, 0, 0, 1 }, // Argument("expand1", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 667, 0, 0, 1 }, // Argument("expand2", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 668, 0, 0, 1 }, // Argument("expand3", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 669, 0, 0, 1 }, // Argument("sumdim", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 670, 0, 22, 2 }, // Argument("unroll_dim", as_tensor(int64_t(1)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 672, 0, 0, 0 }, // Argument("anchor", at::nullopt, at::nullopt, DynamicType::get()) 
    { 673, 0, 0, 0 }, // Argument("positive", at::nullopt, at::nullopt, DynamicType::get()) 
    { 674, 0, 0, 0 }, // Argument("negative", at::nullopt, at::nullopt, DynamicType::get()) 
    { 233, 0, 24, 10 }, // Argument("margin", as_tensor(double(1.0)), AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 95, 0, 25, 10 }, // Argument("p", as_tensor(double(2)), AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 383, 0, 26, 10 }, // Argument("eps", as_tensor(double(1e-06)), AttributeInfo{ AttributeKind::f, at::nullopt }, DynamicType::get()) 
    { 675, 0, 1, 2 }, // Argument("swap", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 217, 0, 3, 2 }, // Argument("size_average", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 218, 0, 3, 2 }, // Argument("reduce", as_tensor(bool(true)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 56, 0, 1, 2 }, // Argument("sorted", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 679, 0, 1, 2 }, // Argument("return_inverse", as_tensor(bool(false)), AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 435, 0, 0, 0 }, // Argument("result0", at::nullopt, at::nullopt, DynamicType::get()) 
    { 436, 0, 0, 0 }, // Argument("result1", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 13, 0, 0, 1 }, // Argument("size", at::nullopt, AttributeInfo{ AttributeKind::is, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 15, 0, 0, 2 }, // Argument("dim", at::nullopt, AttributeInfo{ AttributeKind::i, at::nullopt }, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 684, 0, 0, 0 }, // Argument("condition", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 684, 0, 0, 0 }, // Argument("condition", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 24, 0, 0, 0 }, // Argument("other", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 175, 0, 0, 0 }, // Argument("output", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get()) 
    { 1, 0, 0, 0 }, // Argument("self", at::nullopt, at::nullopt, DynamicType::get()) 
    { 2, 0, 0, 0 }, // Argument("result", at::nullopt, at::nullopt, DynamicType::get())
  };

  // FunctionSchema(string name, vector<Argument> args, vector<Argument> returns)
  // the integer for args and returns is the _number_ of argument objects
  // which are read sequentially off of the arguments array above
  using OperatorCtor = uint32_t[3];
  OperatorCtor operators[] = {
    { 0, 1, 1 }, // FunctionSchema("storage_offset", <1 arguments>, <1 returns>) 
    { 3, 1, 1 }, // FunctionSchema("numel", <1 arguments>, <1 returns>) 
    { 4, 1, 1 }, // FunctionSchema("is_contiguous", <1 arguments>, <1 returns>) 
    { 5, 2, 1 }, // FunctionSchema("is_set_to", <2 arguments>, <1 returns>) 
    { 7, 2, 1 }, // FunctionSchema("masked_select", <2 arguments>, <1 returns>) 
    { 9, 1, 1 }, // FunctionSchema("nonzero", <1 arguments>, <1 returns>) 
    { 10, 1, 1 }, // FunctionSchema("contiguous", <1 arguments>, <1 returns>) 
    { 11, 1, 1 }, // FunctionSchema("clone", <1 arguments>, <1 returns>) 
    { 12, 2, 1 }, // FunctionSchema("view", <2 arguments>, <1 returns>) 
    { 14, 3, 1 }, // FunctionSchema("index_select", <3 arguments>, <1 returns>) 
    { 17, 2, 1 }, // FunctionSchema("take", <2 arguments>, <1 returns>) 
    { 18, 4, 1 }, // FunctionSchema("unfold", <4 arguments>, <1 returns>) 
    { 21, 3, 1 }, // FunctionSchema("gather", <3 arguments>, <1 returns>) 
    { 22, 1, 1 }, // FunctionSchema("data_ptr", <1 arguments>, <1 returns>) 
    { 23, 2, 1 }, // FunctionSchema("equal", <2 arguments>, <1 returns>) 
    { 25, 2, 1 }, // FunctionSchema("__and__", <2 arguments>, <1 returns>) 
    { 25, 2, 1 }, // FunctionSchema("__and__", <2 arguments>, <1 returns>) 
    { 26, 2, 1 }, // FunctionSchema("__iand__", <2 arguments>, <1 returns>) 
    { 26, 2, 1 }, // FunctionSchema("__iand__", <2 arguments>, <1 returns>) 
    { 27, 2, 1 }, // FunctionSchema("__or__", <2 arguments>, <1 returns>) 
    { 27, 2, 1 }, // FunctionSchema("__or__", <2 arguments>, <1 returns>) 
    { 28, 2, 1 }, // FunctionSchema("__ior__", <2 arguments>, <1 returns>) 
    { 28, 2, 1 }, // FunctionSchema("__ior__", <2 arguments>, <1 returns>) 
    { 29, 2, 1 }, // FunctionSchema("__xor__", <2 arguments>, <1 returns>) 
    { 29, 2, 1 }, // FunctionSchema("__xor__", <2 arguments>, <1 returns>) 
    { 30, 2, 1 }, // FunctionSchema("__ixor__", <2 arguments>, <1 returns>) 
    { 30, 2, 1 }, // FunctionSchema("__ixor__", <2 arguments>, <1 returns>) 
    { 31, 2, 1 }, // FunctionSchema("__lshift__", <2 arguments>, <1 returns>) 
    { 31, 2, 1 }, // FunctionSchema("__lshift__", <2 arguments>, <1 returns>) 
    { 32, 2, 1 }, // FunctionSchema("__ilshift__", <2 arguments>, <1 returns>) 
    { 32, 2, 1 }, // FunctionSchema("__ilshift__", <2 arguments>, <1 returns>) 
    { 33, 2, 1 }, // FunctionSchema("__rshift__", <2 arguments>, <1 returns>) 
    { 33, 2, 1 }, // FunctionSchema("__rshift__", <2 arguments>, <1 returns>) 
    { 34, 2, 1 }, // FunctionSchema("__irshift__", <2 arguments>, <1 returns>) 
    { 34, 2, 1 }, // FunctionSchema("__irshift__", <2 arguments>, <1 returns>) 
    { 35, 2, 1 }, // FunctionSchema("lt", <2 arguments>, <1 returns>) 
    { 35, 2, 1 }, // FunctionSchema("lt", <2 arguments>, <1 returns>) 
    { 36, 2, 1 }, // FunctionSchema("gt", <2 arguments>, <1 returns>) 
    { 36, 2, 1 }, // FunctionSchema("gt", <2 arguments>, <1 returns>) 
    { 37, 2, 1 }, // FunctionSchema("le", <2 arguments>, <1 returns>) 
    { 37, 2, 1 }, // FunctionSchema("le", <2 arguments>, <1 returns>) 
    { 38, 2, 1 }, // FunctionSchema("ge", <2 arguments>, <1 returns>) 
    { 38, 2, 1 }, // FunctionSchema("ge", <2 arguments>, <1 returns>) 
    { 39, 2, 1 }, // FunctionSchema("eq", <2 arguments>, <1 returns>) 
    { 39, 2, 1 }, // FunctionSchema("eq", <2 arguments>, <1 returns>) 
    { 40, 2, 1 }, // FunctionSchema("ne", <2 arguments>, <1 returns>) 
    { 40, 2, 1 }, // FunctionSchema("ne", <2 arguments>, <1 returns>) 
    { 41, 3, 2 }, // FunctionSchema("min", <3 arguments>, <2 returns>) 
    { 41, 2, 1 }, // FunctionSchema("min", <2 arguments>, <1 returns>) 
    { 41, 1, 1 }, // FunctionSchema("min", <1 arguments>, <1 returns>) 
    { 44, 3, 2 }, // FunctionSchema("max", <3 arguments>, <2 returns>) 
    { 44, 2, 1 }, // FunctionSchema("max", <2 arguments>, <1 returns>) 
    { 44, 1, 1 }, // FunctionSchema("max", <1 arguments>, <1 returns>) 
    { 46, 4, 2 }, // FunctionSchema("kthvalue", <4 arguments>, <2 returns>) 
    { 50, 3, 2 }, // FunctionSchema("mode", <3 arguments>, <2 returns>) 
    { 51, 3, 2 }, // FunctionSchema("median", <3 arguments>, <2 returns>) 
    { 51, 1, 1 }, // FunctionSchema("median", <1 arguments>, <1 returns>) 
    { 52, 3, 2 }, // FunctionSchema("sort", <3 arguments>, <2 returns>) 
    { 54, 5, 2 }, // FunctionSchema("topk", <5 arguments>, <2 returns>) 
    { 57, 3, 1 }, // FunctionSchema("all", <3 arguments>, <1 returns>) 
    { 57, 1, 1 }, // FunctionSchema("all", <1 arguments>, <1 returns>) 
    { 58, 3, 1 }, // FunctionSchema("any", <3 arguments>, <1 returns>) 
    { 58, 1, 1 }, // FunctionSchema("any", <1 arguments>, <1 returns>) 
    { 59, 1, 1 }, // FunctionSchema("get_device", <1 arguments>, <1 returns>) 
    { 60, 1, 1 }, // FunctionSchema("_abs", <1 arguments>, <1 returns>) 
    { 61, 1, 1 }, // FunctionSchema("sigmoid", <1 arguments>, <1 returns>) 
    { 62, 1, 1 }, // FunctionSchema("_log", <1 arguments>, <1 returns>) 
    { 63, 1, 1 }, // FunctionSchema("_log10", <1 arguments>, <1 returns>) 
    { 64, 1, 1 }, // FunctionSchema("_log1p", <1 arguments>, <1 returns>) 
    { 65, 1, 1 }, // FunctionSchema("_log2", <1 arguments>, <1 returns>) 
    { 66, 1, 1 }, // FunctionSchema("lgamma", <1 arguments>, <1 returns>) 
    { 67, 1, 1 }, // FunctionSchema("digamma", <1 arguments>, <1 returns>) 
    { 68, 2, 1 }, // FunctionSchema("polygamma", <2 arguments>, <1 returns>) 
    { 70, 1, 1 }, // FunctionSchema("_exp", <1 arguments>, <1 returns>) 
    { 71, 1, 1 }, // FunctionSchema("_expm1", <1 arguments>, <1 returns>) 
    { 72, 1, 1 }, // FunctionSchema("_cos", <1 arguments>, <1 returns>) 
    { 73, 1, 1 }, // FunctionSchema("_acos", <1 arguments>, <1 returns>) 
    { 74, 1, 1 }, // FunctionSchema("_cosh", <1 arguments>, <1 returns>) 
    { 75, 1, 1 }, // FunctionSchema("_sin", <1 arguments>, <1 returns>) 
    { 76, 1, 1 }, // FunctionSchema("_asin", <1 arguments>, <1 returns>) 
    { 77, 1, 1 }, // FunctionSchema("_sinh", <1 arguments>, <1 returns>) 
    { 78, 1, 1 }, // FunctionSchema("_tan", <1 arguments>, <1 returns>) 
    { 79, 1, 1 }, // FunctionSchema("_atan", <1 arguments>, <1 returns>) 
    { 80, 1, 1 }, // FunctionSchema("_th_tanh", <1 arguments>, <1 returns>) 
    { 81, 1, 1 }, // FunctionSchema("_erf", <1 arguments>, <1 returns>) 
    { 82, 1, 1 }, // FunctionSchema("erfinv", <1 arguments>, <1 returns>) 
    { 83, 1, 1 }, // FunctionSchema("_sqrt", <1 arguments>, <1 returns>) 
    { 84, 1, 1 }, // FunctionSchema("_rsqrt", <1 arguments>, <1 returns>) 
    { 85, 1, 1 }, // FunctionSchema("_ceil", <1 arguments>, <1 returns>) 
    { 86, 1, 1 }, // FunctionSchema("_floor", <1 arguments>, <1 returns>) 
    { 87, 1, 1 }, // FunctionSchema("_round", <1 arguments>, <1 returns>) 
    { 88, 1, 1 }, // FunctionSchema("_trunc", <1 arguments>, <1 returns>) 
    { 89, 1, 1 }, // FunctionSchema("frac", <1 arguments>, <1 returns>) 
    { 90, 3, 1 }, // FunctionSchema("mean", <3 arguments>, <1 returns>) 
    { 90, 1, 1 }, // FunctionSchema("mean", <1 arguments>, <1 returns>) 
    { 91, 4, 1 }, // FunctionSchema("var", <4 arguments>, <1 returns>) 
    { 91, 2, 1 }, // FunctionSchema("var", <2 arguments>, <1 returns>) 
    { 93, 4, 1 }, // FunctionSchema("std", <4 arguments>, <1 returns>) 
    { 93, 2, 1 }, // FunctionSchema("std", <2 arguments>, <1 returns>) 
    { 94, 4, 1 }, // FunctionSchema("norm", <4 arguments>, <1 returns>) 
    { 94, 2, 1 }, // FunctionSchema("norm", <2 arguments>, <1 returns>) 
    { 96, 4, 1 }, // FunctionSchema("renorm", <4 arguments>, <1 returns>) 
    { 98, 3, 1 }, // FunctionSchema("dist", <3 arguments>, <1 returns>) 
    { 99, 1, 1 }, // FunctionSchema("reciprocal", <1 arguments>, <1 returns>) 
    { 100, 1, 1 }, // FunctionSchema("neg", <1 arguments>, <1 returns>) 
    { 101, 2, 1 }, // FunctionSchema("atan2", <2 arguments>, <1 returns>) 
    { 102, 2, 1 }, // FunctionSchema("pow", <2 arguments>, <1 returns>) 
    { 102, 2, 1 }, // FunctionSchema("pow", <2 arguments>, <1 returns>) 
    { 102, 2, 1 }, // FunctionSchema("pow", <2 arguments>, <1 returns>) 
    { 105, 3, 1 }, // FunctionSchema("lerp", <3 arguments>, <1 returns>) 
    { 108, 4, 1 }, // FunctionSchema("histc", <4 arguments>, <1 returns>) 
    { 110, 1, 1 }, // FunctionSchema("_sumall", <1 arguments>, <1 returns>) 
    { 111, 3, 1 }, // FunctionSchema("_th_sum", <3 arguments>, <1 returns>) 
    { 112, 1, 1 }, // FunctionSchema("_prodall", <1 arguments>, <1 returns>) 
    { 113, 3, 1 }, // FunctionSchema("_th_prod", <3 arguments>, <1 returns>) 
    { 114, 2, 1 }, // FunctionSchema("_cumsum", <2 arguments>, <1 returns>) 
    { 115, 2, 1 }, // FunctionSchema("_cumprod", <2 arguments>, <1 returns>) 
    { 116, 1, 1 }, // FunctionSchema("sign", <1 arguments>, <1 returns>) 
    { 117, 1, 1 }, // FunctionSchema("trace", <1 arguments>, <1 returns>) 
    { 118, 3, 1 }, // FunctionSchema("add", <3 arguments>, <1 returns>) 
    { 118, 3, 1 }, // FunctionSchema("add", <3 arguments>, <1 returns>) 
    { 120, 3, 1 }, // FunctionSchema("sub", <3 arguments>, <1 returns>) 
    { 120, 3, 1 }, // FunctionSchema("sub", <3 arguments>, <1 returns>) 
    { 121, 2, 1 }, // FunctionSchema("mul", <2 arguments>, <1 returns>) 
    { 121, 2, 1 }, // FunctionSchema("mul", <2 arguments>, <1 returns>) 
    { 122, 2, 1 }, // FunctionSchema("div", <2 arguments>, <1 returns>) 
    { 122, 2, 1 }, // FunctionSchema("div", <2 arguments>, <1 returns>) 
    { 123, 2, 1 }, // FunctionSchema("fmod", <2 arguments>, <1 returns>) 
    { 123, 2, 1 }, // FunctionSchema("fmod", <2 arguments>, <1 returns>) 
    { 124, 2, 1 }, // FunctionSchema("remainder", <2 arguments>, <1 returns>) 
    { 124, 2, 1 }, // FunctionSchema("remainder", <2 arguments>, <1 returns>) 
    { 125, 3, 1 }, // FunctionSchema("clamp", <3 arguments>, <1 returns>) 
    { 126, 2, 1 }, // FunctionSchema("clamp_min", <2 arguments>, <1 returns>) 
    { 127, 2, 1 }, // FunctionSchema("clamp_max", <2 arguments>, <1 returns>) 
    { 128, 2, 1 }, // FunctionSchema("_dot", <2 arguments>, <1 returns>) 
    { 129, 2, 1 }, // FunctionSchema("tril", <2 arguments>, <1 returns>) 
    { 131, 2, 1 }, // FunctionSchema("triu", <2 arguments>, <1 returns>) 
    { 132, 3, 1 }, // FunctionSchema("cross", <3 arguments>, <1 returns>) 
    { 133, 2, 1 }, // FunctionSchema("diag", <2 arguments>, <1 returns>) 
    { 134, 5, 1 }, // FunctionSchema("addmm", <5 arguments>, <1 returns>) 
    { 138, 5, 1 }, // FunctionSchema("_addmv", <5 arguments>, <1 returns>) 
    { 141, 5, 1 }, // FunctionSchema("_addr", <5 arguments>, <1 returns>) 
    { 144, 2, 1 }, // FunctionSchema("_ger", <2 arguments>, <1 returns>) 
    { 145, 2, 1 }, // FunctionSchema("_mv", <2 arguments>, <1 returns>) 
    { 146, 2, 1 }, // FunctionSchema("_mm", <2 arguments>, <1 returns>) 
    { 147, 2, 1 }, // FunctionSchema("bmm", <2 arguments>, <1 returns>) 
    { 148, 5, 1 }, // FunctionSchema("addbmm", <5 arguments>, <1 returns>) 
    { 151, 5, 1 }, // FunctionSchema("baddbmm", <5 arguments>, <1 returns>) 
    { 152, 4, 1 }, // FunctionSchema("addcmul", <4 arguments>, <1 returns>) 
    { 156, 4, 1 }, // FunctionSchema("addcdiv", <4 arguments>, <1 returns>) 
    { 157, 2, 2 }, // FunctionSchema("_gesv_single", <2 arguments>, <2 returns>) 
    { 161, 2, 2 }, // FunctionSchema("gels", <2 arguments>, <2 returns>) 
    { 164, 5, 2 }, // FunctionSchema("trtrs", <5 arguments>, <2 returns>) 
    { 168, 3, 2 }, // FunctionSchema("symeig", <3 arguments>, <2 returns>) 
    { 170, 2, 2 }, // FunctionSchema("eig", <2 arguments>, <2 returns>) 
    { 171, 2, 3 }, // FunctionSchema("svd", <2 arguments>, <3 returns>) 
    { 174, 1, 1 }, // FunctionSchema("inverse", <1 arguments>, <1 returns>) 
    { 176, 2, 1 }, // FunctionSchema("potrf", <2 arguments>, <1 returns>) 
    { 177, 3, 1 }, // FunctionSchema("potrs", <3 arguments>, <1 returns>) 
    { 179, 2, 1 }, // FunctionSchema("potri", <2 arguments>, <1 returns>) 
    { 180, 3, 2 }, // FunctionSchema("pstrf", <3 arguments>, <2 returns>) 
    { 182, 1, 2 }, // FunctionSchema("qr", <1 arguments>, <2 returns>) 
    { 183, 1, 2 }, // FunctionSchema("geqrf", <1 arguments>, <2 returns>) 
    { 184, 2, 1 }, // FunctionSchema("orgqr", <2 arguments>, <1 returns>) 
    { 185, 5, 1 }, // FunctionSchema("ormqr", <5 arguments>, <1 returns>) 
    { 188, 2, 2 }, // FunctionSchema("btrifact", <2 arguments>, <2 returns>) 
    { 191, 2, 3 }, // FunctionSchema("btrifact_with_info", <2 arguments>, <3 returns>) 
    { 193, 3, 1 }, // FunctionSchema("btrisolve", <3 arguments>, <1 returns>) 
    { 196, 3, 1 }, // FunctionSchema("_dirichlet_grad", <3 arguments>, <1 returns>) 
    { 199, 3, 1 }, // FunctionSchema("sparse_coo_tensor", <3 arguments>, <1 returns>) 
    { 199, 2, 1 }, // FunctionSchema("sparse_coo_tensor", <2 arguments>, <1 returns>) 
    { 200, 1, 1 }, // FunctionSchema("alias", <1 arguments>, <1 returns>) 
    { 201, 3, 1 }, // FunctionSchema("_sparse_coo_tensor_unsafe", <3 arguments>, <1 returns>) 
    { 202, 4, 1 }, // FunctionSchema("as_strided", <4 arguments>, <1 returns>) 
    { 204, 2, 1 }, // FunctionSchema("_cat", <2 arguments>, <1 returns>) 
    { 206, 1, 1 }, // FunctionSchema("to_dense", <1 arguments>, <1 returns>) 
    { 207, 1, 1 }, // FunctionSchema("_dimI", <1 arguments>, <1 returns>) 
    { 208, 1, 1 }, // FunctionSchema("_dimV", <1 arguments>, <1 returns>) 
    { 209, 1, 1 }, // FunctionSchema("_nnz", <1 arguments>, <1 returns>) 
    { 210, 1, 1 }, // FunctionSchema("coalesce", <1 arguments>, <1 returns>) 
    { 211, 1, 1 }, // FunctionSchema("is_coalesced", <1 arguments>, <1 returns>) 
    { 212, 1, 1 }, // FunctionSchema("_indices", <1 arguments>, <1 returns>) 
    { 213, 1, 1 }, // FunctionSchema("_values", <1 arguments>, <1 returns>) 
    { 214, 2, 1 }, // FunctionSchema("hspmm", <2 arguments>, <1 returns>) 
    { 215, 5, 1 }, // FunctionSchema("binary_cross_entropy", <5 arguments>, <1 returns>) 
    { 219, 5, 1 }, // FunctionSchema("binary_cross_entropy_forward", <5 arguments>, <1 returns>) 
    { 220, 6, 1 }, // FunctionSchema("binary_cross_entropy_backward", <6 arguments>, <1 returns>) 
    { 223, 4, 1 }, // FunctionSchema("kl_div", <4 arguments>, <1 returns>) 
    { 224, 4, 1 }, // FunctionSchema("kl_div_forward", <4 arguments>, <1 returns>) 
    { 225, 5, 1 }, // FunctionSchema("kl_div_backward", <5 arguments>, <1 returns>) 
    { 226, 4, 1 }, // FunctionSchema("l1_loss", <4 arguments>, <1 returns>) 
    { 227, 4, 1 }, // FunctionSchema("l1_loss_forward", <4 arguments>, <1 returns>) 
    { 228, 5, 1 }, // FunctionSchema("l1_loss_backward", <5 arguments>, <1 returns>) 
    { 229, 4, 1 }, // FunctionSchema("mse_loss", <4 arguments>, <1 returns>) 
    { 230, 4, 1 }, // FunctionSchema("mse_loss_forward", <4 arguments>, <1 returns>) 
    { 231, 5, 1 }, // FunctionSchema("mse_loss_backward", <5 arguments>, <1 returns>) 
    { 232, 7, 1 }, // FunctionSchema("multi_margin_loss", <7 arguments>, <1 returns>) 
    { 234, 7, 1 }, // FunctionSchema("multi_margin_loss_forward", <7 arguments>, <1 returns>) 
    { 235, 8, 1 }, // FunctionSchema("multi_margin_loss_backward", <8 arguments>, <1 returns>) 
    { 236, 4, 1 }, // FunctionSchema("multilabel_margin_loss", <4 arguments>, <1 returns>) 
    { 237, 4, 2 }, // FunctionSchema("multilabel_margin_loss_forward", <4 arguments>, <2 returns>) 
    { 239, 6, 1 }, // FunctionSchema("multilabel_margin_loss_backward", <6 arguments>, <1 returns>) 
    { 240, 6, 1 }, // FunctionSchema("nll_loss", <6 arguments>, <1 returns>) 
    { 242, 6, 2 }, // FunctionSchema("nll_loss_forward", <6 arguments>, <2 returns>) 
    { 244, 8, 1 }, // FunctionSchema("nll_loss_backward", <8 arguments>, <1 returns>) 
    { 245, 6, 1 }, // FunctionSchema("nll_loss2d", <6 arguments>, <1 returns>) 
    { 246, 6, 2 }, // FunctionSchema("nll_loss2d_forward", <6 arguments>, <2 returns>) 
    { 247, 8, 1 }, // FunctionSchema("nll_loss2d_backward", <8 arguments>, <1 returns>) 
    { 248, 4, 1 }, // FunctionSchema("smooth_l1_loss", <4 arguments>, <1 returns>) 
    { 249, 4, 1 }, // FunctionSchema("smooth_l1_loss_forward", <4 arguments>, <1 returns>) 
    { 250, 5, 1 }, // FunctionSchema("smooth_l1_loss_backward", <5 arguments>, <1 returns>) 
    { 251, 4, 1 }, // FunctionSchema("soft_margin_loss", <4 arguments>, <1 returns>) 
    { 252, 4, 1 }, // FunctionSchema("soft_margin_loss_forward", <4 arguments>, <1 returns>) 
    { 253, 5, 1 }, // FunctionSchema("soft_margin_loss_backward", <5 arguments>, <1 returns>) 
    { 254, 3, 1 }, // FunctionSchema("elu", <3 arguments>, <1 returns>) 
    { 256, 3, 1 }, // FunctionSchema("elu_forward", <3 arguments>, <1 returns>) 
    { 257, 4, 1 }, // FunctionSchema("elu_backward", <4 arguments>, <1 returns>) 
    { 258, 2, 1 }, // FunctionSchema("glu", <2 arguments>, <1 returns>) 
    { 259, 2, 1 }, // FunctionSchema("glu_forward", <2 arguments>, <1 returns>) 
    { 260, 3, 1 }, // FunctionSchema("glu_backward", <3 arguments>, <1 returns>) 
    { 261, 2, 1 }, // FunctionSchema("hardshrink", <2 arguments>, <1 returns>) 
    { 263, 2, 1 }, // FunctionSchema("hardshrink_forward", <2 arguments>, <1 returns>) 
    { 264, 3, 1 }, // FunctionSchema("hardshrink_backward", <3 arguments>, <1 returns>) 
    { 265, 3, 1 }, // FunctionSchema("hardtanh", <3 arguments>, <1 returns>) 
    { 268, 3, 1 }, // FunctionSchema("hardtanh_forward", <3 arguments>, <1 returns>) 
    { 269, 4, 1 }, // FunctionSchema("hardtanh_backward", <4 arguments>, <1 returns>) 
    { 270, 2, 1 }, // FunctionSchema("leaky_relu", <2 arguments>, <1 returns>) 
    { 272, 2, 1 }, // FunctionSchema("leaky_relu_forward", <2 arguments>, <1 returns>) 
    { 273, 3, 1 }, // FunctionSchema("leaky_relu_backward", <3 arguments>, <1 returns>) 
    { 274, 1, 1 }, // FunctionSchema("log_sigmoid", <1 arguments>, <1 returns>) 
    { 275, 1, 2 }, // FunctionSchema("log_sigmoid_forward", <1 arguments>, <2 returns>) 
    { 277, 3, 1 }, // FunctionSchema("log_sigmoid_backward", <3 arguments>, <1 returns>) 
    { 278, 2, 1 }, // FunctionSchema("prelu", <2 arguments>, <1 returns>) 
    { 279, 2, 1 }, // FunctionSchema("prelu_forward", <2 arguments>, <1 returns>) 
    { 280, 4, 2 }, // FunctionSchema("prelu_backward", <4 arguments>, <2 returns>) 
    { 283, 6, 1 }, // FunctionSchema("rrelu_with_noise_backward", <6 arguments>, <1 returns>) 
    { 287, 3, 1 }, // FunctionSchema("softplus", <3 arguments>, <1 returns>) 
    { 289, 3, 1 }, // FunctionSchema("softplus_forward", <3 arguments>, <1 returns>) 
    { 290, 5, 1 }, // FunctionSchema("softplus_backward", <5 arguments>, <1 returns>) 
    { 291, 2, 1 }, // FunctionSchema("softshrink", <2 arguments>, <1 returns>) 
    { 292, 2, 1 }, // FunctionSchema("softshrink_forward", <2 arguments>, <1 returns>) 
    { 293, 3, 1 }, // FunctionSchema("softshrink_backward", <3 arguments>, <1 returns>) 
    { 288, 3, 1 }, // FunctionSchema("threshold", <3 arguments>, <1 returns>) 
    { 294, 3, 1 }, // FunctionSchema("threshold_forward", <3 arguments>, <1 returns>) 
    { 295, 4, 1 }, // FunctionSchema("threshold_backward", <4 arguments>, <1 returns>) 
    { 296, 2, 1 }, // FunctionSchema("adaptive_avg_pool2d", <2 arguments>, <1 returns>) 
    { 298, 2, 1 }, // FunctionSchema("adaptive_avg_pool2d_forward", <2 arguments>, <1 returns>) 
    { 299, 2, 1 }, // FunctionSchema("adaptive_avg_pool2d_backward", <2 arguments>, <1 returns>) 
    { 300, 2, 1 }, // FunctionSchema("adaptive_avg_pool3d", <2 arguments>, <1 returns>) 
    { 301, 2, 1 }, // FunctionSchema("adaptive_avg_pool3d_forward", <2 arguments>, <1 returns>) 
    { 302, 2, 1 }, // FunctionSchema("adaptive_avg_pool3d_backward", <2 arguments>, <1 returns>) 
    { 303, 2, 2 }, // FunctionSchema("adaptive_max_pool2d", <2 arguments>, <2 returns>) 
    { 304, 2, 2 }, // FunctionSchema("adaptive_max_pool2d_forward", <2 arguments>, <2 returns>) 
    { 305, 3, 1 }, // FunctionSchema("adaptive_max_pool2d_backward", <3 arguments>, <1 returns>) 
    { 306, 2, 2 }, // FunctionSchema("adaptive_max_pool3d", <2 arguments>, <2 returns>) 
    { 307, 2, 2 }, // FunctionSchema("adaptive_max_pool3d_forward", <2 arguments>, <2 returns>) 
    { 308, 3, 1 }, // FunctionSchema("adaptive_max_pool3d_backward", <3 arguments>, <1 returns>) 
    { 309, 6, 1 }, // FunctionSchema("avg_pool2d", <6 arguments>, <1 returns>) 
    { 314, 6, 1 }, // FunctionSchema("avg_pool2d_forward", <6 arguments>, <1 returns>) 
    { 315, 7, 1 }, // FunctionSchema("avg_pool2d_backward", <7 arguments>, <1 returns>) 
    { 316, 6, 1 }, // FunctionSchema("avg_pool3d", <6 arguments>, <1 returns>) 
    { 317, 6, 1 }, // FunctionSchema("avg_pool3d_forward", <6 arguments>, <1 returns>) 
    { 318, 7, 1 }, // FunctionSchema("avg_pool3d_backward", <7 arguments>, <1 returns>) 
    { 319, 4, 2 }, // FunctionSchema("fractional_max_pool2d", <4 arguments>, <2 returns>) 
    { 321, 4, 2 }, // FunctionSchema("fractional_max_pool2d_forward", <4 arguments>, <2 returns>) 
    { 322, 5, 1 }, // FunctionSchema("fractional_max_pool2d_backward", <5 arguments>, <1 returns>) 
    { 323, 6, 2 }, // FunctionSchema("max_pool2d", <6 arguments>, <2 returns>) 
    { 325, 6, 2 }, // FunctionSchema("max_pool2d_forward", <6 arguments>, <2 returns>) 
    { 326, 8, 1 }, // FunctionSchema("max_pool2d_backward", <8 arguments>, <1 returns>) 
    { 327, 6, 2 }, // FunctionSchema("max_pool3d", <6 arguments>, <2 returns>) 
    { 328, 6, 2 }, // FunctionSchema("max_pool3d_forward", <6 arguments>, <2 returns>) 
    { 329, 8, 1 }, // FunctionSchema("max_pool3d_backward", <8 arguments>, <1 returns>) 
    { 330, 3, 1 }, // FunctionSchema("max_unpool2d", <3 arguments>, <1 returns>) 
    { 331, 3, 1 }, // FunctionSchema("max_unpool2d_forward", <3 arguments>, <1 returns>) 
    { 332, 4, 1 }, // FunctionSchema("max_unpool2d_backward", <4 arguments>, <1 returns>) 
    { 333, 5, 1 }, // FunctionSchema("max_unpool3d", <5 arguments>, <1 returns>) 
    { 334, 5, 1 }, // FunctionSchema("max_unpool3d_forward", <5 arguments>, <1 returns>) 
    { 335, 6, 1 }, // FunctionSchema("max_unpool3d_backward", <6 arguments>, <1 returns>) 
    { 336, 2, 1 }, // FunctionSchema("reflection_pad1d", <2 arguments>, <1 returns>) 
    { 337, 2, 1 }, // FunctionSchema("reflection_pad1d_forward", <2 arguments>, <1 returns>) 
    { 338, 3, 1 }, // FunctionSchema("reflection_pad1d_backward", <3 arguments>, <1 returns>) 
    { 339, 2, 1 }, // FunctionSchema("reflection_pad2d", <2 arguments>, <1 returns>) 
    { 340, 2, 1 }, // FunctionSchema("reflection_pad2d_forward", <2 arguments>, <1 returns>) 
    { 341, 3, 1 }, // FunctionSchema("reflection_pad2d_backward", <3 arguments>, <1 returns>) 
    { 342, 2, 1 }, // FunctionSchema("replication_pad1d", <2 arguments>, <1 returns>) 
    { 343, 2, 1 }, // FunctionSchema("replication_pad1d_forward", <2 arguments>, <1 returns>) 
    { 344, 3, 1 }, // FunctionSchema("replication_pad1d_backward", <3 arguments>, <1 returns>) 
    { 345, 2, 1 }, // FunctionSchema("replication_pad2d", <2 arguments>, <1 returns>) 
    { 346, 2, 1 }, // FunctionSchema("replication_pad2d_forward", <2 arguments>, <1 returns>) 
    { 347, 3, 1 }, // FunctionSchema("replication_pad2d_backward", <3 arguments>, <1 returns>) 
    { 348, 2, 1 }, // FunctionSchema("replication_pad3d", <2 arguments>, <1 returns>) 
    { 349, 2, 1 }, // FunctionSchema("replication_pad3d_forward", <2 arguments>, <1 returns>) 
    { 350, 3, 1 }, // FunctionSchema("replication_pad3d_backward", <3 arguments>, <1 returns>) 
    { 351, 3, 1 }, // FunctionSchema("upsample_linear1d", <3 arguments>, <1 returns>) 
    { 353, 3, 1 }, // FunctionSchema("upsample_linear1d_forward", <3 arguments>, <1 returns>) 
    { 354, 4, 1 }, // FunctionSchema("upsample_linear1d_backward", <4 arguments>, <1 returns>) 
    { 356, 3, 1 }, // FunctionSchema("upsample_bilinear2d", <3 arguments>, <1 returns>) 
    { 357, 3, 1 }, // FunctionSchema("upsample_bilinear2d_forward", <3 arguments>, <1 returns>) 
    { 358, 4, 1 }, // FunctionSchema("upsample_bilinear2d_backward", <4 arguments>, <1 returns>) 
    { 359, 3, 1 }, // FunctionSchema("upsample_trilinear3d", <3 arguments>, <1 returns>) 
    { 360, 3, 1 }, // FunctionSchema("upsample_trilinear3d_forward", <3 arguments>, <1 returns>) 
    { 361, 4, 1 }, // FunctionSchema("upsample_trilinear3d_backward", <4 arguments>, <1 returns>) 
    { 362, 2, 1 }, // FunctionSchema("upsample_nearest1d", <2 arguments>, <1 returns>) 
    { 364, 2, 1 }, // FunctionSchema("upsample_nearest1d_forward", <2 arguments>, <1 returns>) 
    { 365, 3, 1 }, // FunctionSchema("upsample_nearest1d_backward", <3 arguments>, <1 returns>) 
    { 366, 2, 1 }, // FunctionSchema("upsample_nearest2d", <2 arguments>, <1 returns>) 
    { 367, 2, 1 }, // FunctionSchema("upsample_nearest2d_forward", <2 arguments>, <1 returns>) 
    { 368, 3, 1 }, // FunctionSchema("upsample_nearest2d_backward", <3 arguments>, <1 returns>) 
    { 369, 2, 1 }, // FunctionSchema("upsample_nearest3d", <2 arguments>, <1 returns>) 
    { 370, 2, 1 }, // FunctionSchema("upsample_nearest3d_forward", <2 arguments>, <1 returns>) 
    { 371, 3, 1 }, // FunctionSchema("upsample_nearest3d_backward", <3 arguments>, <1 returns>) 
    { 372, 1, 1 }, // FunctionSchema("_sigmoid", <1 arguments>, <1 returns>) 
    { 373, 1, 1 }, // FunctionSchema("_sigmoid_forward", <1 arguments>, <1 returns>) 
    { 374, 2, 1 }, // FunctionSchema("_sigmoid_backward", <2 arguments>, <1 returns>) 
    { 375, 1, 1 }, // FunctionSchema("_tanh", <1 arguments>, <1 returns>) 
    { 376, 1, 1 }, // FunctionSchema("_tanh_forward", <1 arguments>, <1 returns>) 
    { 377, 2, 1 }, // FunctionSchema("_tanh_backward", <2 arguments>, <1 returns>) 
    { 378, 8, 1 }, // FunctionSchema("thnn_batch_norm", <8 arguments>, <1 returns>) 
    { 384, 8, 3 }, // FunctionSchema("thnn_batch_norm_forward", <8 arguments>, <3 returns>) 
    { 387, 10, 3 }, // FunctionSchema("thnn_batch_norm_backward", <10 arguments>, <3 returns>) 
    { 389, 8, 1 }, // FunctionSchema("thnn_conv_transpose2d", <8 arguments>, <1 returns>) 
    { 391, 8, 3 }, // FunctionSchema("thnn_conv_transpose2d_forward", <8 arguments>, <3 returns>) 
    { 394, 11, 3 }, // FunctionSchema("thnn_conv_transpose2d_backward", <11 arguments>, <3 returns>) 
    { 395, 8, 1 }, // FunctionSchema("thnn_conv_transpose3d", <8 arguments>, <1 returns>) 
    { 396, 8, 3 }, // FunctionSchema("thnn_conv_transpose3d_forward", <8 arguments>, <3 returns>) 
    { 399, 11, 3 }, // FunctionSchema("thnn_conv_transpose3d_backward", <11 arguments>, <3 returns>) 
    { 400, 6, 1 }, // FunctionSchema("thnn_conv2d", <6 arguments>, <1 returns>) 
    { 401, 6, 3 }, // FunctionSchema("thnn_conv2d_forward", <6 arguments>, <3 returns>) 
    { 402, 9, 3 }, // FunctionSchema("thnn_conv2d_backward", <9 arguments>, <3 returns>) 
    { 403, 7, 1 }, // FunctionSchema("thnn_conv_depthwise2d", <7 arguments>, <1 returns>) 
    { 404, 7, 1 }, // FunctionSchema("thnn_conv_depthwise2d_forward", <7 arguments>, <1 returns>) 
    { 405, 8, 2 }, // FunctionSchema("thnn_conv_depthwise2d_backward", <8 arguments>, <2 returns>) 
    { 406, 6, 1 }, // FunctionSchema("thnn_conv3d", <6 arguments>, <1 returns>) 
    { 407, 6, 3 }, // FunctionSchema("thnn_conv3d_forward", <6 arguments>, <3 returns>) 
    { 408, 9, 3 }, // FunctionSchema("thnn_conv3d_backward", <9 arguments>, <3 returns>) 
    { 409, 7, 1 }, // FunctionSchema("thnn_conv_dilated2d", <7 arguments>, <1 returns>) 
    { 410, 7, 3 }, // FunctionSchema("thnn_conv_dilated2d_forward", <7 arguments>, <3 returns>) 
    { 411, 10, 3 }, // FunctionSchema("thnn_conv_dilated2d_backward", <10 arguments>, <3 returns>) 
    { 412, 7, 1 }, // FunctionSchema("thnn_conv_dilated3d", <7 arguments>, <1 returns>) 
    { 413, 7, 3 }, // FunctionSchema("thnn_conv_dilated3d_forward", <7 arguments>, <3 returns>) 
    { 414, 10, 3 }, // FunctionSchema("thnn_conv_dilated3d_backward", <10 arguments>, <3 returns>) 
    { 415, 2, 1 }, // FunctionSchema("_cast_uint8_t", <2 arguments>, <1 returns>) 
    { 417, 2, 1 }, // FunctionSchema("_cast_int8_t", <2 arguments>, <1 returns>) 
    { 418, 2, 1 }, // FunctionSchema("_cast_double", <2 arguments>, <1 returns>) 
    { 419, 2, 1 }, // FunctionSchema("_cast_float", <2 arguments>, <1 returns>) 
    { 420, 2, 1 }, // FunctionSchema("_cast_int", <2 arguments>, <1 returns>) 
    { 421, 2, 1 }, // FunctionSchema("_cast_int64_t", <2 arguments>, <1 returns>) 
    { 422, 2, 1 }, // FunctionSchema("_cast_int16_t", <2 arguments>, <1 returns>) 
    { 423, 2, 1 }, // FunctionSchema("_cast_Half", <2 arguments>, <1 returns>) 
    { 424, 8, 1 }, // FunctionSchema("_cudnn_rnn_flatten_weight", <8 arguments>, <1 returns>) 
    { 431, 1, 1 }, // FunctionSchema("abs", <1 arguments>, <1 returns>) 
    { 432, 1, 1 }, // FunctionSchema("acos", <1 arguments>, <1 returns>) 
    { 433, 2, 1 }, // FunctionSchema("adaptive_avg_pool1d", <2 arguments>, <1 returns>) 
    { 434, 2, 2 }, // FunctionSchema("adaptive_max_pool1d", <2 arguments>, <2 returns>) 
    { 437, 5, 1 }, // FunctionSchema("allclose", <5 arguments>, <1 returns>) 
    { 441, 5, 1 }, // FunctionSchema("addmv", <5 arguments>, <1 returns>) 
    { 442, 5, 1 }, // FunctionSchema("addr", <5 arguments>, <1 returns>) 
    { 443, 3, 1 }, // FunctionSchema("argmax", <3 arguments>, <1 returns>) 
    { 443, 1, 1 }, // FunctionSchema("argmax", <1 arguments>, <1 returns>) 
    { 444, 3, 1 }, // FunctionSchema("_argmax", <3 arguments>, <1 returns>) 
    { 445, 3, 1 }, // FunctionSchema("argmin", <3 arguments>, <1 returns>) 
    { 445, 1, 1 }, // FunctionSchema("argmin", <1 arguments>, <1 returns>) 
    { 446, 3, 1 }, // FunctionSchema("_argmin", <3 arguments>, <1 returns>) 
    { 447, 1, 1 }, // FunctionSchema("asin", <1 arguments>, <1 returns>) 
    { 448, 1, 1 }, // FunctionSchema("atan", <1 arguments>, <1 returns>) 
    { 449, 9, 1 }, // FunctionSchema("batch_norm", <9 arguments>, <1 returns>) 
    { 452, 1, 1 }, // FunctionSchema("bernoulli", <1 arguments>, <1 returns>) 
    { 453, 4, 1 }, // FunctionSchema("bilinear", <4 arguments>, <1 returns>) 
    { 455, 2, 1 }, // FunctionSchema("cat", <2 arguments>, <1 returns>) 
    { 456, 1, 1 }, // FunctionSchema("ceil", <1 arguments>, <1 returns>) 
    { 457, 3, 1 }, // FunctionSchema("chunk", <3 arguments>, <1 returns>) 
    { 459, 1, 1 }, // FunctionSchema("cudnn_is_acceptable", <1 arguments>, <1 returns>) 
    { 460, 9, 1 }, // FunctionSchema("convolution", <9 arguments>, <1 returns>) 
    { 463, 12, 1 }, // FunctionSchema("_convolution", <12 arguments>, <1 returns>) 
    { 466, 8, 1 }, // FunctionSchema("_convolution_nogroup", <8 arguments>, <1 returns>) 
    { 467, 16, 3 }, // FunctionSchema("_convolution_double_backward", <16 arguments>, <3 returns>) 
    { 473, 7, 1 }, // FunctionSchema("conv1d", <7 arguments>, <1 returns>) 
    { 474, 7, 1 }, // FunctionSchema("conv2d", <7 arguments>, <1 returns>) 
    { 475, 7, 1 }, // FunctionSchema("conv3d", <7 arguments>, <1 returns>) 
    { 476, 4, 1 }, // FunctionSchema("conv_tbc", <4 arguments>, <1 returns>) 
    { 478, 5, 3 }, // FunctionSchema("conv_tbc_backward", <5 arguments>, <3 returns>) 
    { 479, 8, 1 }, // FunctionSchema("conv_transpose1d", <8 arguments>, <1 returns>) 
    { 480, 8, 1 }, // FunctionSchema("conv_transpose2d", <8 arguments>, <1 returns>) 
    { 481, 8, 1 }, // FunctionSchema("conv_transpose3d", <8 arguments>, <1 returns>) 
    { 482, 1, 1 }, // FunctionSchema("cos", <1 arguments>, <1 returns>) 
    { 483, 1, 1 }, // FunctionSchema("cosh", <1 arguments>, <1 returns>) 
    { 484, 6, 1 }, // FunctionSchema("cosine_embedding_loss", <6 arguments>, <1 returns>) 
    { 485, 5, 1 }, // FunctionSchema("cudnn_affine_grid_generator", <5 arguments>, <1 returns>) 
    { 492, 5, 1 }, // FunctionSchema("cudnn_affine_grid_generator_backward", <5 arguments>, <1 returns>) 
    { 495, 8, 3 }, // FunctionSchema("cudnn_batch_norm", <8 arguments>, <3 returns>) 
    { 498, 8, 3 }, // FunctionSchema("cudnn_batch_norm_backward", <8 arguments>, <3 returns>) 
    { 500, 9, 1 }, // FunctionSchema("cudnn_convolution", <9 arguments>, <1 returns>) 
    { 501, 9, 1 }, // FunctionSchema("cudnn_convolution_backward_input", <9 arguments>, <1 returns>) 
    { 503, 10, 3 }, // FunctionSchema("cudnn_convolution_backward", <10 arguments>, <3 returns>) 
    { 504, 1, 1 }, // FunctionSchema("cudnn_convolution_backward_bias", <1 arguments>, <1 returns>) 
    { 505, 9, 1 }, // FunctionSchema("cudnn_convolution_backward_weight", <9 arguments>, <1 returns>) 
    { 507, 10, 1 }, // FunctionSchema("cudnn_convolution_transpose", <10 arguments>, <1 returns>) 
    { 508, 11, 3 }, // FunctionSchema("cudnn_convolution_transpose_backward", <11 arguments>, <3 returns>) 
    { 509, 1, 1 }, // FunctionSchema("cudnn_convolution_transpose_backward_bias", <1 arguments>, <1 returns>) 
    { 510, 8, 1 }, // FunctionSchema("cudnn_convolution_transpose_backward_input", <8 arguments>, <1 returns>) 
    { 511, 9, 1 }, // FunctionSchema("cudnn_convolution_transpose_backward_weight", <9 arguments>, <1 returns>) 
    { 512, 2, 1 }, // FunctionSchema("cudnn_grid_sampler", <2 arguments>, <1 returns>) 
    { 513, 3, 2 }, // FunctionSchema("cudnn_grid_sampler_backward", <3 arguments>, <2 returns>) 
    { 516, 2, 1 }, // FunctionSchema("cumsum", <2 arguments>, <1 returns>) 
    { 517, 2, 1 }, // FunctionSchema("cumprod", <2 arguments>, <1 returns>) 
    { 518, 1, 1 }, // FunctionSchema("det", <1 arguments>, <1 returns>) 
    { 519, 2, 1 }, // FunctionSchema("diagflat", <2 arguments>, <1 returns>) 
    { 130, 4, 1 }, // FunctionSchema("diagonal", <4 arguments>, <1 returns>) 
    { 523, 2, 1 }, // FunctionSchema("dot", <2 arguments>, <1 returns>) 
    { 524, 5, 1 }, // FunctionSchema("embedding", <5 arguments>, <1 returns>) 
    { 528, 6, 1 }, // FunctionSchema("embedding_backward", <6 arguments>, <1 returns>) 
    { 530, 5, 1 }, // FunctionSchema("embedding_dense_backward", <5 arguments>, <1 returns>) 
    { 531, 5, 1 }, // FunctionSchema("embedding_sparse_backward", <5 arguments>, <1 returns>) 
    { 532, 6, 4 }, // FunctionSchema("embedding_bag", <6 arguments>, <4 returns>) 
    { 535, 10, 1 }, // FunctionSchema("embedding_bag_backward", <10 arguments>, <1 returns>) 
    { 539, 8, 1 }, // FunctionSchema("embedding_bag_sparse_backward", <8 arguments>, <1 returns>) 
    { 540, 9, 1 }, // FunctionSchema("embedding_bag_dense_backward", <9 arguments>, <1 returns>) 
    { 541, 1, 1 }, // FunctionSchema("empty_like", <1 arguments>, <1 returns>) 
    { 542, 1, 1 }, // FunctionSchema("erf", <1 arguments>, <1 returns>) 
    { 543, 1, 1 }, // FunctionSchema("exp", <1 arguments>, <1 returns>) 
    { 544, 1, 1 }, // FunctionSchema("expm1", <1 arguments>, <1 returns>) 
    { 545, 3, 1 }, // FunctionSchema("expand", <3 arguments>, <1 returns>) 
    { 547, 2, 1 }, // FunctionSchema("expand_as", <2 arguments>, <1 returns>) 
    { 548, 1, 1 }, // FunctionSchema("floor", <1 arguments>, <1 returns>) 
    { 549, 2, 1 }, // FunctionSchema("full_like", <2 arguments>, <1 returns>) 
    { 551, 5, 1 }, // FunctionSchema("hinge_embedding_loss", <5 arguments>, <1 returns>) 
    { 552, 2, 1 }, // FunctionSchema("ger", <2 arguments>, <1 returns>) 
    { 553, 2, 2 }, // FunctionSchema("gesv", <2 arguments>, <2 returns>) 
    { 554, 2, 2 }, // FunctionSchema("_gesv_helper", <2 arguments>, <2 returns>) 
    { 555, 6, 1 }, // FunctionSchema("group_norm", <6 arguments>, <1 returns>) 
    { 557, 3, 1 }, // FunctionSchema("fft", <3 arguments>, <1 returns>) 
    { 560, 3, 1 }, // FunctionSchema("ifft", <3 arguments>, <1 returns>) 
    { 561, 4, 1 }, // FunctionSchema("rfft", <4 arguments>, <1 returns>) 
    { 563, 5, 1 }, // FunctionSchema("irfft", <5 arguments>, <1 returns>) 
    { 565, 9, 1 }, // FunctionSchema("_fft_with_size", <9 arguments>, <1 returns>) 
    { 570, 5, 1 }, // FunctionSchema("isclose", <5 arguments>, <1 returns>) 
    { 571, 1, 1 }, // FunctionSchema("is_cuda", <1 arguments>, <1 returns>) 
    { 572, 1, 1 }, // FunctionSchema("is_distributed", <1 arguments>, <1 returns>) 
    { 573, 1, 1 }, // FunctionSchema("is_floating_point", <1 arguments>, <1 returns>) 
    { 574, 1, 1 }, // FunctionSchema("is_nonzero", <1 arguments>, <1 returns>) 
    { 575, 2, 1 }, // FunctionSchema("is_same_size", <2 arguments>, <1 returns>) 
    { 576, 1, 1 }, // FunctionSchema("is_signed", <1 arguments>, <1 returns>) 
    { 577, 1, 1 }, // FunctionSchema("is_sparse", <1 arguments>, <1 returns>) 
    { 578, 6, 1 }, // FunctionSchema("layer_norm", <6 arguments>, <1 returns>) 
    { 581, 1, 1 }, // FunctionSchema("log", <1 arguments>, <1 returns>) 
    { 582, 1, 1 }, // FunctionSchema("log10", <1 arguments>, <1 returns>) 
    { 583, 1, 1 }, // FunctionSchema("log1p", <1 arguments>, <1 returns>) 
    { 584, 1, 1 }, // FunctionSchema("log2", <1 arguments>, <1 returns>) 
    { 585, 1, 1 }, // FunctionSchema("logdet", <1 arguments>, <1 returns>) 
    { 586, 2, 1 }, // FunctionSchema("log_softmax", <2 arguments>, <1 returns>) 
    { 587, 4, 1 }, // FunctionSchema("log_softmax_backward_data", <4 arguments>, <1 returns>) 
    { 588, 3, 1 }, // FunctionSchema("logsumexp", <3 arguments>, <1 returns>) 
    { 589, 6, 1 }, // FunctionSchema("margin_ranking_loss", <6 arguments>, <1 returns>) 
    { 590, 2, 1 }, // FunctionSchema("matmul", <2 arguments>, <1 returns>) 
    { 591, 3, 1 }, // FunctionSchema("max_values", <3 arguments>, <1 returns>) 
    { 592, 6, 2 }, // FunctionSchema("max_pool1d", <6 arguments>, <2 returns>) 
    { 593, 3, 1 }, // FunctionSchema("min_values", <3 arguments>, <1 returns>) 
    { 594, 6, 1 }, // FunctionSchema("mkldnn_convolution", <6 arguments>, <1 returns>) 
    { 595, 7, 1 }, // FunctionSchema("mkldnn_convolution_backward_input", <7 arguments>, <1 returns>) 
    { 597, 7, 2 }, // FunctionSchema("mkldnn_convolution_backward_weights", <7 arguments>, <2 returns>) 
    { 598, 7, 3 }, // FunctionSchema("mkldnn_convolution_backward", <7 arguments>, <3 returns>) 
    { 599, 2, 1 }, // FunctionSchema("mm", <2 arguments>, <1 returns>) 
    { 600, 2, 1 }, // FunctionSchema("mv", <2 arguments>, <1 returns>) 
    { 601, 4, 1 }, // FunctionSchema("narrow", <4 arguments>, <1 returns>) 
    { 604, 1, 1 }, // FunctionSchema("ones_like", <1 arguments>, <1 returns>) 
    { 605, 5, 1 }, // FunctionSchema("pairwise_distance", <5 arguments>, <1 returns>) 
    { 608, 2, 1 }, // FunctionSchema("permute", <2 arguments>, <1 returns>) 
    { 610, 1, 1 }, // FunctionSchema("pin_memory", <1 arguments>, <1 returns>) 
    { 611, 1, 1 }, // FunctionSchema("rand_like", <1 arguments>, <1 returns>) 
    { 612, 2, 1 }, // FunctionSchema("randint_like", <2 arguments>, <1 returns>) 
    { 612, 3, 1 }, // FunctionSchema("randint_like", <3 arguments>, <1 returns>) 
    { 615, 1, 1 }, // FunctionSchema("randn_like", <1 arguments>, <1 returns>) 
    { 616, 2, 1 }, // FunctionSchema("repeat", <2 arguments>, <1 returns>) 
    { 618, 2, 1 }, // FunctionSchema("reshape", <2 arguments>, <1 returns>) 
    { 620, 5, 2 }, // FunctionSchema("RoiPooling2d_forward", <5 arguments>, <2 returns>) 
    { 625, 7, 1 }, // FunctionSchema("RoiPooling2d_backward", <7 arguments>, <1 returns>) 
    { 628, 1, 1 }, // FunctionSchema("round", <1 arguments>, <1 returns>) 
    { 629, 1, 1 }, // FunctionSchema("relu", <1 arguments>, <1 returns>) 
    { 630, 1, 1 }, // FunctionSchema("rsqrt", <1 arguments>, <1 returns>) 
    { 631, 3, 1 }, // FunctionSchema("select", <3 arguments>, <1 returns>) 
    { 632, 1, 1 }, // FunctionSchema("selu", <1 arguments>, <1 returns>) 
    { 633, 1, 1 }, // FunctionSchema("sin", <1 arguments>, <1 returns>) 
    { 634, 1, 1 }, // FunctionSchema("sinh", <1 arguments>, <1 returns>) 
    { 13, 2, 1 }, // FunctionSchema("size", <2 arguments>, <1 returns>) 
    { 635, 5, 1 }, // FunctionSchema("slice", <5 arguments>, <1 returns>) 
    { 636, 1, 2 }, // FunctionSchema("slogdet", <1 arguments>, <2 returns>) 
    { 637, 2, 1 }, // FunctionSchema("smm", <2 arguments>, <1 returns>) 
    { 638, 2, 1 }, // FunctionSchema("softmax", <2 arguments>, <1 returns>) 
    { 639, 4, 1 }, // FunctionSchema("softmax_backward_data", <4 arguments>, <1 returns>) 
    { 640, 3, 1 }, // FunctionSchema("split", <3 arguments>, <1 returns>) 
    { 642, 3, 1 }, // FunctionSchema("split_with_sizes", <3 arguments>, <1 returns>) 
    { 644, 1, 1 }, // FunctionSchema("squeeze", <1 arguments>, <1 returns>) 
    { 644, 2, 1 }, // FunctionSchema("squeeze", <2 arguments>, <1 returns>) 
    { 645, 5, 1 }, // FunctionSchema("sspaddmm", <5 arguments>, <1 returns>) 
    { 646, 2, 1 }, // FunctionSchema("stack", <2 arguments>, <1 returns>) 
    { 647, 8, 1 }, // FunctionSchema("stft", <8 arguments>, <1 returns>) 
    { 203, 2, 1 }, // FunctionSchema("stride", <2 arguments>, <1 returns>) 
    { 653, 1, 1 }, // FunctionSchema("sum", <1 arguments>, <1 returns>) 
    { 654, 1, 1 }, // FunctionSchema("_sum", <1 arguments>, <1 returns>) 
    { 653, 3, 1 }, // FunctionSchema("sum", <3 arguments>, <1 returns>) 
    { 654, 3, 1 }, // FunctionSchema("_sum", <3 arguments>, <1 returns>) 
    { 655, 1, 1 }, // FunctionSchema("sqrt", <1 arguments>, <1 returns>) 
    { 656, 1, 1 }, // FunctionSchema("prod", <1 arguments>, <1 returns>) 
    { 657, 1, 1 }, // FunctionSchema("_prod", <1 arguments>, <1 returns>) 
    { 656, 3, 1 }, // FunctionSchema("prod", <3 arguments>, <1 returns>) 
    { 657, 3, 1 }, // FunctionSchema("_prod", <3 arguments>, <1 returns>) 
    { 658, 1, 1 }, // FunctionSchema("t", <1 arguments>, <1 returns>) 
    { 659, 1, 1 }, // FunctionSchema("tan", <1 arguments>, <1 returns>) 
    { 660, 1, 1 }, // FunctionSchema("tanh", <1 arguments>, <1 returns>) 
    { 166, 3, 1 }, // FunctionSchema("transpose", <3 arguments>, <1 returns>) 
    { 662, 8, 1 }, // FunctionSchema("_trilinear", <8 arguments>, <1 returns>) 
    { 671, 9, 1 }, // FunctionSchema("triplet_margin_loss", <9 arguments>, <1 returns>) 
    { 676, 1, 1 }, // FunctionSchema("trunc", <1 arguments>, <1 returns>) 
    { 677, 2, 1 }, // FunctionSchema("type_as", <2 arguments>, <1 returns>) 
    { 678, 3, 2 }, // FunctionSchema("_unique", <3 arguments>, <2 returns>) 
    { 680, 2, 1 }, // FunctionSchema("_unsafe_view", <2 arguments>, <1 returns>) 
    { 681, 2, 1 }, // FunctionSchema("unsqueeze", <2 arguments>, <1 returns>) 
    { 682, 2, 1 }, // FunctionSchema("view_as", <2 arguments>, <1 returns>) 
    { 683, 3, 1 }, // FunctionSchema("where", <3 arguments>, <1 returns>) 
    { 685, 3, 1 }, // FunctionSchema("_s_where", <3 arguments>, <1 returns>) 
    { 686, 1, 1 }, // FunctionSchema("zeros_like", <1 arguments>, <1 returns>) 
    { 687, 2, 1 }, // FunctionSchema("_standard_gamma_grad", <2 arguments>, <1 returns>) 
    { 688, 1, 1 }, // FunctionSchema("sizes", <1 arguments>, <1 returns>) 
    { 689, 1, 1 }, // FunctionSchema("strides", <1 arguments>, <1 returns>) 
    { 15, 1, 1 }, // FunctionSchema("dim", <1 arguments>, <1 returns>)
  };
  size_t n_operators = 525;

  size_t next_argument = 0;

  auto getArgumentList = [&](uint32_t N){
    std::vector<Argument> result;
    for(size_t i = 0; i < N; ++i) {
      auto & a = arguments[next_argument++];
      result.push_back({ names[a[0]], types[a[1]], tensors[a[2]], attributes[a[3]] });
    }
    return result;
  };

  for(size_t i = 0; i < n_operators; ++i) {
    auto & op = operators[i];
    schemas.push_back({names[op[0]], getArgumentList(op[1]), getArgumentList(op[2])});
  }
  return schemas;
}

std::vector<FunctionSchema> & getOperatorSchemas() {
  static std::vector<FunctionSchema> schema = createOperatorSchemas();
  return schema;
}

static SchemaMap createSchemaMap() {
  auto& schemas = getOperatorSchemas();
  SchemaMap result;
  for(auto & schema : schemas) {
    auto it = result.find(schema.name);
    if(it == result.end()) {
      it = result.insert({schema.name, {}}).first;
    }
    it->second.push_back(std::move(schema));
  }
  return result;
}

const std::vector<FunctionSchema>& getOperatorSchema(const std::string& name) {
  static SchemaMap map = createSchemaMap();
  static std::vector<FunctionSchema> empty;
  auto it = map.find(name);
  if(it != map.end())
    return it->second;
  return empty;
}



}}

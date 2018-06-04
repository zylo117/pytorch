#include "python_nn_functions.h"

// generated from tools/autograd/templates/python_nn_functions.cpp

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils/python_arg_parser.h"

#include "python_nn_functions_dispatch.h"

using at::Tensor;
using at::Scalar;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

static PyObject * THPVariable__sigmoid(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sigmoid(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch__sigmoid(r.tensor(0)));
    } else {
      return wrap(dispatch__sigmoid(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable__tanh(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_tanh(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch__tanh(r.tensor(0)));
    } else {
      return wrap(dispatch__tanh(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_adaptive_avg_pool2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_avg_pool2d(Tensor input, IntList[2] output_size, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_adaptive_avg_pool2d(r.tensor(0), r.intlist(1)));
    } else {
      return wrap(dispatch_adaptive_avg_pool2d(r.tensor(0), r.intlist(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_adaptive_avg_pool3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_avg_pool3d(Tensor input, IntList[3] output_size, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_adaptive_avg_pool3d(r.tensor(0), r.intlist(1)));
    } else {
      return wrap(dispatch_adaptive_avg_pool3d(r.tensor(0), r.intlist(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_adaptive_max_pool2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_max_pool2d(Tensor input, IntList[2] output_size, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_adaptive_max_pool2d(r.tensor(0), r.intlist(1)));
    } else {
      auto results = r.tensorlist_n<2>(2);
      return wrap(dispatch_adaptive_max_pool2d(r.tensor(0), r.intlist(1), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_adaptive_max_pool3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_max_pool3d(Tensor input, IntList[3] output_size, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_adaptive_max_pool3d(r.tensor(0), r.intlist(1)));
    } else {
      auto results = r.tensorlist_n<2>(2);
      return wrap(dispatch_adaptive_max_pool3d(r.tensor(0), r.intlist(1), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_avg_pool2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "avg_pool2d(Tensor input, IntList[2] kernel_size, IntList[2] stride=None, IntList[2] padding=0, bool ceil_mode=False, bool count_include_pad=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(6)) {
      return wrap(dispatch_avg_pool2d(r.tensor(0), r.intlist(1), r.intlist(2), r.intlist(3), r.toBool(4), r.toBool(5)));
    } else {
      return wrap(dispatch_avg_pool2d(r.tensor(0), r.intlist(1), r.intlist(2), r.intlist(3), r.toBool(4), r.toBool(5), r.tensor(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_avg_pool3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "avg_pool3d(Tensor input, IntList[3] kernel_size, IntList[3] stride=None, IntList[3] padding=0, bool ceil_mode=False, bool count_include_pad=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(6)) {
      return wrap(dispatch_avg_pool3d(r.tensor(0), r.intlist(1), r.intlist(2), r.intlist(3), r.toBool(4), r.toBool(5)));
    } else {
      return wrap(dispatch_avg_pool3d(r.tensor(0), r.intlist(1), r.intlist(2), r.intlist(3), r.toBool(4), r.toBool(5), r.tensor(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_binary_cross_entropy(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "binary_cross_entropy(Tensor input, Tensor target, Tensor weight=None, bool size_average=True, bool reduce=True, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_binary_cross_entropy(r.tensor(0), r.tensor(1), r.tensor(2), r.toBool(3), r.toBool(4)));
    } else {
      return wrap(dispatch_binary_cross_entropy(r.tensor(0), r.tensor(1), r.tensor(2), r.toBool(3), r.toBool(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_elu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "elu(Tensor input, Scalar alpha=1, Scalar scale=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_elu(r.tensor(0), r.scalar(1), r.scalar(2)));
    } else {
      return wrap(dispatch_elu(r.tensor(0), r.scalar(1), r.scalar(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_elu_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "elu_(Tensor input, Scalar alpha=1, Scalar scale=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_elu_(r.tensor(0), r.scalar(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_elu_forward_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "elu_forward_(Tensor input, Scalar alpha, Scalar scale)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_elu_forward_(r.tensor(0), r.scalar(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_fractional_max_pool2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fractional_max_pool2d(Tensor input, IntList[2] kernel_size, IntList[2] output_size, Tensor random_samples, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_fractional_max_pool2d(r.tensor(0), r.intlist(1), r.intlist(2), r.tensor(3)));
    } else {
      auto results = r.tensorlist_n<2>(4);
      return wrap(dispatch_fractional_max_pool2d(r.tensor(0), r.intlist(1), r.intlist(2), r.tensor(3), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_glu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "glu(Tensor input, int64_t dim=-1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_glu(r.tensor(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_glu(r.tensor(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_hardshrink(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardshrink(Tensor input, Scalar lambd=0.5, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_hardshrink(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_hardshrink(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_hardtanh(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardtanh(Tensor input, Scalar min_val=-1, Scalar max_val=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_hardtanh(r.tensor(0), r.scalar(1), r.scalar(2)));
    } else {
      return wrap(dispatch_hardtanh(r.tensor(0), r.scalar(1), r.scalar(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_hardtanh_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardtanh_(Tensor input, Scalar min_val=-1, Scalar max_val=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_hardtanh_(r.tensor(0), r.scalar(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_hardtanh_forward_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardtanh_forward_(Tensor input, Scalar min_val, Scalar max_val)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_hardtanh_forward_(r.tensor(0), r.scalar(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_kl_div(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "kl_div(Tensor input, Tensor target, bool size_average=True, bool reduce=True, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_kl_div(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3)));
    } else {
      return wrap(dispatch_kl_div(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_l1_loss(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "l1_loss(Tensor input, Tensor target, bool size_average=True, bool reduce=True, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_l1_loss(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3)));
    } else {
      return wrap(dispatch_l1_loss(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_leaky_relu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "leaky_relu(Tensor input, Scalar negative_slope=0.01, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_leaky_relu(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_leaky_relu(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_leaky_relu_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "leaky_relu_(Tensor input, Scalar negative_slope=0.01)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_leaky_relu_(r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_leaky_relu_forward_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "leaky_relu_forward_(Tensor input, Scalar negative_slope)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_leaky_relu_forward_(r.tensor(0), r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_log_sigmoid(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log_sigmoid(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(1)) {
      return wrap(dispatch_log_sigmoid(r.tensor(0)));
    } else {
      return wrap(dispatch_log_sigmoid(r.tensor(0), r.tensor(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_max_pool2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool2d(Tensor input, IntList[2] kernel_size, IntList[2] stride=None, IntList[2] padding=0, IntList[2] dilation=1, bool ceil_mode=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(6)) {
      return wrap(dispatch_max_pool2d(r.tensor(0), r.intlist(1), r.intlist(2), r.intlist(3), r.intlist(4), r.toBool(5)));
    } else {
      auto results = r.tensorlist_n<2>(6);
      return wrap(dispatch_max_pool2d(r.tensor(0), r.intlist(1), r.intlist(2), r.intlist(3), r.intlist(4), r.toBool(5), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_max_pool3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool3d(Tensor input, IntList[3] kernel_size, IntList[3] stride=None, IntList[3] padding=0, IntList[3] dilation=1, bool ceil_mode=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(6)) {
      return wrap(dispatch_max_pool3d(r.tensor(0), r.intlist(1), r.intlist(2), r.intlist(3), r.intlist(4), r.toBool(5)));
    } else {
      auto results = r.tensorlist_n<2>(6);
      return wrap(dispatch_max_pool3d(r.tensor(0), r.intlist(1), r.intlist(2), r.intlist(3), r.intlist(4), r.toBool(5), results[0], results[1]));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_max_unpool2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_unpool2d(Tensor input, Tensor indices, IntList[2] output_size, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_max_unpool2d(r.tensor(0), r.tensor(1), r.intlist(2)));
    } else {
      return wrap(dispatch_max_unpool2d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_max_unpool3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_unpool3d(Tensor input, Tensor indices, IntList[3] output_size, IntList[3] stride, IntList[3] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(5)) {
      return wrap(dispatch_max_unpool3d(r.tensor(0), r.tensor(1), r.intlist(2), r.intlist(3), r.intlist(4)));
    } else {
      return wrap(dispatch_max_unpool3d(r.tensor(0), r.tensor(1), r.intlist(2), r.intlist(3), r.intlist(4), r.tensor(5)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_mse_loss(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mse_loss(Tensor input, Tensor target, bool size_average=True, bool reduce=True, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_mse_loss(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3)));
    } else {
      return wrap(dispatch_mse_loss(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_multi_margin_loss(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "multi_margin_loss(Tensor input, Tensor target, Scalar p=1, Scalar margin=1, Tensor weight=None, bool size_average=True, bool reduce=True, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(7)) {
      return wrap(dispatch_multi_margin_loss(r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3), r.tensor(4), r.toBool(5), r.toBool(6)));
    } else {
      return wrap(dispatch_multi_margin_loss(r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3), r.tensor(4), r.toBool(5), r.toBool(6), r.tensor(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_multilabel_margin_loss(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "multilabel_margin_loss(Tensor input, Tensor target, bool size_average=True, bool reduce=True, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_multilabel_margin_loss(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3)));
    } else {
      return wrap(dispatch_multilabel_margin_loss(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_nll_loss(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nll_loss(Tensor input, Tensor target, Tensor weight=None, bool size_average=True, int64_t ignore_index=-100, bool reduce=True, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(6)) {
      return wrap(dispatch_nll_loss(r.tensor(0), r.tensor(1), r.tensor(2), r.toBool(3), r.toInt64(4), r.toBool(5)));
    } else {
      return wrap(dispatch_nll_loss(r.tensor(0), r.tensor(1), r.tensor(2), r.toBool(3), r.toInt64(4), r.toBool(5), r.tensor(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_nll_loss2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nll_loss2d(Tensor input, Tensor target, Tensor weight=None, bool size_average=True, int64_t ignore_index=-100, bool reduce=True, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(6)) {
      return wrap(dispatch_nll_loss2d(r.tensor(0), r.tensor(1), r.tensor(2), r.toBool(3), r.toInt64(4), r.toBool(5)));
    } else {
      return wrap(dispatch_nll_loss2d(r.tensor(0), r.tensor(1), r.tensor(2), r.toBool(3), r.toInt64(4), r.toBool(5), r.tensor(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_prelu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "prelu(Tensor input, Tensor weight, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_prelu(r.tensor(0), r.tensor(1)));
    } else {
      return wrap(dispatch_prelu(r.tensor(0), r.tensor(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_reflection_pad1d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reflection_pad1d(Tensor input, IntList[2] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_reflection_pad1d(r.tensor(0), r.intlist(1)));
    } else {
      return wrap(dispatch_reflection_pad1d(r.tensor(0), r.intlist(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_reflection_pad2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reflection_pad2d(Tensor input, IntList[4] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_reflection_pad2d(r.tensor(0), r.intlist(1)));
    } else {
      return wrap(dispatch_reflection_pad2d(r.tensor(0), r.intlist(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_replication_pad1d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "replication_pad1d(Tensor input, IntList[2] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_replication_pad1d(r.tensor(0), r.intlist(1)));
    } else {
      return wrap(dispatch_replication_pad1d(r.tensor(0), r.intlist(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_replication_pad2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "replication_pad2d(Tensor input, IntList[4] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_replication_pad2d(r.tensor(0), r.intlist(1)));
    } else {
      return wrap(dispatch_replication_pad2d(r.tensor(0), r.intlist(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_replication_pad3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "replication_pad3d(Tensor input, IntList[6] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_replication_pad3d(r.tensor(0), r.intlist(1)));
    } else {
      return wrap(dispatch_replication_pad3d(r.tensor(0), r.intlist(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rrelu_with_noise(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu_with_noise(Tensor input, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator generator=None, *, Tensor out=None)",
  }, /*traceable=*/false);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(6)) {
      return wrap(dispatch_rrelu_with_noise(r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3), r.toBool(4), r.generator(5)));
    } else {
      return wrap(dispatch_rrelu_with_noise(r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3), r.toBool(4), r.generator(5), r.tensor(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rrelu_with_noise_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu_with_noise_(Tensor input, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator generator=None)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rrelu_with_noise_(r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3), r.toBool(4), r.generator(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_rrelu_with_noise_forward_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu_with_noise_forward_(Tensor input, Tensor noise, Scalar lower, Scalar upper, bool training, Generator generator)",
  }, /*traceable=*/false);

  ParsedArgs<6> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_rrelu_with_noise_forward_(r.tensor(0), r.tensor(1), r.scalar(2), r.scalar(3), r.toBool(4), r.generator(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_smooth_l1_loss(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "smooth_l1_loss(Tensor input, Tensor target, bool size_average=True, bool reduce=True, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_smooth_l1_loss(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3)));
    } else {
      return wrap(dispatch_smooth_l1_loss(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_soft_margin_loss(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "soft_margin_loss(Tensor input, Tensor target, bool size_average=True, bool reduce=True, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(4)) {
      return wrap(dispatch_soft_margin_loss(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3)));
    } else {
      return wrap(dispatch_soft_margin_loss(r.tensor(0), r.tensor(1), r.toBool(2), r.toBool(3), r.tensor(4)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_softplus(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "softplus(Tensor input, Scalar beta=1, Scalar threshold=20, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_softplus(r.tensor(0), r.scalar(1), r.scalar(2)));
    } else {
      return wrap(dispatch_softplus(r.tensor(0), r.scalar(1), r.scalar(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_softshrink(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "softshrink(Tensor input, Scalar lambd=0.5, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_softshrink(r.tensor(0), r.scalar(1)));
    } else {
      return wrap(dispatch_softshrink(r.tensor(0), r.scalar(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_thnn_batch_norm(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "thnn_batch_norm(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, bool training, double momentum, double eps, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(8)) {
      return wrap(dispatch_thnn_batch_norm(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4), r.toBool(5), r.toDouble(6), r.toDouble(7)));
    } else {
      return wrap(dispatch_thnn_batch_norm(r.tensor(0), r.tensor(1), r.tensor(2), r.tensor(3), r.tensor(4), r.toBool(5), r.toDouble(6), r.toDouble(7), r.tensor(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_thnn_conv2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "thnn_conv2d(Tensor input, Tensor weight, IntList[2] kernel_size, Tensor bias=None, IntList[2] stride=1, IntList[2] padding=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(6)) {
      return wrap(dispatch_thnn_conv2d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3), r.intlist(4), r.intlist(5)));
    } else {
      return wrap(dispatch_thnn_conv2d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3), r.intlist(4), r.intlist(5), r.tensor(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_thnn_conv3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "thnn_conv3d(Tensor input, Tensor weight, IntList[3] kernel_size, Tensor bias=None, IntList[3] stride=1, IntList[3] padding=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(6)) {
      return wrap(dispatch_thnn_conv3d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3), r.intlist(4), r.intlist(5)));
    } else {
      return wrap(dispatch_thnn_conv3d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3), r.intlist(4), r.intlist(5), r.tensor(6)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_thnn_conv_depthwise2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "thnn_conv_depthwise2d(Tensor input, Tensor weight, IntList[2] kernel_size, Tensor bias=None, IntList[2] stride=1, IntList[2] padding=0, IntList[2] dilation=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(7)) {
      return wrap(dispatch_thnn_conv_depthwise2d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3), r.intlist(4), r.intlist(5), r.intlist(6)));
    } else {
      return wrap(dispatch_thnn_conv_depthwise2d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3), r.intlist(4), r.intlist(5), r.intlist(6), r.tensor(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_thnn_conv_dilated2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "thnn_conv_dilated2d(Tensor input, Tensor weight, IntList[2] kernel_size, Tensor bias=None, IntList[2] stride=1, IntList[2] padding=0, IntList[2] dilation=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(7)) {
      return wrap(dispatch_thnn_conv_dilated2d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3), r.intlist(4), r.intlist(5), r.intlist(6)));
    } else {
      return wrap(dispatch_thnn_conv_dilated2d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3), r.intlist(4), r.intlist(5), r.intlist(6), r.tensor(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_thnn_conv_dilated3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "thnn_conv_dilated3d(Tensor input, Tensor weight, IntList[3] kernel_size, Tensor bias=None, IntList[3] stride=1, IntList[3] padding=0, IntList[3] dilation=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(7)) {
      return wrap(dispatch_thnn_conv_dilated3d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3), r.intlist(4), r.intlist(5), r.intlist(6)));
    } else {
      return wrap(dispatch_thnn_conv_dilated3d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3), r.intlist(4), r.intlist(5), r.intlist(6), r.tensor(7)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_thnn_conv_transpose2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "thnn_conv_transpose2d(Tensor input, Tensor weight, IntList[2] kernel_size, Tensor bias=None, IntList[2] stride=1, IntList[2] padding=0, IntList[2] output_padding=0, IntList[2] dilation=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(8)) {
      return wrap(dispatch_thnn_conv_transpose2d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3), r.intlist(4), r.intlist(5), r.intlist(6), r.intlist(7)));
    } else {
      return wrap(dispatch_thnn_conv_transpose2d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3), r.intlist(4), r.intlist(5), r.intlist(6), r.intlist(7), r.tensor(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_thnn_conv_transpose3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "thnn_conv_transpose3d(Tensor input, Tensor weight, IntList[3] kernel_size, Tensor bias=None, IntList[3] stride=1, IntList[3] padding=0, IntList[3] output_padding=0, IntList[3] dilation=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(8)) {
      return wrap(dispatch_thnn_conv_transpose3d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3), r.intlist(4), r.intlist(5), r.intlist(6), r.intlist(7)));
    } else {
      return wrap(dispatch_thnn_conv_transpose3d(r.tensor(0), r.tensor(1), r.intlist(2), r.tensor(3), r.intlist(4), r.intlist(5), r.intlist(6), r.intlist(7), r.tensor(8)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_threshold(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "threshold(Tensor input, Scalar threshold, Scalar value, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_threshold(r.tensor(0), r.scalar(1), r.scalar(2)));
    } else {
      return wrap(dispatch_threshold(r.tensor(0), r.scalar(1), r.scalar(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_threshold_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "threshold_(Tensor input, Scalar threshold, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_threshold_(r.tensor(0), r.scalar(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_threshold_forward_(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "threshold_forward_(Tensor input, Scalar threshold, Scalar value)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    return wrap(dispatch_threshold_forward_(r.tensor(0), r.scalar(1), r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_upsample_bilinear2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_bilinear2d(Tensor input, IntList[2] output_size, bool align_corners, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_upsample_bilinear2d(r.tensor(0), r.intlist(1), r.toBool(2)));
    } else {
      return wrap(dispatch_upsample_bilinear2d(r.tensor(0), r.intlist(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_upsample_linear1d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_linear1d(Tensor input, IntList[1] output_size, bool align_corners, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_upsample_linear1d(r.tensor(0), r.intlist(1), r.toBool(2)));
    } else {
      return wrap(dispatch_upsample_linear1d(r.tensor(0), r.intlist(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_upsample_nearest1d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_nearest1d(Tensor input, int64_t scale_factor, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_upsample_nearest1d(r.tensor(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_upsample_nearest1d(r.tensor(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_upsample_nearest2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_nearest2d(Tensor input, int64_t scale_factor, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_upsample_nearest2d(r.tensor(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_upsample_nearest2d(r.tensor(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_upsample_nearest3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_nearest3d(Tensor input, int64_t scale_factor, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(2)) {
      return wrap(dispatch_upsample_nearest3d(r.tensor(0), r.toInt64(1)));
    } else {
      return wrap(dispatch_upsample_nearest3d(r.tensor(0), r.toInt64(1), r.tensor(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
static PyObject * THPVariable_upsample_trilinear3d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_trilinear3d(Tensor input, IntList[3] output_size, bool align_corners, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    if (r.isNone(3)) {
      return wrap(dispatch_upsample_trilinear3d(r.tensor(0), r.intlist(1), r.toBool(2)));
    } else {
      return wrap(dispatch_upsample_trilinear3d(r.tensor(0), r.intlist(1), r.toBool(2), r.tensor(3)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyMethodDef nn_functions[] = {
  {"_sigmoid", (PyCFunction)THPVariable__sigmoid, METH_VARARGS | METH_KEYWORDS, NULL},
  {"_tanh", (PyCFunction)THPVariable__tanh, METH_VARARGS | METH_KEYWORDS, NULL},
  {"adaptive_avg_pool2d", (PyCFunction)THPVariable_adaptive_avg_pool2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"adaptive_avg_pool3d", (PyCFunction)THPVariable_adaptive_avg_pool3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"adaptive_max_pool2d", (PyCFunction)THPVariable_adaptive_max_pool2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"adaptive_max_pool3d", (PyCFunction)THPVariable_adaptive_max_pool3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"avg_pool2d", (PyCFunction)THPVariable_avg_pool2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"avg_pool3d", (PyCFunction)THPVariable_avg_pool3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"binary_cross_entropy", (PyCFunction)THPVariable_binary_cross_entropy, METH_VARARGS | METH_KEYWORDS, NULL},
  {"elu", (PyCFunction)THPVariable_elu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"elu_", (PyCFunction)THPVariable_elu_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"elu_forward_", (PyCFunction)THPVariable_elu_forward_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"fractional_max_pool2d", (PyCFunction)THPVariable_fractional_max_pool2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"glu", (PyCFunction)THPVariable_glu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"hardshrink", (PyCFunction)THPVariable_hardshrink, METH_VARARGS | METH_KEYWORDS, NULL},
  {"hardtanh", (PyCFunction)THPVariable_hardtanh, METH_VARARGS | METH_KEYWORDS, NULL},
  {"hardtanh_", (PyCFunction)THPVariable_hardtanh_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"hardtanh_forward_", (PyCFunction)THPVariable_hardtanh_forward_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"kl_div", (PyCFunction)THPVariable_kl_div, METH_VARARGS | METH_KEYWORDS, NULL},
  {"l1_loss", (PyCFunction)THPVariable_l1_loss, METH_VARARGS | METH_KEYWORDS, NULL},
  {"leaky_relu", (PyCFunction)THPVariable_leaky_relu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"leaky_relu_", (PyCFunction)THPVariable_leaky_relu_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"leaky_relu_forward_", (PyCFunction)THPVariable_leaky_relu_forward_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"log_sigmoid", (PyCFunction)THPVariable_log_sigmoid, METH_VARARGS | METH_KEYWORDS, NULL},
  {"max_pool2d", (PyCFunction)THPVariable_max_pool2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"max_pool3d", (PyCFunction)THPVariable_max_pool3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"max_unpool2d", (PyCFunction)THPVariable_max_unpool2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"max_unpool3d", (PyCFunction)THPVariable_max_unpool3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"mse_loss", (PyCFunction)THPVariable_mse_loss, METH_VARARGS | METH_KEYWORDS, NULL},
  {"multi_margin_loss", (PyCFunction)THPVariable_multi_margin_loss, METH_VARARGS | METH_KEYWORDS, NULL},
  {"multilabel_margin_loss", (PyCFunction)THPVariable_multilabel_margin_loss, METH_VARARGS | METH_KEYWORDS, NULL},
  {"nll_loss", (PyCFunction)THPVariable_nll_loss, METH_VARARGS | METH_KEYWORDS, NULL},
  {"nll_loss2d", (PyCFunction)THPVariable_nll_loss2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"prelu", (PyCFunction)THPVariable_prelu, METH_VARARGS | METH_KEYWORDS, NULL},
  {"reflection_pad1d", (PyCFunction)THPVariable_reflection_pad1d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"reflection_pad2d", (PyCFunction)THPVariable_reflection_pad2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"replication_pad1d", (PyCFunction)THPVariable_replication_pad1d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"replication_pad2d", (PyCFunction)THPVariable_replication_pad2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"replication_pad3d", (PyCFunction)THPVariable_replication_pad3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"rrelu_with_noise", (PyCFunction)THPVariable_rrelu_with_noise, METH_VARARGS | METH_KEYWORDS, NULL},
  {"rrelu_with_noise_", (PyCFunction)THPVariable_rrelu_with_noise_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"rrelu_with_noise_forward_", (PyCFunction)THPVariable_rrelu_with_noise_forward_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"smooth_l1_loss", (PyCFunction)THPVariable_smooth_l1_loss, METH_VARARGS | METH_KEYWORDS, NULL},
  {"soft_margin_loss", (PyCFunction)THPVariable_soft_margin_loss, METH_VARARGS | METH_KEYWORDS, NULL},
  {"softplus", (PyCFunction)THPVariable_softplus, METH_VARARGS | METH_KEYWORDS, NULL},
  {"softshrink", (PyCFunction)THPVariable_softshrink, METH_VARARGS | METH_KEYWORDS, NULL},
  {"thnn_batch_norm", (PyCFunction)THPVariable_thnn_batch_norm, METH_VARARGS | METH_KEYWORDS, NULL},
  {"thnn_conv2d", (PyCFunction)THPVariable_thnn_conv2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"thnn_conv3d", (PyCFunction)THPVariable_thnn_conv3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"thnn_conv_depthwise2d", (PyCFunction)THPVariable_thnn_conv_depthwise2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"thnn_conv_dilated2d", (PyCFunction)THPVariable_thnn_conv_dilated2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"thnn_conv_dilated3d", (PyCFunction)THPVariable_thnn_conv_dilated3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"thnn_conv_transpose2d", (PyCFunction)THPVariable_thnn_conv_transpose2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"thnn_conv_transpose3d", (PyCFunction)THPVariable_thnn_conv_transpose3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"threshold", (PyCFunction)THPVariable_threshold, METH_VARARGS | METH_KEYWORDS, NULL},
  {"threshold_", (PyCFunction)THPVariable_threshold_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"threshold_forward_", (PyCFunction)THPVariable_threshold_forward_, METH_VARARGS | METH_KEYWORDS, NULL},
  {"upsample_bilinear2d", (PyCFunction)THPVariable_upsample_bilinear2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"upsample_linear1d", (PyCFunction)THPVariable_upsample_linear1d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"upsample_nearest1d", (PyCFunction)THPVariable_upsample_nearest1d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"upsample_nearest2d", (PyCFunction)THPVariable_upsample_nearest2d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"upsample_nearest3d", (PyCFunction)THPVariable_upsample_nearest3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {"upsample_trilinear3d", (PyCFunction)THPVariable_upsample_trilinear3d, METH_VARARGS | METH_KEYWORDS, NULL},
  {NULL}
};

void initNNFunctions(PyObject* module) {
#if PY_MAJOR_VERSION == 2
  PyObject* nn = Py_InitModule("torch._C._nn", nn_functions);
  Py_XINCREF(nn);  // Py_InitModule returns "borrowed" reference
#else
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._nn",
     NULL,
     -1,
     nn_functions
  };
  PyObject* nn = PyModule_Create(&def);
#endif
  if (!nn) {
    throw python_error();
  }
  // steals a reference to nn
  if (PyModule_AddObject(module, "_nn", nn) != 0) {
    throw python_error();
  }
}

}} // namespace torch::autograd

#include "torch/csrc/jit/aten_dispatch.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/tensor_conversions.h"
#include "torch/csrc/utils/functional.h"

#include <unordered_map>
#include <cstring>
#include <tuple>

// generated from tools/autograd/templates/aten_dispatch.cpp

namespace torch { namespace jit {

using autograd::Variable;
using autograd::variable_list;
using at::Scalar;
using at::Tensor;
using at::IntList;
using at::TensorList;

namespace {

// The packer here is carefully written not to make any unnecessary
// copies.

// pack takes the return values of aten functions pushes them onto the stack
template<typename T>
void pack(Stack & stack, T&& v) {
  stack.push_back(as_variable(std::move(v)));
}
template<>
void pack(Stack & stack, Tensor&& v) {
  stack.push_back(std::move(v));
}
template<>
void pack(Stack & stack, std::vector<Tensor>&& ts) {
  for(auto& t : ts) {
    stack.push_back(std::move(t));
  }
}

template<std::size_t remaining, typename... Args>
struct TuplePacker
{
  // NB: *Not* a universal reference.
  static void execute(Stack & stack, std::tuple<Args...> && t)
  {
    // NB: The move here does not "destroy" the entire tuple, that is
    // not what std::move does; only the particular tuple index
    // processed here gets stolen.
    pack(stack, std::get<sizeof...(Args) - remaining>(std::move(t)));
    TuplePacker<remaining - 1, Args...>::execute(stack, std::move(t));
  }
};

template<typename... Args>
struct TuplePacker<0, Args...>
{
  static void execute(Stack & stack, std::tuple<Args...> && t) {};
};

template<typename... Args>
void pack(Stack & stack, std::tuple<Args...> && t) {
  TuplePacker<sizeof...(Args), Args...>::execute(stack, std::move(t));
}

int deviceForInputs(Stack & stack, size_t N) {
  if(N == 0)
    return -1;
  auto & t = *(stack.end() - N);
  return t.type().is_cuda() ? (int) t.get_device() : -1;
}

// A list of functions taking TensorList arguments (where we can't use
// the number of inputs to choose an overload).
std::unordered_set<Symbol> tensor_vararg_fns = {
  aten::cat,
  aten::stack,
};

template<size_t N>
std::array<bool, N> as_bool_array(const std::vector<int64_t>& vec) {
  std::array<bool, N> res;
  JIT_ASSERT(vec.size() == N);
  std::copy(vec.begin(), vec.end(), res.begin());
  return res;
}

using operator_constructor = std::function<TensorOp(jit::Node*)>;
std::unordered_map<std::string, operator_constructor> constructors = {
  {"RoiPooling2d_backward-4-pooledHeight_i-pooledWidth_i-spatialScale_f", [](Node *node) {
    auto pooledHeight = int64_t(node->i(Symbol::attr("pooledHeight")));
    auto pooledWidth = int64_t(node->i(Symbol::attr("pooledWidth")));
    auto spatialScale = double(node->f(Symbol::attr("spatialScale")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("RoiPooling2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
  
      auto result = at::RoiPooling2d_backward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), pooledHeight, pooledWidth, spatialScale, std::move(peek(stack, 2, 4)), std::move(peek(stack, 3, 4)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "RoiPooling2d_backward", 4, 1);
  }},
  {"RoiPooling2d_backward-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("RoiPooling2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto pooledHeight = tensor_as<int64_t>(std::move(peek(stack, 2, 7)));
      auto pooledWidth = tensor_as<int64_t>(std::move(peek(stack, 3, 7)));
      auto spatialScale = tensor_as<double>(std::move(peek(stack, 4, 7)));
      auto result = at::RoiPooling2d_backward(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), pooledHeight, pooledWidth, spatialScale, std::move(peek(stack, 5, 7)), std::move(peek(stack, 6, 7)));
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "RoiPooling2d_backward", 7, 1);
  }},
  {"RoiPooling2d_forward-2-pooledHeight_i-pooledWidth_i-spatialScale_f", [](Node *node) {
    auto pooledHeight = int64_t(node->i(Symbol::attr("pooledHeight")));
    auto pooledWidth = int64_t(node->i(Symbol::attr("pooledWidth")));
    auto spatialScale = double(node->f(Symbol::attr("spatialScale")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("RoiPooling2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::RoiPooling2d_forward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), pooledHeight, pooledWidth, spatialScale);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "RoiPooling2d_forward", 2, 2);
  }},
  {"RoiPooling2d_forward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("RoiPooling2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto pooledHeight = tensor_as<int64_t>(std::move(peek(stack, 2, 5)));
      auto pooledWidth = tensor_as<int64_t>(std::move(peek(stack, 3, 5)));
      auto spatialScale = tensor_as<double>(std::move(peek(stack, 4, 5)));
      auto result = at::RoiPooling2d_forward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), pooledHeight, pooledWidth, spatialScale);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "RoiPooling2d_forward", 5, 2);
  }},
  {"__and__-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__and__");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::__and__(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "__and__", 1, 1);
  }},
  {"__and__-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__and__");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::__and__(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "__and__", 2, 1);
  }},
  {"__iand__-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__iand__");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).__iand__(other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "__iand__", 1, 1);
  }},
  {"__iand__-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__iand__");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = (std::move(peek(stack, 0, 2))).__iand__(std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "__iand__", 2, 1);
  }},
  {"__ilshift__-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__ilshift__");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).__ilshift__(other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "__ilshift__", 1, 1);
  }},
  {"__ilshift__-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__ilshift__");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = (std::move(peek(stack, 0, 2))).__ilshift__(std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "__ilshift__", 2, 1);
  }},
  {"__ior__-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__ior__");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).__ior__(other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "__ior__", 1, 1);
  }},
  {"__ior__-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__ior__");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = (std::move(peek(stack, 0, 2))).__ior__(std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "__ior__", 2, 1);
  }},
  {"__irshift__-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__irshift__");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).__irshift__(other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "__irshift__", 1, 1);
  }},
  {"__irshift__-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__irshift__");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = (std::move(peek(stack, 0, 2))).__irshift__(std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "__irshift__", 2, 1);
  }},
  {"__ixor__-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__ixor__");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).__ixor__(other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "__ixor__", 1, 1);
  }},
  {"__ixor__-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__ixor__");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = (std::move(peek(stack, 0, 2))).__ixor__(std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "__ixor__", 2, 1);
  }},
  {"__lshift__-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__lshift__");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::__lshift__(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "__lshift__", 1, 1);
  }},
  {"__lshift__-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__lshift__");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::__lshift__(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "__lshift__", 2, 1);
  }},
  {"__or__-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__or__");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::__or__(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "__or__", 1, 1);
  }},
  {"__or__-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__or__");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::__or__(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "__or__", 2, 1);
  }},
  {"__rshift__-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__rshift__");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::__rshift__(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "__rshift__", 1, 1);
  }},
  {"__rshift__-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__rshift__");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::__rshift__(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "__rshift__", 2, 1);
  }},
  {"__xor__-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__xor__");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::__xor__(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "__xor__", 1, 1);
  }},
  {"__xor__-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("__xor__");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::__xor__(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "__xor__", 2, 1);
  }},
  {"_abs-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_abs");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_abs(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_abs", 1, 1);
  }},
  {"_acos-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_acos");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_acos(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_acos", 1, 1);
  }},
  {"_addmv-3-alpha_t-beta_t", [](Node *node) {
    auto beta = Scalar(node->t(Symbol::attr("beta")));
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_addmv");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::_addmv(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_addmv", 3, 1);
  }},
  {"_addmv-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_addmv");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto beta = tensor_as<Scalar>(std::move(peek(stack, 3, 5)));
      auto alpha = tensor_as<Scalar>(std::move(peek(stack, 4, 5)));
      auto result = at::_addmv(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "_addmv", 5, 1);
  }},
  {"_addr-3-alpha_t-beta_t", [](Node *node) {
    auto beta = Scalar(node->t(Symbol::attr("beta")));
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_addr");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::_addr(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_addr", 3, 1);
  }},
  {"_addr-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_addr");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto beta = tensor_as<Scalar>(std::move(peek(stack, 3, 5)));
      auto alpha = tensor_as<Scalar>(std::move(peek(stack, 4, 5)));
      auto result = at::_addr(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "_addr", 5, 1);
  }},
  {"_argmax-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_argmax");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_argmax(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_argmax", 1, 1);
  }},
  {"_argmax-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_argmax");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::_argmax(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_argmax", 3, 1);
  }},
  {"_argmin-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_argmin");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_argmin(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_argmin", 1, 1);
  }},
  {"_argmin-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_argmin");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::_argmin(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_argmin", 3, 1);
  }},
  {"_asin-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_asin");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_asin(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_asin", 1, 1);
  }},
  {"_atan-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_atan");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_atan(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_atan", 1, 1);
  }},
  {"_cast_Half-1-non_blocking_i", [](Node *node) {
    auto non_blocking = bool(node->i(Symbol::attr("non_blocking")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_Half");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_cast_Half(std::move(peek(stack, 0, 1)), non_blocking);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_Half", 1, 1);
  }},
  {"_cast_Half-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_Half");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto non_blocking = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::_cast_Half(std::move(peek(stack, 0, 2)), non_blocking);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_Half", 2, 1);
  }},
  {"_cast_double-1-non_blocking_i", [](Node *node) {
    auto non_blocking = bool(node->i(Symbol::attr("non_blocking")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_double");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_cast_double(std::move(peek(stack, 0, 1)), non_blocking);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_double", 1, 1);
  }},
  {"_cast_double-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_double");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto non_blocking = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::_cast_double(std::move(peek(stack, 0, 2)), non_blocking);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_double", 2, 1);
  }},
  {"_cast_float-1-non_blocking_i", [](Node *node) {
    auto non_blocking = bool(node->i(Symbol::attr("non_blocking")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_float");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_cast_float(std::move(peek(stack, 0, 1)), non_blocking);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_float", 1, 1);
  }},
  {"_cast_float-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_float");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto non_blocking = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::_cast_float(std::move(peek(stack, 0, 2)), non_blocking);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_float", 2, 1);
  }},
  {"_cast_int-1-non_blocking_i", [](Node *node) {
    auto non_blocking = bool(node->i(Symbol::attr("non_blocking")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_int");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_cast_int(std::move(peek(stack, 0, 1)), non_blocking);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_int", 1, 1);
  }},
  {"_cast_int-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_int");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto non_blocking = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::_cast_int(std::move(peek(stack, 0, 2)), non_blocking);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_int", 2, 1);
  }},
  {"_cast_int16_t-1-non_blocking_i", [](Node *node) {
    auto non_blocking = bool(node->i(Symbol::attr("non_blocking")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_int16_t");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_cast_int16_t(std::move(peek(stack, 0, 1)), non_blocking);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_int16_t", 1, 1);
  }},
  {"_cast_int16_t-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_int16_t");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto non_blocking = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::_cast_int16_t(std::move(peek(stack, 0, 2)), non_blocking);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_int16_t", 2, 1);
  }},
  {"_cast_int64_t-1-non_blocking_i", [](Node *node) {
    auto non_blocking = bool(node->i(Symbol::attr("non_blocking")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_int64_t");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_cast_int64_t(std::move(peek(stack, 0, 1)), non_blocking);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_int64_t", 1, 1);
  }},
  {"_cast_int64_t-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_int64_t");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto non_blocking = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::_cast_int64_t(std::move(peek(stack, 0, 2)), non_blocking);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_int64_t", 2, 1);
  }},
  {"_cast_int8_t-1-non_blocking_i", [](Node *node) {
    auto non_blocking = bool(node->i(Symbol::attr("non_blocking")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_int8_t");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_cast_int8_t(std::move(peek(stack, 0, 1)), non_blocking);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_int8_t", 1, 1);
  }},
  {"_cast_int8_t-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_int8_t");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto non_blocking = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::_cast_int8_t(std::move(peek(stack, 0, 2)), non_blocking);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_int8_t", 2, 1);
  }},
  {"_cast_uint8_t-1-non_blocking_i", [](Node *node) {
    auto non_blocking = bool(node->i(Symbol::attr("non_blocking")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_uint8_t");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_cast_uint8_t(std::move(peek(stack, 0, 1)), non_blocking);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_uint8_t", 1, 1);
  }},
  {"_cast_uint8_t-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cast_uint8_t");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto non_blocking = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::_cast_uint8_t(std::move(peek(stack, 0, 2)), non_blocking);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_cast_uint8_t", 2, 1);
  }},
  {"_cat-*", [](Node *node) {
    size_t varargs_length = node->inputs().size();
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cat");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 0, 1)));
      auto result = at::_cat(peekSlice(stack, 0, varargs_length - 1, varargs_length), dim);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "_cat", varargs_length, 1);
  }},
  {"_cat-*-dim_i", [](Node *node) {
    size_t varargs_length = node->inputs().size();
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cat");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length));
  
      auto result = at::_cat(peekSlice(stack, 0, varargs_length - 0, varargs_length), dim);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "_cat", varargs_length, 1);
  }},
  {"_ceil-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_ceil");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_ceil(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_ceil", 1, 1);
  }},
  {"_convolution-12", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_convolution");
      AutoGPU device_guard(deviceForInputs(stack, 12));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 3, 12)));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 4, 12)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 5, 12)));
      auto transposed = tensor_as<bool>(std::move(peek(stack, 6, 12)));
      auto output_padding = tensor_as<IntList>(std::move(peek(stack, 7, 12)));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 8, 12)));
      auto benchmark = tensor_as<bool>(std::move(peek(stack, 9, 12)));
      auto deterministic = tensor_as<bool>(std::move(peek(stack, 10, 12)));
      auto cudnn_enabled = tensor_as<bool>(std::move(peek(stack, 11, 12)));
      auto result = at::_convolution(std::move(peek(stack, 0, 12)), std::move(peek(stack, 1, 12)), std::move(peek(stack, 2, 12)), stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
      drop(stack, 12);
      pack(stack, std::move(result));
      return 0;
    }, "_convolution", 12, 1);
  }},
  {"_convolution-3-benchmark_i-cudnn_enabled_i-deterministic_i-dilation_is-groups_i-output_padding_is-padding_is-stride_is-transposed_i", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto transposed = bool(node->i(Symbol::attr("transposed")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    auto benchmark = bool(node->i(Symbol::attr("benchmark")));
    auto deterministic = bool(node->i(Symbol::attr("deterministic")));
    auto cudnn_enabled = bool(node->i(Symbol::attr("cudnn_enabled")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_convolution");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::_convolution(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_convolution", 3, 1);
  }},
  {"_convolution_double_backward-16", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_convolution_double_backward");
      AutoGPU device_guard(deviceForInputs(stack, 16));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 6, 16)));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 7, 16)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 8, 16)));
      auto transposed = tensor_as<bool>(std::move(peek(stack, 9, 16)));
      auto output_padding = tensor_as<IntList>(std::move(peek(stack, 10, 16)));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 11, 16)));
      auto benchmark = tensor_as<bool>(std::move(peek(stack, 12, 16)));
      auto deterministic = tensor_as<bool>(std::move(peek(stack, 13, 16)));
      auto cudnn_enabled = tensor_as<bool>(std::move(peek(stack, 14, 16)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(peek(stack, 15, 16)));
      auto result = at::_convolution_double_backward(std::move(peek(stack, 0, 16)), std::move(peek(stack, 1, 16)), std::move(peek(stack, 2, 16)), std::move(peek(stack, 3, 16)), std::move(peek(stack, 4, 16)), std::move(peek(stack, 5, 16)), stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask);
      drop(stack, 16);
      pack(stack, std::move(result));
      return 0;
    }, "_convolution_double_backward", 16, 3);
  }},
  {"_convolution_double_backward-6-benchmark_i-cudnn_enabled_i-deterministic_i-dilation_is-groups_i-output_mask_is-output_padding_is-padding_is-stride_is-transposed_i", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto transposed = bool(node->i(Symbol::attr("transposed")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    auto benchmark = bool(node->i(Symbol::attr("benchmark")));
    auto deterministic = bool(node->i(Symbol::attr("deterministic")));
    auto cudnn_enabled = bool(node->i(Symbol::attr("cudnn_enabled")));
    auto output_mask = as_bool_array<3>(node->is(Symbol::attr("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_convolution_double_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
  
      auto result = at::_convolution_double_backward(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), std::move(peek(stack, 3, 6)), std::move(peek(stack, 4, 6)), std::move(peek(stack, 5, 6)), stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "_convolution_double_backward", 6, 3);
  }},
  {"_convolution_nogroup-3-dilation_is-output_padding_is-padding_is-stride_is-transposed_i", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto transposed = bool(node->i(Symbol::attr("transposed")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_convolution_nogroup");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::_convolution_nogroup(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), stride, padding, dilation, transposed, output_padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_convolution_nogroup", 3, 1);
  }},
  {"_convolution_nogroup-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_convolution_nogroup");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 3, 8)));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 4, 8)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 5, 8)));
      auto transposed = tensor_as<bool>(std::move(peek(stack, 6, 8)));
      auto output_padding = tensor_as<IntList>(std::move(peek(stack, 7, 8)));
      auto result = at::_convolution_nogroup(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), std::move(peek(stack, 2, 8)), stride, padding, dilation, transposed, output_padding);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "_convolution_nogroup", 8, 1);
  }},
  {"_cos-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cos");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_cos(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_cos", 1, 1);
  }},
  {"_cosh-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cosh");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_cosh(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_cosh", 1, 1);
  }},
  {"_cudnn_rnn_flatten_weight-*", [](Node *node) {
    size_t varargs_length = node->inputs().size();
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cudnn_rnn_flatten_weight");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length));
      auto weight_stride0 = tensor_as<int64_t>(std::move(peek(stack, 0, 7)));
      auto input_size = tensor_as<int64_t>(std::move(peek(stack, 1, 7)));
      auto mode = tensor_as<int64_t>(std::move(peek(stack, 2, 7)));
      auto hidden_size = tensor_as<int64_t>(std::move(peek(stack, 3, 7)));
      auto num_layers = tensor_as<int64_t>(std::move(peek(stack, 4, 7)));
      auto batch_first = tensor_as<bool>(std::move(peek(stack, 5, 7)));
      auto bidirectional = tensor_as<bool>(std::move(peek(stack, 6, 7)));
      auto result = at::_cudnn_rnn_flatten_weight(peekSlice(stack, 0, varargs_length - 7, varargs_length), weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "_cudnn_rnn_flatten_weight", varargs_length, 1);
  }},
  {"_cudnn_rnn_flatten_weight-*-batch_first_i-bidirectional_i-hidden_size_i-input_size_i-mode_i-num_layers_i-weight_stride0_i", [](Node *node) {
    size_t varargs_length = node->inputs().size();
    auto weight_stride0 = int64_t(node->i(Symbol::attr("weight_stride0")));
    auto input_size = int64_t(node->i(Symbol::attr("input_size")));
    auto mode = int64_t(node->i(Symbol::attr("mode")));
    auto hidden_size = int64_t(node->i(Symbol::attr("hidden_size")));
    auto num_layers = int64_t(node->i(Symbol::attr("num_layers")));
    auto batch_first = bool(node->i(Symbol::attr("batch_first")));
    auto bidirectional = bool(node->i(Symbol::attr("bidirectional")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cudnn_rnn_flatten_weight");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length));
  
      auto result = at::_cudnn_rnn_flatten_weight(peekSlice(stack, 0, varargs_length - 0, varargs_length), weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "_cudnn_rnn_flatten_weight", varargs_length, 1);
  }},
  {"_cumprod-1-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cumprod");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_cumprod(std::move(peek(stack, 0, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_cumprod", 1, 1);
  }},
  {"_cumprod-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cumprod");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::_cumprod(std::move(peek(stack, 0, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_cumprod", 2, 1);
  }},
  {"_cumsum-1-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cumsum");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_cumsum(std::move(peek(stack, 0, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_cumsum", 1, 1);
  }},
  {"_cumsum-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_cumsum");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::_cumsum(std::move(peek(stack, 0, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_cumsum", 2, 1);
  }},
  {"_dimI-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_dimI");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1)))._dimI();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_dimI", 1, 1);
  }},
  {"_dimV-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_dimV");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1)))._dimV();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_dimV", 1, 1);
  }},
  {"_dirichlet_grad-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_dirichlet_grad");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::_dirichlet_grad(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_dirichlet_grad", 3, 1);
  }},
  {"_dot-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_dot");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::_dot(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_dot", 2, 1);
  }},
  {"_erf-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_erf");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_erf(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_erf", 1, 1);
  }},
  {"_exp-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_exp");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_exp(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_exp", 1, 1);
  }},
  {"_expm1-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_expm1");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_expm1(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_expm1", 1, 1);
  }},
  {"_fft_with_size-1-checked_signal_sizes_is-complex_input_i-complex_output_i-inverse_i-normalized_i-onesided_i-output_sizes_is-signal_ndim_i", [](Node *node) {
    auto signal_ndim = int64_t(node->i(Symbol::attr("signal_ndim")));
    auto complex_input = bool(node->i(Symbol::attr("complex_input")));
    auto complex_output = bool(node->i(Symbol::attr("complex_output")));
    auto inverse = bool(node->i(Symbol::attr("inverse")));
    auto checked_signal_sizes = std::vector<int64_t>(node->is(Symbol::attr("checked_signal_sizes")));
    auto normalized = bool(node->i(Symbol::attr("normalized")));
    auto onesided = bool(node->i(Symbol::attr("onesided")));
    auto output_sizes = std::vector<int64_t>(node->is(Symbol::attr("output_sizes")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_fft_with_size");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_fft_with_size(std::move(peek(stack, 0, 1)), signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_fft_with_size", 1, 1);
  }},
  {"_fft_with_size-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_fft_with_size");
      AutoGPU device_guard(deviceForInputs(stack, 9));
      auto signal_ndim = tensor_as<int64_t>(std::move(peek(stack, 1, 9)));
      auto complex_input = tensor_as<bool>(std::move(peek(stack, 2, 9)));
      auto complex_output = tensor_as<bool>(std::move(peek(stack, 3, 9)));
      auto inverse = tensor_as<bool>(std::move(peek(stack, 4, 9)));
      auto checked_signal_sizes = tensor_as<IntList>(std::move(peek(stack, 5, 9)));
      auto normalized = tensor_as<bool>(std::move(peek(stack, 6, 9)));
      auto onesided = tensor_as<bool>(std::move(peek(stack, 7, 9)));
      auto output_sizes = tensor_as<IntList>(std::move(peek(stack, 8, 9)));
      auto result = at::_fft_with_size(std::move(peek(stack, 0, 9)), signal_ndim, complex_input, complex_output, inverse, checked_signal_sizes, normalized, onesided, output_sizes);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "_fft_with_size", 9, 1);
  }},
  {"_floor-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_floor");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_floor(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_floor", 1, 1);
  }},
  {"_ger-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_ger");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::_ger(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_ger", 2, 1);
  }},
  {"_gesv_helper-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_gesv_helper");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::_gesv_helper(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_gesv_helper", 2, 2);
  }},
  {"_gesv_single-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_gesv_single");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::_gesv_single(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_gesv_single", 2, 2);
  }},
  {"_indices-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_indices");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1)))._indices();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_indices", 1, 1);
  }},
  {"_log-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_log");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_log(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_log", 1, 1);
  }},
  {"_log10-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_log10");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_log10(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_log10", 1, 1);
  }},
  {"_log1p-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_log1p");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_log1p(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_log1p", 1, 1);
  }},
  {"_log2-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_log2");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_log2(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_log2", 1, 1);
  }},
  {"_mm-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_mm");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::_mm(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_mm", 2, 1);
  }},
  {"_mv-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_mv");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::_mv(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_mv", 2, 1);
  }},
  {"_nnz-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_nnz");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1)))._nnz();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_nnz", 1, 1);
  }},
  {"_prod-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_prod");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_prod(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_prod", 1, 1);
  }},
  {"_prod-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_prod");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_prod(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_prod", 1, 1);
  }},
  {"_prod-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_prod");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::_prod(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_prod", 3, 1);
  }},
  {"_prodall-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_prodall");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_prodall(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_prodall", 1, 1);
  }},
  {"_round-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_round");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_round(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_round", 1, 1);
  }},
  {"_rsqrt-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_rsqrt");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_rsqrt(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_rsqrt", 1, 1);
  }},
  {"_s_where-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_s_where");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::_s_where(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_s_where", 3, 1);
  }},
  {"_sigmoid-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sigmoid");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_sigmoid(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_sigmoid", 1, 1);
  }},
  {"_sigmoid_backward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sigmoid_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::_sigmoid_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_sigmoid_backward", 2, 1);
  }},
  {"_sigmoid_forward-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sigmoid_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_sigmoid_forward(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_sigmoid_forward", 1, 1);
  }},
  {"_sin-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sin");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_sin(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_sin", 1, 1);
  }},
  {"_sinh-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sinh");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_sinh(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_sinh", 1, 1);
  }},
  {"_sparse_coo_tensor_unsafe-2-size_is", [](Node *node) {
    auto size = std::vector<int64_t>(node->is(Symbol::attr("size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sparse_coo_tensor_unsafe");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::_sparse_coo_tensor_unsafe(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_sparse_coo_tensor_unsafe", 2, 1);
  }},
  {"_sparse_coo_tensor_unsafe-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sparse_coo_tensor_unsafe");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto size = tensor_as<IntList>(std::move(peek(stack, 2, 3)));
      auto result = at::_sparse_coo_tensor_unsafe(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), size);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_sparse_coo_tensor_unsafe", 3, 1);
  }},
  {"_sqrt-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sqrt");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_sqrt(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_sqrt", 1, 1);
  }},
  {"_standard_gamma_grad-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_standard_gamma_grad");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::_standard_gamma_grad(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_standard_gamma_grad", 2, 1);
  }},
  {"_sum-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sum");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_sum(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_sum", 1, 1);
  }},
  {"_sum-1-dim_is-keepdim_i", [](Node *node) {
    auto dim = std::vector<int64_t>(node->is(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sum");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_sum(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_sum", 1, 1);
  }},
  {"_sum-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sum");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim_tensor = peek(stack, 1, 3);
      if (dim_tensor.dim() == 0)
          dim_tensor = dim_tensor.expand(1);
      auto dim = tensor_as<at::IntList>(std::move(dim_tensor));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::_sum(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_sum", 3, 1);
  }},
  {"_sumall-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_sumall");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_sumall(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_sumall", 1, 1);
  }},
  {"_tan-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_tan");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_tan(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_tan", 1, 1);
  }},
  {"_tanh-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_tanh");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_tanh(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_tanh", 1, 1);
  }},
  {"_tanh_backward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_tanh_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::_tanh_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_tanh_backward", 2, 1);
  }},
  {"_tanh_forward-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_tanh_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_tanh_forward(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_tanh_forward", 1, 1);
  }},
  {"_th_prod-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_th_prod");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_th_prod(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_th_prod", 1, 1);
  }},
  {"_th_prod-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_th_prod");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::_th_prod(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_th_prod", 3, 1);
  }},
  {"_th_sum-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_th_sum");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_th_sum(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_th_sum", 1, 1);
  }},
  {"_th_sum-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_th_sum");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::_th_sum(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_th_sum", 3, 1);
  }},
  {"_th_tanh-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_th_tanh");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_th_tanh(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_th_tanh", 1, 1);
  }},
  {"_trilinear-3-expand1_is-expand2_is-expand3_is-sumdim_is-unroll_dim_i", [](Node *node) {
    auto expand1 = std::vector<int64_t>(node->is(Symbol::attr("expand1")));
    auto expand2 = std::vector<int64_t>(node->is(Symbol::attr("expand2")));
    auto expand3 = std::vector<int64_t>(node->is(Symbol::attr("expand3")));
    auto sumdim = std::vector<int64_t>(node->is(Symbol::attr("sumdim")));
    auto unroll_dim = int64_t(node->i(Symbol::attr("unroll_dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_trilinear");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::_trilinear(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), expand1, expand2, expand3, sumdim, unroll_dim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_trilinear", 3, 1);
  }},
  {"_trilinear-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_trilinear");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto expand1 = tensor_as<IntList>(std::move(peek(stack, 3, 8)));
      auto expand2 = tensor_as<IntList>(std::move(peek(stack, 4, 8)));
      auto expand3 = tensor_as<IntList>(std::move(peek(stack, 5, 8)));
      auto sumdim = tensor_as<IntList>(std::move(peek(stack, 6, 8)));
      auto unroll_dim = tensor_as<int64_t>(std::move(peek(stack, 7, 8)));
      auto result = at::_trilinear(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), std::move(peek(stack, 2, 8)), expand1, expand2, expand3, sumdim, unroll_dim);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "_trilinear", 8, 1);
  }},
  {"_trunc-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_trunc");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_trunc(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_trunc", 1, 1);
  }},
  {"_unique-1-return_inverse_i-sorted_i", [](Node *node) {
    auto sorted = bool(node->i(Symbol::attr("sorted")));
    auto return_inverse = bool(node->i(Symbol::attr("return_inverse")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_unique");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_unique(std::move(peek(stack, 0, 1)), sorted, return_inverse);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_unique", 1, 2);
  }},
  {"_unique-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_unique");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto sorted = tensor_as<bool>(std::move(peek(stack, 1, 3)));
      auto return_inverse = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::_unique(std::move(peek(stack, 0, 3)), sorted, return_inverse);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "_unique", 3, 2);
  }},
  {"_unsafe_view-1-size_is", [](Node *node) {
    auto size = std::vector<int64_t>(node->is(Symbol::attr("size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_unsafe_view");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::_unsafe_view(std::move(peek(stack, 0, 1)), size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_unsafe_view", 1, 1);
  }},
  {"_unsafe_view-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_unsafe_view");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto size = tensor_as<IntList>(std::move(peek(stack, 1, 2)));
      auto result = at::_unsafe_view(std::move(peek(stack, 0, 2)), size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "_unsafe_view", 2, 1);
  }},
  {"_values-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("_values");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1)))._values();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "_values", 1, 1);
  }},
  {"abs-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("abs");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::abs(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "abs", 1, 1);
  }},
  {"acos-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("acos");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::acos(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "acos", 1, 1);
  }},
  {"adaptive_avg_pool1d-1-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool1d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::adaptive_avg_pool1d(std::move(peek(stack, 0, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool1d", 1, 1);
  }},
  {"adaptive_avg_pool1d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool1d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto output_size_tensor = peek(stack, 1, 2);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(1);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::adaptive_avg_pool1d(std::move(peek(stack, 0, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool1d", 2, 1);
  }},
  {"adaptive_avg_pool2d-1-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::adaptive_avg_pool2d(std::move(peek(stack, 0, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool2d", 1, 1);
  }},
  {"adaptive_avg_pool2d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto output_size_tensor = peek(stack, 1, 2);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(2);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::adaptive_avg_pool2d(std::move(peek(stack, 0, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool2d", 2, 1);
  }},
  {"adaptive_avg_pool2d_backward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::adaptive_avg_pool2d_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool2d_backward", 2, 1);
  }},
  {"adaptive_avg_pool2d_forward-1-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::adaptive_avg_pool2d_forward(std::move(peek(stack, 0, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool2d_forward", 1, 1);
  }},
  {"adaptive_avg_pool2d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto output_size_tensor = peek(stack, 1, 2);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(2);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::adaptive_avg_pool2d_forward(std::move(peek(stack, 0, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool2d_forward", 2, 1);
  }},
  {"adaptive_avg_pool3d-1-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::adaptive_avg_pool3d(std::move(peek(stack, 0, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool3d", 1, 1);
  }},
  {"adaptive_avg_pool3d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto output_size_tensor = peek(stack, 1, 2);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(3);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::adaptive_avg_pool3d(std::move(peek(stack, 0, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool3d", 2, 1);
  }},
  {"adaptive_avg_pool3d_backward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::adaptive_avg_pool3d_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool3d_backward", 2, 1);
  }},
  {"adaptive_avg_pool3d_forward-1-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::adaptive_avg_pool3d_forward(std::move(peek(stack, 0, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool3d_forward", 1, 1);
  }},
  {"adaptive_avg_pool3d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_avg_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto output_size_tensor = peek(stack, 1, 2);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(3);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::adaptive_avg_pool3d_forward(std::move(peek(stack, 0, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_avg_pool3d_forward", 2, 1);
  }},
  {"adaptive_max_pool1d-1-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool1d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::adaptive_max_pool1d(std::move(peek(stack, 0, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool1d", 1, 2);
  }},
  {"adaptive_max_pool1d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool1d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto output_size_tensor = peek(stack, 1, 2);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(1);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::adaptive_max_pool1d(std::move(peek(stack, 0, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool1d", 2, 2);
  }},
  {"adaptive_max_pool2d-1-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::adaptive_max_pool2d(std::move(peek(stack, 0, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool2d", 1, 2);
  }},
  {"adaptive_max_pool2d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto output_size_tensor = peek(stack, 1, 2);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(2);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::adaptive_max_pool2d(std::move(peek(stack, 0, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool2d", 2, 2);
  }},
  {"adaptive_max_pool2d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::adaptive_max_pool2d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool2d_backward", 3, 1);
  }},
  {"adaptive_max_pool2d_forward-1-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::adaptive_max_pool2d_forward(std::move(peek(stack, 0, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool2d_forward", 1, 2);
  }},
  {"adaptive_max_pool2d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto output_size_tensor = peek(stack, 1, 2);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(2);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::adaptive_max_pool2d_forward(std::move(peek(stack, 0, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool2d_forward", 2, 2);
  }},
  {"adaptive_max_pool3d-1-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::adaptive_max_pool3d(std::move(peek(stack, 0, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool3d", 1, 2);
  }},
  {"adaptive_max_pool3d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto output_size_tensor = peek(stack, 1, 2);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(3);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::adaptive_max_pool3d(std::move(peek(stack, 0, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool3d", 2, 2);
  }},
  {"adaptive_max_pool3d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::adaptive_max_pool3d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool3d_backward", 3, 1);
  }},
  {"adaptive_max_pool3d_forward-1-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::adaptive_max_pool3d_forward(std::move(peek(stack, 0, 1)), output_size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool3d_forward", 1, 2);
  }},
  {"adaptive_max_pool3d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("adaptive_max_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto output_size_tensor = peek(stack, 1, 2);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(3);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::adaptive_max_pool3d_forward(std::move(peek(stack, 0, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "adaptive_max_pool3d_forward", 2, 2);
  }},
  {"add-1-alpha_t-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("add");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::add(std::move(peek(stack, 0, 1)), other, alpha);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "add", 1, 1);
  }},
  {"add-2-alpha_t", [](Node *node) {
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("add");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::add(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), alpha);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "add", 2, 1);
  }},
  {"add-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("add");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto alpha = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::add(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "add", 3, 1);
  }},
  {"addbmm-3-alpha_t-beta_t", [](Node *node) {
    auto beta = Scalar(node->t(Symbol::attr("beta")));
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addbmm");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::addbmm(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "addbmm", 3, 1);
  }},
  {"addbmm-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addbmm");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto beta = tensor_as<Scalar>(std::move(peek(stack, 3, 5)));
      auto alpha = tensor_as<Scalar>(std::move(peek(stack, 4, 5)));
      auto result = at::addbmm(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "addbmm", 5, 1);
  }},
  {"addcdiv-3-value_t", [](Node *node) {
    auto value = Scalar(node->t(Symbol::attr("value")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addcdiv");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::addcdiv(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), value);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "addcdiv", 3, 1);
  }},
  {"addcdiv-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addcdiv");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto value = tensor_as<Scalar>(std::move(peek(stack, 3, 4)));
      auto result = at::addcdiv(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), std::move(peek(stack, 2, 4)), value);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "addcdiv", 4, 1);
  }},
  {"addcmul-3-value_t", [](Node *node) {
    auto value = Scalar(node->t(Symbol::attr("value")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addcmul");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::addcmul(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), value);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "addcmul", 3, 1);
  }},
  {"addcmul-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addcmul");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto value = tensor_as<Scalar>(std::move(peek(stack, 3, 4)));
      auto result = at::addcmul(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), std::move(peek(stack, 2, 4)), value);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "addcmul", 4, 1);
  }},
  {"addmm-3-alpha_t-beta_t", [](Node *node) {
    auto beta = Scalar(node->t(Symbol::attr("beta")));
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addmm");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::addmm(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "addmm", 3, 1);
  }},
  {"addmm-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addmm");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto beta = tensor_as<Scalar>(std::move(peek(stack, 3, 5)));
      auto alpha = tensor_as<Scalar>(std::move(peek(stack, 4, 5)));
      auto result = at::addmm(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "addmm", 5, 1);
  }},
  {"addmv-3-alpha_t-beta_t", [](Node *node) {
    auto beta = Scalar(node->t(Symbol::attr("beta")));
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addmv");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::addmv(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "addmv", 3, 1);
  }},
  {"addmv-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addmv");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto beta = tensor_as<Scalar>(std::move(peek(stack, 3, 5)));
      auto alpha = tensor_as<Scalar>(std::move(peek(stack, 4, 5)));
      auto result = at::addmv(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "addmv", 5, 1);
  }},
  {"addr-3-alpha_t-beta_t", [](Node *node) {
    auto beta = Scalar(node->t(Symbol::attr("beta")));
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addr");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::addr(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "addr", 3, 1);
  }},
  {"addr-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("addr");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto beta = tensor_as<Scalar>(std::move(peek(stack, 3, 5)));
      auto alpha = tensor_as<Scalar>(std::move(peek(stack, 4, 5)));
      auto result = at::addr(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "addr", 5, 1);
  }},
  {"alias-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("alias");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::alias(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "alias", 1, 1);
  }},
  {"all-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("all");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::all(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "all", 1, 1);
  }},
  {"all-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("all");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::all(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "all", 1, 1);
  }},
  {"all-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("all");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::all(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "all", 3, 1);
  }},
  {"allclose-2-atol_f-equal_nan_i-rtol_f", [](Node *node) {
    auto rtol = double(node->f(Symbol::attr("rtol")));
    auto atol = double(node->f(Symbol::attr("atol")));
    auto equal_nan = bool(node->i(Symbol::attr("equal_nan")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("allclose");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::allclose(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), rtol, atol, equal_nan);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "allclose", 2, 1);
  }},
  {"allclose-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("allclose");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto rtol = tensor_as<double>(std::move(peek(stack, 2, 5)));
      auto atol = tensor_as<double>(std::move(peek(stack, 3, 5)));
      auto equal_nan = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::allclose(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), rtol, atol, equal_nan);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "allclose", 5, 1);
  }},
  {"any-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("any");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::any(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "any", 1, 1);
  }},
  {"any-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("any");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::any(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "any", 1, 1);
  }},
  {"any-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("any");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::any(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "any", 3, 1);
  }},
  {"argmax-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("argmax");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::argmax(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "argmax", 1, 1);
  }},
  {"argmax-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("argmax");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::argmax(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "argmax", 1, 1);
  }},
  {"argmax-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("argmax");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::argmax(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "argmax", 3, 1);
  }},
  {"argmin-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("argmin");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::argmin(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "argmin", 1, 1);
  }},
  {"argmin-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("argmin");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::argmin(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "argmin", 1, 1);
  }},
  {"argmin-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("argmin");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::argmin(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "argmin", 3, 1);
  }},
  {"as_strided-1-size_is-storage_offset_i-stride_is", [](Node *node) {
    auto size = std::vector<int64_t>(node->is(Symbol::attr("size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto storage_offset = int64_t(node->i(Symbol::attr("storage_offset")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("as_strided");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::as_strided(std::move(peek(stack, 0, 1)), size, stride, storage_offset);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "as_strided", 1, 1);
  }},
  {"as_strided-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("as_strided");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto size = tensor_as<IntList>(std::move(peek(stack, 1, 4)));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 2, 4)));
      auto storage_offset = tensor_as<int64_t>(std::move(peek(stack, 3, 4)));
      auto result = at::as_strided(std::move(peek(stack, 0, 4)), size, stride, storage_offset);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "as_strided", 4, 1);
  }},
  {"asin-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("asin");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::asin(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "asin", 1, 1);
  }},
  {"atan-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("atan");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::atan(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "atan", 1, 1);
  }},
  {"atan2-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("atan2");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::atan2(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "atan2", 2, 1);
  }},
  {"avg_pool2d-1-ceil_mode_i-count_include_pad_i-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto ceil_mode = bool(node->i(Symbol::attr("ceil_mode")));
    auto count_include_pad = bool(node->i(Symbol::attr("count_include_pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::avg_pool2d(std::move(peek(stack, 0, 1)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool2d", 1, 1);
  }},
  {"avg_pool2d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto kernel_size_tensor = peek(stack, 1, 6);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 2, 6);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 3, 6);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto ceil_mode = tensor_as<bool>(std::move(peek(stack, 4, 6)));
      auto count_include_pad = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::avg_pool2d(std::move(peek(stack, 0, 6)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool2d", 6, 1);
  }},
  {"avg_pool2d_backward-2-ceil_mode_i-count_include_pad_i-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto ceil_mode = bool(node->i(Symbol::attr("ceil_mode")));
    auto count_include_pad = bool(node->i(Symbol::attr("count_include_pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::avg_pool2d_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool2d_backward", 2, 1);
  }},
  {"avg_pool2d_backward-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto kernel_size_tensor = peek(stack, 2, 7);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 3, 7);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 4, 7);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto ceil_mode = tensor_as<bool>(std::move(peek(stack, 5, 7)));
      auto count_include_pad = tensor_as<bool>(std::move(peek(stack, 6, 7)));
      auto result = at::avg_pool2d_backward(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool2d_backward", 7, 1);
  }},
  {"avg_pool2d_forward-1-ceil_mode_i-count_include_pad_i-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto ceil_mode = bool(node->i(Symbol::attr("ceil_mode")));
    auto count_include_pad = bool(node->i(Symbol::attr("count_include_pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::avg_pool2d_forward(std::move(peek(stack, 0, 1)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool2d_forward", 1, 1);
  }},
  {"avg_pool2d_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto kernel_size_tensor = peek(stack, 1, 6);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 2, 6);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 3, 6);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto ceil_mode = tensor_as<bool>(std::move(peek(stack, 4, 6)));
      auto count_include_pad = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::avg_pool2d_forward(std::move(peek(stack, 0, 6)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool2d_forward", 6, 1);
  }},
  {"avg_pool3d-1-ceil_mode_i-count_include_pad_i-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto ceil_mode = bool(node->i(Symbol::attr("ceil_mode")));
    auto count_include_pad = bool(node->i(Symbol::attr("count_include_pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::avg_pool3d(std::move(peek(stack, 0, 1)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool3d", 1, 1);
  }},
  {"avg_pool3d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto kernel_size_tensor = peek(stack, 1, 6);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 2, 6);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 3, 6);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto ceil_mode = tensor_as<bool>(std::move(peek(stack, 4, 6)));
      auto count_include_pad = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::avg_pool3d(std::move(peek(stack, 0, 6)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool3d", 6, 1);
  }},
  {"avg_pool3d_backward-2-ceil_mode_i-count_include_pad_i-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto ceil_mode = bool(node->i(Symbol::attr("ceil_mode")));
    auto count_include_pad = bool(node->i(Symbol::attr("count_include_pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::avg_pool3d_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool3d_backward", 2, 1);
  }},
  {"avg_pool3d_backward-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto kernel_size_tensor = peek(stack, 2, 7);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 3, 7);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 4, 7);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto ceil_mode = tensor_as<bool>(std::move(peek(stack, 5, 7)));
      auto count_include_pad = tensor_as<bool>(std::move(peek(stack, 6, 7)));
      auto result = at::avg_pool3d_backward(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool3d_backward", 7, 1);
  }},
  {"avg_pool3d_forward-1-ceil_mode_i-count_include_pad_i-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto ceil_mode = bool(node->i(Symbol::attr("ceil_mode")));
    auto count_include_pad = bool(node->i(Symbol::attr("count_include_pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::avg_pool3d_forward(std::move(peek(stack, 0, 1)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool3d_forward", 1, 1);
  }},
  {"avg_pool3d_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("avg_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto kernel_size_tensor = peek(stack, 1, 6);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 2, 6);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 3, 6);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto ceil_mode = tensor_as<bool>(std::move(peek(stack, 4, 6)));
      auto count_include_pad = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::avg_pool3d_forward(std::move(peek(stack, 0, 6)), kernel_size, stride, padding, ceil_mode, count_include_pad);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "avg_pool3d_forward", 6, 1);
  }},
  {"baddbmm-3-alpha_t-beta_t", [](Node *node) {
    auto beta = Scalar(node->t(Symbol::attr("beta")));
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("baddbmm");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::baddbmm(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "baddbmm", 3, 1);
  }},
  {"baddbmm-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("baddbmm");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto beta = tensor_as<Scalar>(std::move(peek(stack, 3, 5)));
      auto alpha = tensor_as<Scalar>(std::move(peek(stack, 4, 5)));
      auto result = at::baddbmm(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "baddbmm", 5, 1);
  }},
  {"batch_norm-5-cudnn_enabled_i-eps_f-momentum_f-training_i", [](Node *node) {
    auto training = bool(node->i(Symbol::attr("training")));
    auto momentum = double(node->f(Symbol::attr("momentum")));
    auto eps = double(node->f(Symbol::attr("eps")));
    auto cudnn_enabled = bool(node->i(Symbol::attr("cudnn_enabled")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("batch_norm");
      AutoGPU device_guard(deviceForInputs(stack, 5));
  
      auto result = at::batch_norm(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), std::move(peek(stack, 3, 5)), std::move(peek(stack, 4, 5)), training, momentum, eps, cudnn_enabled);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "batch_norm", 5, 1);
  }},
  {"batch_norm-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("batch_norm");
      AutoGPU device_guard(deviceForInputs(stack, 9));
      auto training = tensor_as<bool>(std::move(peek(stack, 5, 9)));
      auto momentum = tensor_as<double>(std::move(peek(stack, 6, 9)));
      auto eps = tensor_as<double>(std::move(peek(stack, 7, 9)));
      auto cudnn_enabled = tensor_as<bool>(std::move(peek(stack, 8, 9)));
      auto result = at::batch_norm(std::move(peek(stack, 0, 9)), std::move(peek(stack, 1, 9)), std::move(peek(stack, 2, 9)), std::move(peek(stack, 3, 9)), std::move(peek(stack, 4, 9)), training, momentum, eps, cudnn_enabled);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "batch_norm", 9, 1);
  }},
  {"bernoulli-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("bernoulli");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::bernoulli(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "bernoulli", 1, 1);
  }},
  {"bilinear-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("bilinear");
      AutoGPU device_guard(deviceForInputs(stack, 4));
  
      auto result = at::bilinear(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), std::move(peek(stack, 2, 4)), std::move(peek(stack, 3, 4)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "bilinear", 4, 1);
  }},
  {"binary_cross_entropy-3-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("binary_cross_entropy");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::binary_cross_entropy(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "binary_cross_entropy", 3, 1);
  }},
  {"binary_cross_entropy-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("binary_cross_entropy");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 3, 5)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::binary_cross_entropy(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "binary_cross_entropy", 5, 1);
  }},
  {"binary_cross_entropy_backward-4-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("binary_cross_entropy_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
  
      auto result = at::binary_cross_entropy_backward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), std::move(peek(stack, 2, 4)), std::move(peek(stack, 3, 4)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "binary_cross_entropy_backward", 4, 1);
  }},
  {"binary_cross_entropy_backward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("binary_cross_entropy_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 4, 6)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::binary_cross_entropy_backward(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), std::move(peek(stack, 3, 6)), size_average, reduce);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "binary_cross_entropy_backward", 6, 1);
  }},
  {"binary_cross_entropy_forward-3-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("binary_cross_entropy_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::binary_cross_entropy_forward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "binary_cross_entropy_forward", 3, 1);
  }},
  {"binary_cross_entropy_forward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("binary_cross_entropy_forward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 3, 5)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::binary_cross_entropy_forward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "binary_cross_entropy_forward", 5, 1);
  }},
  {"bmm-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("bmm");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::bmm(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "bmm", 2, 1);
  }},
  {"btrifact-1-pivot_i", [](Node *node) {
    auto pivot = bool(node->i(Symbol::attr("pivot")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("btrifact");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::btrifact(std::move(peek(stack, 0, 1)), pivot);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "btrifact", 1, 2);
  }},
  {"btrifact-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("btrifact");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto pivot = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::btrifact(std::move(peek(stack, 0, 2)), pivot);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "btrifact", 2, 2);
  }},
  {"btrifact_with_info-1-pivot_i", [](Node *node) {
    auto pivot = bool(node->i(Symbol::attr("pivot")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("btrifact_with_info");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::btrifact_with_info(std::move(peek(stack, 0, 1)), pivot);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "btrifact_with_info", 1, 3);
  }},
  {"btrifact_with_info-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("btrifact_with_info");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto pivot = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::btrifact_with_info(std::move(peek(stack, 0, 2)), pivot);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "btrifact_with_info", 2, 3);
  }},
  {"btrisolve-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("btrisolve");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::btrisolve(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "btrisolve", 3, 1);
  }},
  {"cat-*", [](Node *node) {
    size_t varargs_length = node->inputs().size();
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cat");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 0, 1)));
      auto result = at::cat(peekSlice(stack, 0, varargs_length - 1, varargs_length), dim);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "cat", varargs_length, 1);
  }},
  {"cat-*-dim_i", [](Node *node) {
    size_t varargs_length = node->inputs().size();
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cat");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length));
  
      auto result = at::cat(peekSlice(stack, 0, varargs_length - 0, varargs_length), dim);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "cat", varargs_length, 1);
  }},
  {"ceil-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ceil");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::ceil(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "ceil", 1, 1);
  }},
  {"chunk-1-chunks_i-dim_i", [](Node *node) {
    auto chunks = int64_t(node->i(Symbol::attr("chunks")));
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("chunk");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::chunk(std::move(peek(stack, 0, 1)), chunks, dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "chunk", 1, UNKNOWN_OUTPUTS);
  }},
  {"chunk-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("chunk");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto chunks = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 2, 3)));
      auto result = at::chunk(std::move(peek(stack, 0, 3)), chunks, dim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "chunk", 3, UNKNOWN_OUTPUTS);
  }},
  {"clamp-1-max_t-min_t", [](Node *node) {
    auto min = Scalar(node->t(Symbol::attr("min")));
    auto max = Scalar(node->t(Symbol::attr("max")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("clamp");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::clamp(std::move(peek(stack, 0, 1)), min, max);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "clamp", 1, 1);
  }},
  {"clamp-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("clamp");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto min = tensor_as<Scalar>(std::move(peek(stack, 1, 3)));
      auto max = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::clamp(std::move(peek(stack, 0, 3)), min, max);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "clamp", 3, 1);
  }},
  {"clamp_max-1-max_t", [](Node *node) {
    auto max = Scalar(node->t(Symbol::attr("max")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("clamp_max");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::clamp_max(std::move(peek(stack, 0, 1)), max);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "clamp_max", 1, 1);
  }},
  {"clamp_max-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("clamp_max");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto max = tensor_as<Scalar>(std::move(peek(stack, 1, 2)));
      auto result = at::clamp_max(std::move(peek(stack, 0, 2)), max);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "clamp_max", 2, 1);
  }},
  {"clamp_min-1-min_t", [](Node *node) {
    auto min = Scalar(node->t(Symbol::attr("min")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("clamp_min");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::clamp_min(std::move(peek(stack, 0, 1)), min);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "clamp_min", 1, 1);
  }},
  {"clamp_min-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("clamp_min");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto min = tensor_as<Scalar>(std::move(peek(stack, 1, 2)));
      auto result = at::clamp_min(std::move(peek(stack, 0, 2)), min);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "clamp_min", 2, 1);
  }},
  {"clone-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("clone");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).clone();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "clone", 1, 1);
  }},
  {"coalesce-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("coalesce");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).coalesce();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "coalesce", 1, 1);
  }},
  {"contiguous-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("contiguous");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).contiguous();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "contiguous", 1, 1);
  }},
  {"conv1d-3-dilation_is-groups_i-padding_is-stride_is", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv1d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::conv1d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), stride, padding, dilation, groups);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "conv1d", 3, 1);
  }},
  {"conv1d-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv1d");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto stride_tensor = peek(stack, 3, 7);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(1);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 4, 7);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(1);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 5, 7);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(1);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 6, 7)));
      auto result = at::conv1d(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), std::move(peek(stack, 2, 7)), stride, padding, dilation, groups);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "conv1d", 7, 1);
  }},
  {"conv2d-3-dilation_is-groups_i-padding_is-stride_is", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv2d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::conv2d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), stride, padding, dilation, groups);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "conv2d", 3, 1);
  }},
  {"conv2d-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv2d");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto stride_tensor = peek(stack, 3, 7);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 4, 7);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 5, 7);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(2);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 6, 7)));
      auto result = at::conv2d(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), std::move(peek(stack, 2, 7)), stride, padding, dilation, groups);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "conv2d", 7, 1);
  }},
  {"conv3d-3-dilation_is-groups_i-padding_is-stride_is", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv3d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::conv3d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), stride, padding, dilation, groups);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "conv3d", 3, 1);
  }},
  {"conv3d-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv3d");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto stride_tensor = peek(stack, 3, 7);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 4, 7);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 5, 7);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(3);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 6, 7)));
      auto result = at::conv3d(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), std::move(peek(stack, 2, 7)), stride, padding, dilation, groups);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "conv3d", 7, 1);
  }},
  {"conv_tbc-3-pad_i", [](Node *node) {
    auto pad = int64_t(node->i(Symbol::attr("pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_tbc");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::conv_tbc(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), pad);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "conv_tbc", 3, 1);
  }},
  {"conv_tbc-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_tbc");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto pad = tensor_as<int64_t>(std::move(peek(stack, 3, 4)));
      auto result = at::conv_tbc(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), std::move(peek(stack, 2, 4)), pad);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "conv_tbc", 4, 1);
  }},
  {"conv_tbc_backward-4-pad_i", [](Node *node) {
    auto pad = int64_t(node->i(Symbol::attr("pad")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_tbc_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
  
      auto result = at::conv_tbc_backward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), std::move(peek(stack, 2, 4)), std::move(peek(stack, 3, 4)), pad);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "conv_tbc_backward", 4, 3);
  }},
  {"conv_tbc_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_tbc_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto pad = tensor_as<int64_t>(std::move(peek(stack, 4, 5)));
      auto result = at::conv_tbc_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), std::move(peek(stack, 3, 5)), pad);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "conv_tbc_backward", 5, 3);
  }},
  {"conv_transpose1d-3-dilation_is-groups_i-output_padding_is-padding_is-stride_is", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_transpose1d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::conv_transpose1d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), stride, padding, output_padding, groups, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "conv_transpose1d", 3, 1);
  }},
  {"conv_transpose1d-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_transpose1d");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto stride_tensor = peek(stack, 3, 8);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(1);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 4, 8);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(1);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto output_padding_tensor = peek(stack, 5, 8);
      if (output_padding_tensor.dim() == 0)
          output_padding_tensor = output_padding_tensor.expand(1);
      auto output_padding = tensor_as<at::IntList>(std::move(output_padding_tensor));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 6, 8)));
      auto dilation_tensor = peek(stack, 7, 8);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(1);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto result = at::conv_transpose1d(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), std::move(peek(stack, 2, 8)), stride, padding, output_padding, groups, dilation);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "conv_transpose1d", 8, 1);
  }},
  {"conv_transpose2d-3-dilation_is-groups_i-output_padding_is-padding_is-stride_is", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_transpose2d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::conv_transpose2d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), stride, padding, output_padding, groups, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "conv_transpose2d", 3, 1);
  }},
  {"conv_transpose2d-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_transpose2d");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto stride_tensor = peek(stack, 3, 8);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 4, 8);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto output_padding_tensor = peek(stack, 5, 8);
      if (output_padding_tensor.dim() == 0)
          output_padding_tensor = output_padding_tensor.expand(2);
      auto output_padding = tensor_as<at::IntList>(std::move(output_padding_tensor));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 6, 8)));
      auto dilation_tensor = peek(stack, 7, 8);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(2);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto result = at::conv_transpose2d(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), std::move(peek(stack, 2, 8)), stride, padding, output_padding, groups, dilation);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "conv_transpose2d", 8, 1);
  }},
  {"conv_transpose3d-3-dilation_is-groups_i-output_padding_is-padding_is-stride_is", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_transpose3d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::conv_transpose3d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), stride, padding, output_padding, groups, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "conv_transpose3d", 3, 1);
  }},
  {"conv_transpose3d-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("conv_transpose3d");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto stride_tensor = peek(stack, 3, 8);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 4, 8);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto output_padding_tensor = peek(stack, 5, 8);
      if (output_padding_tensor.dim() == 0)
          output_padding_tensor = output_padding_tensor.expand(3);
      auto output_padding = tensor_as<at::IntList>(std::move(output_padding_tensor));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 6, 8)));
      auto dilation_tensor = peek(stack, 7, 8);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(3);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto result = at::conv_transpose3d(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), std::move(peek(stack, 2, 8)), stride, padding, output_padding, groups, dilation);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "conv_transpose3d", 8, 1);
  }},
  {"convolution-3-dilation_is-groups_i-output_padding_is-padding_is-stride_is-transposed_i", [](Node *node) {
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto transposed = bool(node->i(Symbol::attr("transposed")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("convolution");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::convolution(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), stride, padding, dilation, transposed, output_padding, groups);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "convolution", 3, 1);
  }},
  {"convolution-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("convolution");
      AutoGPU device_guard(deviceForInputs(stack, 9));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 3, 9)));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 4, 9)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 5, 9)));
      auto transposed = tensor_as<bool>(std::move(peek(stack, 6, 9)));
      auto output_padding = tensor_as<IntList>(std::move(peek(stack, 7, 9)));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 8, 9)));
      auto result = at::convolution(std::move(peek(stack, 0, 9)), std::move(peek(stack, 1, 9)), std::move(peek(stack, 2, 9)), stride, padding, dilation, transposed, output_padding, groups);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "convolution", 9, 1);
  }},
  {"cos-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cos");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::cos(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cos", 1, 1);
  }},
  {"cosh-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cosh");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::cosh(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cosh", 1, 1);
  }},
  {"cosine_embedding_loss-3-margin_f-reduce_i-size_average_i", [](Node *node) {
    auto margin = double(node->f(Symbol::attr("margin")));
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cosine_embedding_loss");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::cosine_embedding_loss(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), margin, size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "cosine_embedding_loss", 3, 1);
  }},
  {"cosine_embedding_loss-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cosine_embedding_loss");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto margin = tensor_as<double>(std::move(peek(stack, 3, 6)));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 4, 6)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::cosine_embedding_loss(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), margin, size_average, reduce);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "cosine_embedding_loss", 6, 1);
  }},
  {"cross-2-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cross");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::cross(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cross", 2, 1);
  }},
  {"cross-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cross");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 2, 3)));
      auto result = at::cross(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), dim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "cross", 3, 1);
  }},
  {"cudnn_affine_grid_generator-1-C_i-H_i-N_i-W_i", [](Node *node) {
    auto N = int64_t(node->i(Symbol::attr("N")));
    auto C = int64_t(node->i(Symbol::attr("C")));
    auto H = int64_t(node->i(Symbol::attr("H")));
    auto W = int64_t(node->i(Symbol::attr("W")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_affine_grid_generator");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::cudnn_affine_grid_generator(std::move(peek(stack, 0, 1)), N, C, H, W);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_affine_grid_generator", 1, 1);
  }},
  {"cudnn_affine_grid_generator-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_affine_grid_generator");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto N = tensor_as<int64_t>(std::move(peek(stack, 1, 5)));
      auto C = tensor_as<int64_t>(std::move(peek(stack, 2, 5)));
      auto H = tensor_as<int64_t>(std::move(peek(stack, 3, 5)));
      auto W = tensor_as<int64_t>(std::move(peek(stack, 4, 5)));
      auto result = at::cudnn_affine_grid_generator(std::move(peek(stack, 0, 5)), N, C, H, W);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_affine_grid_generator", 5, 1);
  }},
  {"cudnn_affine_grid_generator_backward-1-C_i-H_i-N_i-W_i", [](Node *node) {
    auto N = int64_t(node->i(Symbol::attr("N")));
    auto C = int64_t(node->i(Symbol::attr("C")));
    auto H = int64_t(node->i(Symbol::attr("H")));
    auto W = int64_t(node->i(Symbol::attr("W")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_affine_grid_generator_backward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::cudnn_affine_grid_generator_backward(std::move(peek(stack, 0, 1)), N, C, H, W);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_affine_grid_generator_backward", 1, 1);
  }},
  {"cudnn_affine_grid_generator_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_affine_grid_generator_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto N = tensor_as<int64_t>(std::move(peek(stack, 1, 5)));
      auto C = tensor_as<int64_t>(std::move(peek(stack, 2, 5)));
      auto H = tensor_as<int64_t>(std::move(peek(stack, 3, 5)));
      auto W = tensor_as<int64_t>(std::move(peek(stack, 4, 5)));
      auto result = at::cudnn_affine_grid_generator_backward(std::move(peek(stack, 0, 5)), N, C, H, W);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_affine_grid_generator_backward", 5, 1);
  }},
  {"cudnn_batch_norm-5-epsilon_f-exponential_average_factor_f-training_i", [](Node *node) {
    auto training = bool(node->i(Symbol::attr("training")));
    auto exponential_average_factor = double(node->f(Symbol::attr("exponential_average_factor")));
    auto epsilon = double(node->f(Symbol::attr("epsilon")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_batch_norm");
      AutoGPU device_guard(deviceForInputs(stack, 5));
  
      auto result = at::cudnn_batch_norm(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), std::move(peek(stack, 3, 5)), std::move(peek(stack, 4, 5)), training, exponential_average_factor, epsilon);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_batch_norm", 5, 3);
  }},
  {"cudnn_batch_norm-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_batch_norm");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto training = tensor_as<bool>(std::move(peek(stack, 5, 8)));
      auto exponential_average_factor = tensor_as<double>(std::move(peek(stack, 6, 8)));
      auto epsilon = tensor_as<double>(std::move(peek(stack, 7, 8)));
      auto result = at::cudnn_batch_norm(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), std::move(peek(stack, 2, 8)), std::move(peek(stack, 3, 8)), std::move(peek(stack, 4, 8)), training, exponential_average_factor, epsilon);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_batch_norm", 8, 3);
  }},
  {"cudnn_batch_norm_backward-7-epsilon_f", [](Node *node) {
    auto epsilon = double(node->f(Symbol::attr("epsilon")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_batch_norm_backward");
      AutoGPU device_guard(deviceForInputs(stack, 7));
  
      auto result = at::cudnn_batch_norm_backward(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), std::move(peek(stack, 2, 7)), std::move(peek(stack, 3, 7)), std::move(peek(stack, 4, 7)), std::move(peek(stack, 5, 7)), std::move(peek(stack, 6, 7)), epsilon);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_batch_norm_backward", 7, 3);
  }},
  {"cudnn_batch_norm_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_batch_norm_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto epsilon = tensor_as<double>(std::move(peek(stack, 7, 8)));
      auto result = at::cudnn_batch_norm_backward(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), std::move(peek(stack, 2, 8)), std::move(peek(stack, 3, 8)), std::move(peek(stack, 4, 8)), std::move(peek(stack, 5, 8)), std::move(peek(stack, 6, 8)), epsilon);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_batch_norm_backward", 8, 3);
  }},
  {"cudnn_convolution-3-benchmark_i-deterministic_i-dilation_is-groups_i-padding_is-stride_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    auto benchmark = bool(node->i(Symbol::attr("benchmark")));
    auto deterministic = bool(node->i(Symbol::attr("deterministic")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::cudnn_convolution(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution", 3, 1);
  }},
  {"cudnn_convolution-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution");
      AutoGPU device_guard(deviceForInputs(stack, 9));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 3, 9)));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 4, 9)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 5, 9)));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 6, 9)));
      auto benchmark = tensor_as<bool>(std::move(peek(stack, 7, 9)));
      auto deterministic = tensor_as<bool>(std::move(peek(stack, 8, 9)));
      auto result = at::cudnn_convolution(std::move(peek(stack, 0, 9)), std::move(peek(stack, 1, 9)), std::move(peek(stack, 2, 9)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution", 9, 1);
  }},
  {"cudnn_convolution_backward-10", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_backward");
      AutoGPU device_guard(deviceForInputs(stack, 10));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 3, 10)));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 4, 10)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 5, 10)));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 6, 10)));
      auto benchmark = tensor_as<bool>(std::move(peek(stack, 7, 10)));
      auto deterministic = tensor_as<bool>(std::move(peek(stack, 8, 10)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(peek(stack, 9, 10)));
      auto result = at::cudnn_convolution_backward(std::move(peek(stack, 0, 10)), std::move(peek(stack, 1, 10)), std::move(peek(stack, 2, 10)), padding, stride, dilation, groups, benchmark, deterministic, output_mask);
      drop(stack, 10);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_backward", 10, 3);
  }},
  {"cudnn_convolution_backward-3-benchmark_i-deterministic_i-dilation_is-groups_i-output_mask_is-padding_is-stride_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    auto benchmark = bool(node->i(Symbol::attr("benchmark")));
    auto deterministic = bool(node->i(Symbol::attr("deterministic")));
    auto output_mask = as_bool_array<3>(node->is(Symbol::attr("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::cudnn_convolution_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), padding, stride, dilation, groups, benchmark, deterministic, output_mask);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_backward", 3, 3);
  }},
  {"cudnn_convolution_backward_bias-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_backward_bias");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::cudnn_convolution_backward_bias(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_backward_bias", 1, 1);
  }},
  {"cudnn_convolution_backward_input-2-benchmark_i-deterministic_i-dilation_is-groups_i-padding_is-self_size_is-stride_is", [](Node *node) {
    auto self_size = std::vector<int64_t>(node->is(Symbol::attr("self_size")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    auto benchmark = bool(node->i(Symbol::attr("benchmark")));
    auto deterministic = bool(node->i(Symbol::attr("deterministic")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_backward_input");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::cudnn_convolution_backward_input(self_size, std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_backward_input", 2, 1);
  }},
  {"cudnn_convolution_backward_input-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_backward_input");
      AutoGPU device_guard(deviceForInputs(stack, 9));
      auto self_size = tensor_as<IntList>(std::move(peek(stack, 0, 9)));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 3, 9)));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 4, 9)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 5, 9)));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 6, 9)));
      auto benchmark = tensor_as<bool>(std::move(peek(stack, 7, 9)));
      auto deterministic = tensor_as<bool>(std::move(peek(stack, 8, 9)));
      auto result = at::cudnn_convolution_backward_input(self_size, std::move(peek(stack, 1, 9)), std::move(peek(stack, 2, 9)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_backward_input", 9, 1);
  }},
  {"cudnn_convolution_backward_weight-2-benchmark_i-deterministic_i-dilation_is-groups_i-padding_is-stride_is-weight_size_is", [](Node *node) {
    auto weight_size = std::vector<int64_t>(node->is(Symbol::attr("weight_size")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    auto benchmark = bool(node->i(Symbol::attr("benchmark")));
    auto deterministic = bool(node->i(Symbol::attr("deterministic")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_backward_weight");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::cudnn_convolution_backward_weight(weight_size, std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_backward_weight", 2, 1);
  }},
  {"cudnn_convolution_backward_weight-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_backward_weight");
      AutoGPU device_guard(deviceForInputs(stack, 9));
      auto weight_size = tensor_as<IntList>(std::move(peek(stack, 0, 9)));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 3, 9)));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 4, 9)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 5, 9)));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 6, 9)));
      auto benchmark = tensor_as<bool>(std::move(peek(stack, 7, 9)));
      auto deterministic = tensor_as<bool>(std::move(peek(stack, 8, 9)));
      auto result = at::cudnn_convolution_backward_weight(weight_size, std::move(peek(stack, 1, 9)), std::move(peek(stack, 2, 9)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_backward_weight", 9, 1);
  }},
  {"cudnn_convolution_transpose-10", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose");
      AutoGPU device_guard(deviceForInputs(stack, 10));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 3, 10)));
      auto output_padding = tensor_as<IntList>(std::move(peek(stack, 4, 10)));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 5, 10)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 6, 10)));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 7, 10)));
      auto benchmark = tensor_as<bool>(std::move(peek(stack, 8, 10)));
      auto deterministic = tensor_as<bool>(std::move(peek(stack, 9, 10)));
      auto result = at::cudnn_convolution_transpose(std::move(peek(stack, 0, 10)), std::move(peek(stack, 1, 10)), std::move(peek(stack, 2, 10)), padding, output_padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 10);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose", 10, 1);
  }},
  {"cudnn_convolution_transpose-3-benchmark_i-deterministic_i-dilation_is-groups_i-output_padding_is-padding_is-stride_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    auto benchmark = bool(node->i(Symbol::attr("benchmark")));
    auto deterministic = bool(node->i(Symbol::attr("deterministic")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::cudnn_convolution_transpose(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), padding, output_padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose", 3, 1);
  }},
  {"cudnn_convolution_transpose_backward-11", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward");
      AutoGPU device_guard(deviceForInputs(stack, 11));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 3, 11)));
      auto output_padding = tensor_as<IntList>(std::move(peek(stack, 4, 11)));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 5, 11)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 6, 11)));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 7, 11)));
      auto benchmark = tensor_as<bool>(std::move(peek(stack, 8, 11)));
      auto deterministic = tensor_as<bool>(std::move(peek(stack, 9, 11)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(peek(stack, 10, 11)));
      auto result = at::cudnn_convolution_transpose_backward(std::move(peek(stack, 0, 11)), std::move(peek(stack, 1, 11)), std::move(peek(stack, 2, 11)), padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
      drop(stack, 11);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose_backward", 11, 3);
  }},
  {"cudnn_convolution_transpose_backward-3-benchmark_i-deterministic_i-dilation_is-groups_i-output_mask_is-output_padding_is-padding_is-stride_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    auto benchmark = bool(node->i(Symbol::attr("benchmark")));
    auto deterministic = bool(node->i(Symbol::attr("deterministic")));
    auto output_mask = as_bool_array<3>(node->is(Symbol::attr("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::cudnn_convolution_transpose_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose_backward", 3, 3);
  }},
  {"cudnn_convolution_transpose_backward_bias-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward_bias");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::cudnn_convolution_transpose_backward_bias(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose_backward_bias", 1, 1);
  }},
  {"cudnn_convolution_transpose_backward_input-2-benchmark_i-deterministic_i-dilation_is-groups_i-padding_is-stride_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    auto benchmark = bool(node->i(Symbol::attr("benchmark")));
    auto deterministic = bool(node->i(Symbol::attr("deterministic")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward_input");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::cudnn_convolution_transpose_backward_input(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose_backward_input", 2, 1);
  }},
  {"cudnn_convolution_transpose_backward_input-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward_input");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 2, 8)));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 3, 8)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 4, 8)));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 5, 8)));
      auto benchmark = tensor_as<bool>(std::move(peek(stack, 6, 8)));
      auto deterministic = tensor_as<bool>(std::move(peek(stack, 7, 8)));
      auto result = at::cudnn_convolution_transpose_backward_input(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose_backward_input", 8, 1);
  }},
  {"cudnn_convolution_transpose_backward_weight-2-benchmark_i-deterministic_i-dilation_is-groups_i-padding_is-stride_is-weight_size_is", [](Node *node) {
    auto weight_size = std::vector<int64_t>(node->is(Symbol::attr("weight_size")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto groups = int64_t(node->i(Symbol::attr("groups")));
    auto benchmark = bool(node->i(Symbol::attr("benchmark")));
    auto deterministic = bool(node->i(Symbol::attr("deterministic")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward_weight");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::cudnn_convolution_transpose_backward_weight(weight_size, std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose_backward_weight", 2, 1);
  }},
  {"cudnn_convolution_transpose_backward_weight-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_convolution_transpose_backward_weight");
      AutoGPU device_guard(deviceForInputs(stack, 9));
      auto weight_size = tensor_as<IntList>(std::move(peek(stack, 0, 9)));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 3, 9)));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 4, 9)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 5, 9)));
      auto groups = tensor_as<int64_t>(std::move(peek(stack, 6, 9)));
      auto benchmark = tensor_as<bool>(std::move(peek(stack, 7, 9)));
      auto deterministic = tensor_as<bool>(std::move(peek(stack, 8, 9)));
      auto result = at::cudnn_convolution_transpose_backward_weight(weight_size, std::move(peek(stack, 1, 9)), std::move(peek(stack, 2, 9)), padding, stride, dilation, groups, benchmark, deterministic);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_convolution_transpose_backward_weight", 9, 1);
  }},
  {"cudnn_grid_sampler-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_grid_sampler");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::cudnn_grid_sampler(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_grid_sampler", 2, 1);
  }},
  {"cudnn_grid_sampler_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_grid_sampler_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::cudnn_grid_sampler_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_grid_sampler_backward", 3, 2);
  }},
  {"cudnn_is_acceptable-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cudnn_is_acceptable");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::cudnn_is_acceptable(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cudnn_is_acceptable", 1, 1);
  }},
  {"cumprod-1-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cumprod");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::cumprod(std::move(peek(stack, 0, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cumprod", 1, 1);
  }},
  {"cumprod-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cumprod");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::cumprod(std::move(peek(stack, 0, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cumprod", 2, 1);
  }},
  {"cumsum-1-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cumsum");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::cumsum(std::move(peek(stack, 0, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "cumsum", 1, 1);
  }},
  {"cumsum-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("cumsum");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::cumsum(std::move(peek(stack, 0, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "cumsum", 2, 1);
  }},
  {"data_ptr-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("data_ptr");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).data_ptr();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "data_ptr", 1, 1);
  }},
  {"det-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("det");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::det(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "det", 1, 1);
  }},
  {"diag-1-diagonal_i", [](Node *node) {
    auto diagonal = int64_t(node->i(Symbol::attr("diagonal")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("diag");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::diag(std::move(peek(stack, 0, 1)), diagonal);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "diag", 1, 1);
  }},
  {"diag-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("diag");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto diagonal = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::diag(std::move(peek(stack, 0, 2)), diagonal);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "diag", 2, 1);
  }},
  {"diagflat-1-offset_i", [](Node *node) {
    auto offset = int64_t(node->i(Symbol::attr("offset")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("diagflat");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::diagflat(std::move(peek(stack, 0, 1)), offset);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "diagflat", 1, 1);
  }},
  {"diagflat-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("diagflat");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto offset = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::diagflat(std::move(peek(stack, 0, 2)), offset);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "diagflat", 2, 1);
  }},
  {"diagonal-1-dim1_i-dim2_i-offset_i", [](Node *node) {
    auto offset = int64_t(node->i(Symbol::attr("offset")));
    auto dim1 = int64_t(node->i(Symbol::attr("dim1")));
    auto dim2 = int64_t(node->i(Symbol::attr("dim2")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("diagonal");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::diagonal(std::move(peek(stack, 0, 1)), offset, dim1, dim2);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "diagonal", 1, 1);
  }},
  {"diagonal-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("diagonal");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto offset = tensor_as<int64_t>(std::move(peek(stack, 1, 4)));
      auto dim1 = tensor_as<int64_t>(std::move(peek(stack, 2, 4)));
      auto dim2 = tensor_as<int64_t>(std::move(peek(stack, 3, 4)));
      auto result = at::diagonal(std::move(peek(stack, 0, 4)), offset, dim1, dim2);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "diagonal", 4, 1);
  }},
  {"digamma-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("digamma");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::digamma(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "digamma", 1, 1);
  }},
  {"dim-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("dim");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).dim();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "dim", 1, 1);
  }},
  {"dist-2-p_t", [](Node *node) {
    auto p = Scalar(node->t(Symbol::attr("p")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("dist");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::dist(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), p);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "dist", 2, 1);
  }},
  {"dist-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("dist");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto p = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::dist(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), p);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "dist", 3, 1);
  }},
  {"div-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("div");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::div(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "div", 1, 1);
  }},
  {"div-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("div");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::div(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "div", 2, 1);
  }},
  {"dot-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("dot");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::dot(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "dot", 2, 1);
  }},
  {"eig-1-eigenvectors_i", [](Node *node) {
    auto eigenvectors = bool(node->i(Symbol::attr("eigenvectors")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("eig");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::eig(std::move(peek(stack, 0, 1)), eigenvectors);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "eig", 1, 2);
  }},
  {"eig-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("eig");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto eigenvectors = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::eig(std::move(peek(stack, 0, 2)), eigenvectors);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "eig", 2, 2);
  }},
  {"elu-1-alpha_t-scale_t", [](Node *node) {
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    auto scale = Scalar(node->t(Symbol::attr("scale")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("elu");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::elu(std::move(peek(stack, 0, 1)), alpha, scale);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "elu", 1, 1);
  }},
  {"elu-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("elu");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto alpha = tensor_as<Scalar>(std::move(peek(stack, 1, 3)));
      auto scale = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::elu(std::move(peek(stack, 0, 3)), alpha, scale);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "elu", 3, 1);
  }},
  {"elu_backward-2-alpha_t-scale_t", [](Node *node) {
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    auto scale = Scalar(node->t(Symbol::attr("scale")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("elu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::elu_backward(std::move(peek(stack, 0, 2)), alpha, scale, std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "elu_backward", 2, 1);
  }},
  {"elu_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("elu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto alpha = tensor_as<Scalar>(std::move(peek(stack, 1, 4)));
      auto scale = tensor_as<Scalar>(std::move(peek(stack, 2, 4)));
      auto result = at::elu_backward(std::move(peek(stack, 0, 4)), alpha, scale, std::move(peek(stack, 3, 4)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "elu_backward", 4, 1);
  }},
  {"elu_forward-1-alpha_t-scale_t", [](Node *node) {
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    auto scale = Scalar(node->t(Symbol::attr("scale")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("elu_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::elu_forward(std::move(peek(stack, 0, 1)), alpha, scale);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "elu_forward", 1, 1);
  }},
  {"elu_forward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("elu_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto alpha = tensor_as<Scalar>(std::move(peek(stack, 1, 3)));
      auto scale = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::elu_forward(std::move(peek(stack, 0, 3)), alpha, scale);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "elu_forward", 3, 1);
  }},
  {"embedding-2-padding_idx_i-scale_grad_by_freq_i-sparse_i", [](Node *node) {
    auto padding_idx = int64_t(node->i(Symbol::attr("padding_idx")));
    auto scale_grad_by_freq = bool(node->i(Symbol::attr("scale_grad_by_freq")));
    auto sparse = bool(node->i(Symbol::attr("sparse")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::embedding(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), padding_idx, scale_grad_by_freq, sparse);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "embedding", 2, 1);
  }},
  {"embedding-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto padding_idx = tensor_as<int64_t>(std::move(peek(stack, 2, 5)));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(peek(stack, 3, 5)));
      auto sparse = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::embedding(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), padding_idx, scale_grad_by_freq, sparse);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "embedding", 5, 1);
  }},
  {"embedding_backward-2-num_weights_i-padding_idx_i-scale_grad_by_freq_i-sparse_i", [](Node *node) {
    auto num_weights = int64_t(node->i(Symbol::attr("num_weights")));
    auto padding_idx = int64_t(node->i(Symbol::attr("padding_idx")));
    auto scale_grad_by_freq = bool(node->i(Symbol::attr("scale_grad_by_freq")));
    auto sparse = bool(node->i(Symbol::attr("sparse")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::embedding_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), num_weights, padding_idx, scale_grad_by_freq, sparse);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_backward", 2, 1);
  }},
  {"embedding_backward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto num_weights = tensor_as<int64_t>(std::move(peek(stack, 2, 6)));
      auto padding_idx = tensor_as<int64_t>(std::move(peek(stack, 3, 6)));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(peek(stack, 4, 6)));
      auto sparse = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::embedding_backward(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), num_weights, padding_idx, scale_grad_by_freq, sparse);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_backward", 6, 1);
  }},
  {"embedding_bag-3-mode_i-scale_grad_by_freq_i-sparse_i", [](Node *node) {
    auto scale_grad_by_freq = bool(node->i(Symbol::attr("scale_grad_by_freq")));
    auto mode = int64_t(node->i(Symbol::attr("mode")));
    auto sparse = bool(node->i(Symbol::attr("sparse")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::embedding_bag(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), scale_grad_by_freq, mode, sparse);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag", 3, 4);
  }},
  {"embedding_bag-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(peek(stack, 3, 6)));
      auto mode = tensor_as<int64_t>(std::move(peek(stack, 4, 6)));
      auto sparse = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::embedding_bag(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), scale_grad_by_freq, mode, sparse);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag", 6, 4);
  }},
  {"embedding_bag_backward-10", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag_backward");
      AutoGPU device_guard(deviceForInputs(stack, 10));
      auto num_weights = tensor_as<int64_t>(std::move(peek(stack, 6, 10)));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(peek(stack, 7, 10)));
      auto mode = tensor_as<int64_t>(std::move(peek(stack, 8, 10)));
      auto sparse = tensor_as<bool>(std::move(peek(stack, 9, 10)));
      auto result = at::embedding_bag_backward(std::move(peek(stack, 0, 10)), std::move(peek(stack, 1, 10)), std::move(peek(stack, 2, 10)), std::move(peek(stack, 3, 10)), std::move(peek(stack, 4, 10)), std::move(peek(stack, 5, 10)), num_weights, scale_grad_by_freq, mode, sparse);
      drop(stack, 10);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag_backward", 10, 1);
  }},
  {"embedding_bag_backward-6-mode_i-num_weights_i-scale_grad_by_freq_i-sparse_i", [](Node *node) {
    auto num_weights = int64_t(node->i(Symbol::attr("num_weights")));
    auto scale_grad_by_freq = bool(node->i(Symbol::attr("scale_grad_by_freq")));
    auto mode = int64_t(node->i(Symbol::attr("mode")));
    auto sparse = bool(node->i(Symbol::attr("sparse")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
  
      auto result = at::embedding_bag_backward(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), std::move(peek(stack, 3, 6)), std::move(peek(stack, 4, 6)), std::move(peek(stack, 5, 6)), num_weights, scale_grad_by_freq, mode, sparse);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag_backward", 6, 1);
  }},
  {"embedding_bag_dense_backward-6-mode_i-num_weights_i-scale_grad_by_freq_i", [](Node *node) {
    auto num_weights = int64_t(node->i(Symbol::attr("num_weights")));
    auto scale_grad_by_freq = bool(node->i(Symbol::attr("scale_grad_by_freq")));
    auto mode = int64_t(node->i(Symbol::attr("mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag_dense_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
  
      auto result = at::embedding_bag_dense_backward(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), std::move(peek(stack, 3, 6)), std::move(peek(stack, 4, 6)), std::move(peek(stack, 5, 6)), num_weights, scale_grad_by_freq, mode);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag_dense_backward", 6, 1);
  }},
  {"embedding_bag_dense_backward-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag_dense_backward");
      AutoGPU device_guard(deviceForInputs(stack, 9));
      auto num_weights = tensor_as<int64_t>(std::move(peek(stack, 6, 9)));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(peek(stack, 7, 9)));
      auto mode = tensor_as<int64_t>(std::move(peek(stack, 8, 9)));
      auto result = at::embedding_bag_dense_backward(std::move(peek(stack, 0, 9)), std::move(peek(stack, 1, 9)), std::move(peek(stack, 2, 9)), std::move(peek(stack, 3, 9)), std::move(peek(stack, 4, 9)), std::move(peek(stack, 5, 9)), num_weights, scale_grad_by_freq, mode);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag_dense_backward", 9, 1);
  }},
  {"embedding_bag_sparse_backward-5-mode_i-num_weights_i-scale_grad_by_freq_i", [](Node *node) {
    auto num_weights = int64_t(node->i(Symbol::attr("num_weights")));
    auto scale_grad_by_freq = bool(node->i(Symbol::attr("scale_grad_by_freq")));
    auto mode = int64_t(node->i(Symbol::attr("mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag_sparse_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
  
      auto result = at::embedding_bag_sparse_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), std::move(peek(stack, 3, 5)), std::move(peek(stack, 4, 5)), num_weights, scale_grad_by_freq, mode);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag_sparse_backward", 5, 1);
  }},
  {"embedding_bag_sparse_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_bag_sparse_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto num_weights = tensor_as<int64_t>(std::move(peek(stack, 5, 8)));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(peek(stack, 6, 8)));
      auto mode = tensor_as<int64_t>(std::move(peek(stack, 7, 8)));
      auto result = at::embedding_bag_sparse_backward(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), std::move(peek(stack, 2, 8)), std::move(peek(stack, 3, 8)), std::move(peek(stack, 4, 8)), num_weights, scale_grad_by_freq, mode);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_bag_sparse_backward", 8, 1);
  }},
  {"embedding_dense_backward-2-num_weights_i-padding_idx_i-scale_grad_by_freq_i", [](Node *node) {
    auto num_weights = int64_t(node->i(Symbol::attr("num_weights")));
    auto padding_idx = int64_t(node->i(Symbol::attr("padding_idx")));
    auto scale_grad_by_freq = bool(node->i(Symbol::attr("scale_grad_by_freq")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_dense_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::embedding_dense_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), num_weights, padding_idx, scale_grad_by_freq);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_dense_backward", 2, 1);
  }},
  {"embedding_dense_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_dense_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto num_weights = tensor_as<int64_t>(std::move(peek(stack, 2, 5)));
      auto padding_idx = tensor_as<int64_t>(std::move(peek(stack, 3, 5)));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::embedding_dense_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), num_weights, padding_idx, scale_grad_by_freq);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_dense_backward", 5, 1);
  }},
  {"embedding_sparse_backward-2-num_weights_i-padding_idx_i-scale_grad_by_freq_i", [](Node *node) {
    auto num_weights = int64_t(node->i(Symbol::attr("num_weights")));
    auto padding_idx = int64_t(node->i(Symbol::attr("padding_idx")));
    auto scale_grad_by_freq = bool(node->i(Symbol::attr("scale_grad_by_freq")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_sparse_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::embedding_sparse_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), num_weights, padding_idx, scale_grad_by_freq);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_sparse_backward", 2, 1);
  }},
  {"embedding_sparse_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("embedding_sparse_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto num_weights = tensor_as<int64_t>(std::move(peek(stack, 2, 5)));
      auto padding_idx = tensor_as<int64_t>(std::move(peek(stack, 3, 5)));
      auto scale_grad_by_freq = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::embedding_sparse_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), num_weights, padding_idx, scale_grad_by_freq);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "embedding_sparse_backward", 5, 1);
  }},
  {"empty_like-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("empty_like");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::empty_like(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "empty_like", 1, 1);
  }},
  {"eq-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("eq");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::eq(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "eq", 1, 1);
  }},
  {"eq-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("eq");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::eq(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "eq", 2, 1);
  }},
  {"equal-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("equal");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::equal(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "equal", 2, 1);
  }},
  {"erf-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("erf");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::erf(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "erf", 1, 1);
  }},
  {"erfinv-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("erfinv");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::erfinv(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "erfinv", 1, 1);
  }},
  {"exp-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("exp");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::exp(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "exp", 1, 1);
  }},
  {"expand-1-implicit_i-size_is", [](Node *node) {
    auto size = std::vector<int64_t>(node->is(Symbol::attr("size")));
    auto implicit = bool(node->i(Symbol::attr("implicit")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("expand");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).expand(size, implicit);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "expand", 1, 1);
  }},
  {"expand-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("expand");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto size = tensor_as<IntList>(std::move(peek(stack, 1, 3)));
      auto implicit = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = (std::move(peek(stack, 0, 3))).expand(size, implicit);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "expand", 3, 1);
  }},
  {"expand_as-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("expand_as");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = (std::move(peek(stack, 0, 2))).expand_as(std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "expand_as", 2, 1);
  }},
  {"expm1-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("expm1");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::expm1(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "expm1", 1, 1);
  }},
  {"fft-1-normalized_i-signal_ndim_i", [](Node *node) {
    auto signal_ndim = int64_t(node->i(Symbol::attr("signal_ndim")));
    auto normalized = bool(node->i(Symbol::attr("normalized")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fft");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::fft(std::move(peek(stack, 0, 1)), signal_ndim, normalized);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "fft", 1, 1);
  }},
  {"fft-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fft");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto signal_ndim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto normalized = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::fft(std::move(peek(stack, 0, 3)), signal_ndim, normalized);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "fft", 3, 1);
  }},
  {"floor-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("floor");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::floor(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "floor", 1, 1);
  }},
  {"fmod-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fmod");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::fmod(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "fmod", 1, 1);
  }},
  {"fmod-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fmod");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::fmod(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "fmod", 2, 1);
  }},
  {"frac-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("frac");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::frac(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "frac", 1, 1);
  }},
  {"fractional_max_pool2d-2-kernel_size_is-output_size_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fractional_max_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::fractional_max_pool2d(std::move(peek(stack, 0, 2)), kernel_size, output_size, std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "fractional_max_pool2d", 2, 2);
  }},
  {"fractional_max_pool2d-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fractional_max_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto kernel_size_tensor = peek(stack, 1, 4);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto output_size_tensor = peek(stack, 2, 4);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(2);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::fractional_max_pool2d(std::move(peek(stack, 0, 4)), kernel_size, output_size, std::move(peek(stack, 3, 4)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "fractional_max_pool2d", 4, 2);
  }},
  {"fractional_max_pool2d_backward-3-kernel_size_is-output_size_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fractional_max_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::fractional_max_pool2d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, output_size, std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "fractional_max_pool2d_backward", 3, 1);
  }},
  {"fractional_max_pool2d_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fractional_max_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto kernel_size_tensor = peek(stack, 2, 5);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto output_size_tensor = peek(stack, 3, 5);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(2);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::fractional_max_pool2d_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), kernel_size, output_size, std::move(peek(stack, 4, 5)));
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "fractional_max_pool2d_backward", 5, 1);
  }},
  {"fractional_max_pool2d_forward-2-kernel_size_is-output_size_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fractional_max_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::fractional_max_pool2d_forward(std::move(peek(stack, 0, 2)), kernel_size, output_size, std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "fractional_max_pool2d_forward", 2, 2);
  }},
  {"fractional_max_pool2d_forward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("fractional_max_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto kernel_size_tensor = peek(stack, 1, 4);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto output_size_tensor = peek(stack, 2, 4);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(2);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::fractional_max_pool2d_forward(std::move(peek(stack, 0, 4)), kernel_size, output_size, std::move(peek(stack, 3, 4)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "fractional_max_pool2d_forward", 4, 2);
  }},
  {"full_like-1-fill_value_t", [](Node *node) {
    auto fill_value = Scalar(node->t(Symbol::attr("fill_value")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("full_like");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::full_like(std::move(peek(stack, 0, 1)), fill_value);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "full_like", 1, 1);
  }},
  {"full_like-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("full_like");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto fill_value = tensor_as<Scalar>(std::move(peek(stack, 1, 2)));
      auto result = at::full_like(std::move(peek(stack, 0, 2)), fill_value);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "full_like", 2, 1);
  }},
  {"gather-2-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("gather");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::gather(std::move(peek(stack, 0, 2)), dim, std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "gather", 2, 1);
  }},
  {"gather-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("gather");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto result = at::gather(std::move(peek(stack, 0, 3)), dim, std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "gather", 3, 1);
  }},
  {"ge-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ge");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::ge(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "ge", 1, 1);
  }},
  {"ge-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ge");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::ge(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "ge", 2, 1);
  }},
  {"gels-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("gels");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::gels(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "gels", 2, 2);
  }},
  {"geqrf-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("geqrf");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::geqrf(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "geqrf", 1, 2);
  }},
  {"ger-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ger");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::ger(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "ger", 2, 1);
  }},
  {"gesv-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("gesv");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::gesv(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "gesv", 2, 2);
  }},
  {"get_device-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("get_device");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).get_device();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "get_device", 1, 1);
  }},
  {"glu-1-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("glu");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::glu(std::move(peek(stack, 0, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "glu", 1, 1);
  }},
  {"glu-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("glu");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::glu(std::move(peek(stack, 0, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "glu", 2, 1);
  }},
  {"glu_backward-2-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("glu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::glu_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "glu_backward", 2, 1);
  }},
  {"glu_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("glu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 2, 3)));
      auto result = at::glu_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), dim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "glu_backward", 3, 1);
  }},
  {"glu_forward-1-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("glu_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::glu_forward(std::move(peek(stack, 0, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "glu_forward", 1, 1);
  }},
  {"glu_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("glu_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::glu_forward(std::move(peek(stack, 0, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "glu_forward", 2, 1);
  }},
  {"group_norm-3-cudnn_enabled_i-eps_f-num_groups_i", [](Node *node) {
    auto num_groups = int64_t(node->i(Symbol::attr("num_groups")));
    auto eps = double(node->f(Symbol::attr("eps")));
    auto cudnn_enabled = bool(node->i(Symbol::attr("cudnn_enabled")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("group_norm");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::group_norm(std::move(peek(stack, 0, 3)), num_groups, std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), eps, cudnn_enabled);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "group_norm", 3, 1);
  }},
  {"group_norm-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("group_norm");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto num_groups = tensor_as<int64_t>(std::move(peek(stack, 1, 6)));
      auto eps = tensor_as<double>(std::move(peek(stack, 4, 6)));
      auto cudnn_enabled = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::group_norm(std::move(peek(stack, 0, 6)), num_groups, std::move(peek(stack, 2, 6)), std::move(peek(stack, 3, 6)), eps, cudnn_enabled);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "group_norm", 6, 1);
  }},
  {"gt-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("gt");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::gt(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "gt", 1, 1);
  }},
  {"gt-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("gt");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::gt(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "gt", 2, 1);
  }},
  {"hardshrink-1-lambd_t", [](Node *node) {
    auto lambd = Scalar(node->t(Symbol::attr("lambd")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardshrink");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::hardshrink(std::move(peek(stack, 0, 1)), lambd);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "hardshrink", 1, 1);
  }},
  {"hardshrink-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardshrink");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto lambd = tensor_as<Scalar>(std::move(peek(stack, 1, 2)));
      auto result = at::hardshrink(std::move(peek(stack, 0, 2)), lambd);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "hardshrink", 2, 1);
  }},
  {"hardshrink_backward-2-lambd_t", [](Node *node) {
    auto lambd = Scalar(node->t(Symbol::attr("lambd")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardshrink_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::hardshrink_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), lambd);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "hardshrink_backward", 2, 1);
  }},
  {"hardshrink_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardshrink_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto lambd = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::hardshrink_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), lambd);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "hardshrink_backward", 3, 1);
  }},
  {"hardshrink_forward-1-lambd_t", [](Node *node) {
    auto lambd = Scalar(node->t(Symbol::attr("lambd")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardshrink_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::hardshrink_forward(std::move(peek(stack, 0, 1)), lambd);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "hardshrink_forward", 1, 1);
  }},
  {"hardshrink_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardshrink_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto lambd = tensor_as<Scalar>(std::move(peek(stack, 1, 2)));
      auto result = at::hardshrink_forward(std::move(peek(stack, 0, 2)), lambd);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "hardshrink_forward", 2, 1);
  }},
  {"hardtanh-1-max_val_t-min_val_t", [](Node *node) {
    auto min_val = Scalar(node->t(Symbol::attr("min_val")));
    auto max_val = Scalar(node->t(Symbol::attr("max_val")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardtanh");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::hardtanh(std::move(peek(stack, 0, 1)), min_val, max_val);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "hardtanh", 1, 1);
  }},
  {"hardtanh-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardtanh");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto min_val = tensor_as<Scalar>(std::move(peek(stack, 1, 3)));
      auto max_val = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::hardtanh(std::move(peek(stack, 0, 3)), min_val, max_val);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "hardtanh", 3, 1);
  }},
  {"hardtanh_backward-2-max_val_t-min_val_t", [](Node *node) {
    auto min_val = Scalar(node->t(Symbol::attr("min_val")));
    auto max_val = Scalar(node->t(Symbol::attr("max_val")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardtanh_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::hardtanh_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), min_val, max_val);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "hardtanh_backward", 2, 1);
  }},
  {"hardtanh_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardtanh_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto min_val = tensor_as<Scalar>(std::move(peek(stack, 2, 4)));
      auto max_val = tensor_as<Scalar>(std::move(peek(stack, 3, 4)));
      auto result = at::hardtanh_backward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), min_val, max_val);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "hardtanh_backward", 4, 1);
  }},
  {"hardtanh_forward-1-max_val_t-min_val_t", [](Node *node) {
    auto min_val = Scalar(node->t(Symbol::attr("min_val")));
    auto max_val = Scalar(node->t(Symbol::attr("max_val")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardtanh_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::hardtanh_forward(std::move(peek(stack, 0, 1)), min_val, max_val);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "hardtanh_forward", 1, 1);
  }},
  {"hardtanh_forward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hardtanh_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto min_val = tensor_as<Scalar>(std::move(peek(stack, 1, 3)));
      auto max_val = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::hardtanh_forward(std::move(peek(stack, 0, 3)), min_val, max_val);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "hardtanh_forward", 3, 1);
  }},
  {"hinge_embedding_loss-2-margin_f-reduce_i-size_average_i", [](Node *node) {
    auto margin = double(node->f(Symbol::attr("margin")));
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hinge_embedding_loss");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::hinge_embedding_loss(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), margin, size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "hinge_embedding_loss", 2, 1);
  }},
  {"hinge_embedding_loss-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hinge_embedding_loss");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto margin = tensor_as<double>(std::move(peek(stack, 2, 5)));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 3, 5)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::hinge_embedding_loss(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), margin, size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "hinge_embedding_loss", 5, 1);
  }},
  {"histc-1-bins_i-max_t-min_t", [](Node *node) {
    auto bins = int64_t(node->i(Symbol::attr("bins")));
    auto min = Scalar(node->t(Symbol::attr("min")));
    auto max = Scalar(node->t(Symbol::attr("max")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("histc");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::histc(std::move(peek(stack, 0, 1)), bins, min, max);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "histc", 1, 1);
  }},
  {"histc-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("histc");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto bins = tensor_as<int64_t>(std::move(peek(stack, 1, 4)));
      auto min = tensor_as<Scalar>(std::move(peek(stack, 2, 4)));
      auto max = tensor_as<Scalar>(std::move(peek(stack, 3, 4)));
      auto result = at::histc(std::move(peek(stack, 0, 4)), bins, min, max);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "histc", 4, 1);
  }},
  {"hspmm-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("hspmm");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::hspmm(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "hspmm", 2, 1);
  }},
  {"ifft-1-normalized_i-signal_ndim_i", [](Node *node) {
    auto signal_ndim = int64_t(node->i(Symbol::attr("signal_ndim")));
    auto normalized = bool(node->i(Symbol::attr("normalized")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ifft");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::ifft(std::move(peek(stack, 0, 1)), signal_ndim, normalized);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "ifft", 1, 1);
  }},
  {"ifft-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ifft");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto signal_ndim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto normalized = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::ifft(std::move(peek(stack, 0, 3)), signal_ndim, normalized);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "ifft", 3, 1);
  }},
  {"index_select-2-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("index_select");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::index_select(std::move(peek(stack, 0, 2)), dim, std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "index_select", 2, 1);
  }},
  {"index_select-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("index_select");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto result = at::index_select(std::move(peek(stack, 0, 3)), dim, std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "index_select", 3, 1);
  }},
  {"inverse-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("inverse");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::inverse(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "inverse", 1, 1);
  }},
  {"irfft-1-normalized_i-onesided_i-signal_ndim_i-signal_sizes_is", [](Node *node) {
    auto signal_ndim = int64_t(node->i(Symbol::attr("signal_ndim")));
    auto normalized = bool(node->i(Symbol::attr("normalized")));
    auto onesided = bool(node->i(Symbol::attr("onesided")));
    auto signal_sizes = std::vector<int64_t>(node->is(Symbol::attr("signal_sizes")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("irfft");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::irfft(std::move(peek(stack, 0, 1)), signal_ndim, normalized, onesided, signal_sizes);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "irfft", 1, 1);
  }},
  {"irfft-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("irfft");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto signal_ndim = tensor_as<int64_t>(std::move(peek(stack, 1, 5)));
      auto normalized = tensor_as<bool>(std::move(peek(stack, 2, 5)));
      auto onesided = tensor_as<bool>(std::move(peek(stack, 3, 5)));
      auto signal_sizes = tensor_as<IntList>(std::move(peek(stack, 4, 5)));
      auto result = at::irfft(std::move(peek(stack, 0, 5)), signal_ndim, normalized, onesided, signal_sizes);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "irfft", 5, 1);
  }},
  {"is_coalesced-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_coalesced");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).is_coalesced();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_coalesced", 1, 1);
  }},
  {"is_contiguous-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_contiguous");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).is_contiguous();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_contiguous", 1, 1);
  }},
  {"is_cuda-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_cuda");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::is_cuda(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_cuda", 1, 1);
  }},
  {"is_distributed-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_distributed");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::is_distributed(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_distributed", 1, 1);
  }},
  {"is_floating_point-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_floating_point");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::is_floating_point(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_floating_point", 1, 1);
  }},
  {"is_nonzero-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_nonzero");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::is_nonzero(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_nonzero", 1, 1);
  }},
  {"is_same_size-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_same_size");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::is_same_size(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "is_same_size", 2, 1);
  }},
  {"is_set_to-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_set_to");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = (std::move(peek(stack, 0, 2))).is_set_to(std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "is_set_to", 2, 1);
  }},
  {"is_signed-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_signed");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::is_signed(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_signed", 1, 1);
  }},
  {"is_sparse-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("is_sparse");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::is_sparse(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "is_sparse", 1, 1);
  }},
  {"isclose-2-atol_f-equal_nan_i-rtol_f", [](Node *node) {
    auto rtol = double(node->f(Symbol::attr("rtol")));
    auto atol = double(node->f(Symbol::attr("atol")));
    auto equal_nan = bool(node->i(Symbol::attr("equal_nan")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("isclose");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::isclose(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), rtol, atol, equal_nan);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "isclose", 2, 1);
  }},
  {"isclose-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("isclose");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto rtol = tensor_as<double>(std::move(peek(stack, 2, 5)));
      auto atol = tensor_as<double>(std::move(peek(stack, 3, 5)));
      auto equal_nan = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::isclose(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), rtol, atol, equal_nan);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "isclose", 5, 1);
  }},
  {"kl_div-2-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kl_div");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::kl_div(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "kl_div", 2, 1);
  }},
  {"kl_div-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kl_div");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::kl_div(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "kl_div", 4, 1);
  }},
  {"kl_div_backward-3-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kl_div_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::kl_div_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "kl_div_backward", 3, 1);
  }},
  {"kl_div_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kl_div_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 3, 5)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::kl_div_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "kl_div_backward", 5, 1);
  }},
  {"kl_div_forward-2-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kl_div_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::kl_div_forward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "kl_div_forward", 2, 1);
  }},
  {"kl_div_forward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kl_div_forward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::kl_div_forward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "kl_div_forward", 4, 1);
  }},
  {"kthvalue-1-dim_i-k_i-keepdim_i", [](Node *node) {
    auto k = int64_t(node->i(Symbol::attr("k")));
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kthvalue");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::kthvalue(std::move(peek(stack, 0, 1)), k, dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "kthvalue", 1, 2);
  }},
  {"kthvalue-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("kthvalue");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto k = tensor_as<int64_t>(std::move(peek(stack, 1, 4)));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 2, 4)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::kthvalue(std::move(peek(stack, 0, 4)), k, dim, keepdim);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "kthvalue", 4, 2);
  }},
  {"l1_loss-2-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("l1_loss");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::l1_loss(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "l1_loss", 2, 1);
  }},
  {"l1_loss-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("l1_loss");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::l1_loss(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "l1_loss", 4, 1);
  }},
  {"l1_loss_backward-3-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("l1_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::l1_loss_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "l1_loss_backward", 3, 1);
  }},
  {"l1_loss_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("l1_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 3, 5)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::l1_loss_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "l1_loss_backward", 5, 1);
  }},
  {"l1_loss_forward-2-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("l1_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::l1_loss_forward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "l1_loss_forward", 2, 1);
  }},
  {"l1_loss_forward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("l1_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::l1_loss_forward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "l1_loss_forward", 4, 1);
  }},
  {"layer_norm-3-cudnn_enable_i-eps_f-normalized_shape_is", [](Node *node) {
    auto normalized_shape = std::vector<int64_t>(node->is(Symbol::attr("normalized_shape")));
    auto eps = double(node->f(Symbol::attr("eps")));
    auto cudnn_enable = bool(node->i(Symbol::attr("cudnn_enable")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("layer_norm");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::layer_norm(std::move(peek(stack, 0, 3)), normalized_shape, std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), eps, cudnn_enable);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "layer_norm", 3, 1);
  }},
  {"layer_norm-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("layer_norm");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto normalized_shape = tensor_as<IntList>(std::move(peek(stack, 1, 6)));
      auto eps = tensor_as<double>(std::move(peek(stack, 4, 6)));
      auto cudnn_enable = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::layer_norm(std::move(peek(stack, 0, 6)), normalized_shape, std::move(peek(stack, 2, 6)), std::move(peek(stack, 3, 6)), eps, cudnn_enable);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "layer_norm", 6, 1);
  }},
  {"le-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("le");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::le(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "le", 1, 1);
  }},
  {"le-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("le");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::le(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "le", 2, 1);
  }},
  {"leaky_relu-1-negative_slope_t", [](Node *node) {
    auto negative_slope = Scalar(node->t(Symbol::attr("negative_slope")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("leaky_relu");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::leaky_relu(std::move(peek(stack, 0, 1)), negative_slope);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "leaky_relu", 1, 1);
  }},
  {"leaky_relu-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("leaky_relu");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto negative_slope = tensor_as<Scalar>(std::move(peek(stack, 1, 2)));
      auto result = at::leaky_relu(std::move(peek(stack, 0, 2)), negative_slope);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "leaky_relu", 2, 1);
  }},
  {"leaky_relu_backward-2-negative_slope_t", [](Node *node) {
    auto negative_slope = Scalar(node->t(Symbol::attr("negative_slope")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("leaky_relu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::leaky_relu_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), negative_slope);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "leaky_relu_backward", 2, 1);
  }},
  {"leaky_relu_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("leaky_relu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto negative_slope = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::leaky_relu_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), negative_slope);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "leaky_relu_backward", 3, 1);
  }},
  {"leaky_relu_forward-1-negative_slope_t", [](Node *node) {
    auto negative_slope = Scalar(node->t(Symbol::attr("negative_slope")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("leaky_relu_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::leaky_relu_forward(std::move(peek(stack, 0, 1)), negative_slope);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "leaky_relu_forward", 1, 1);
  }},
  {"leaky_relu_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("leaky_relu_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto negative_slope = tensor_as<Scalar>(std::move(peek(stack, 1, 2)));
      auto result = at::leaky_relu_forward(std::move(peek(stack, 0, 2)), negative_slope);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "leaky_relu_forward", 2, 1);
  }},
  {"lerp-2-weight_t", [](Node *node) {
    auto weight = Scalar(node->t(Symbol::attr("weight")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("lerp");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::lerp(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), weight);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "lerp", 2, 1);
  }},
  {"lerp-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("lerp");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto weight = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::lerp(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), weight);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "lerp", 3, 1);
  }},
  {"lgamma-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("lgamma");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::lgamma(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "lgamma", 1, 1);
  }},
  {"log-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::log(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "log", 1, 1);
  }},
  {"log10-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log10");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::log10(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "log10", 1, 1);
  }},
  {"log1p-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log1p");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::log1p(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "log1p", 1, 1);
  }},
  {"log2-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log2");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::log2(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "log2", 1, 1);
  }},
  {"log_sigmoid-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_sigmoid");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::log_sigmoid(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "log_sigmoid", 1, 1);
  }},
  {"log_sigmoid_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_sigmoid_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::log_sigmoid_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "log_sigmoid_backward", 3, 1);
  }},
  {"log_sigmoid_forward-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_sigmoid_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::log_sigmoid_forward(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "log_sigmoid_forward", 1, 2);
  }},
  {"log_softmax-1-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_softmax");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::log_softmax(std::move(peek(stack, 0, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "log_softmax", 1, 1);
  }},
  {"log_softmax-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_softmax");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::log_softmax(std::move(peek(stack, 0, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "log_softmax", 2, 1);
  }},
  {"log_softmax_backward_data-3-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_softmax_backward_data");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::log_softmax_backward_data(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), dim, std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "log_softmax_backward_data", 3, 1);
  }},
  {"log_softmax_backward_data-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("log_softmax_backward_data");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 2, 4)));
      auto result = at::log_softmax_backward_data(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), dim, std::move(peek(stack, 3, 4)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "log_softmax_backward_data", 4, 1);
  }},
  {"logdet-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("logdet");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::logdet(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "logdet", 1, 1);
  }},
  {"logsumexp-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("logsumexp");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::logsumexp(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "logsumexp", 1, 1);
  }},
  {"logsumexp-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("logsumexp");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::logsumexp(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "logsumexp", 3, 1);
  }},
  {"lt-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("lt");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::lt(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "lt", 1, 1);
  }},
  {"lt-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("lt");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::lt(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "lt", 2, 1);
  }},
  {"margin_ranking_loss-3-margin_f-reduce_i-size_average_i", [](Node *node) {
    auto margin = double(node->f(Symbol::attr("margin")));
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("margin_ranking_loss");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::margin_ranking_loss(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), margin, size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "margin_ranking_loss", 3, 1);
  }},
  {"margin_ranking_loss-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("margin_ranking_loss");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto margin = tensor_as<double>(std::move(peek(stack, 3, 6)));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 4, 6)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::margin_ranking_loss(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), margin, size_average, reduce);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "margin_ranking_loss", 6, 1);
  }},
  {"masked_select-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("masked_select");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::masked_select(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "masked_select", 2, 1);
  }},
  {"matmul-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("matmul");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::matmul(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "matmul", 2, 1);
  }},
  {"max-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::max(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max", 1, 1);
  }},
  {"max-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::max(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max", 1, 2);
  }},
  {"max-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::max(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "max", 2, 1);
  }},
  {"max-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::max(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max", 3, 2);
  }},
  {"max_pool1d-1-ceil_mode_i-dilation_is-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto ceil_mode = bool(node->i(Symbol::attr("ceil_mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool1d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::max_pool1d(std::move(peek(stack, 0, 1)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool1d", 1, 2);
  }},
  {"max_pool1d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool1d");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto kernel_size_tensor = peek(stack, 1, 6);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(1);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 2, 6);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(1);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 3, 6);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(1);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 4, 6);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(1);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto ceil_mode = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::max_pool1d(std::move(peek(stack, 0, 6)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool1d", 6, 2);
  }},
  {"max_pool2d-1-ceil_mode_i-dilation_is-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto ceil_mode = bool(node->i(Symbol::attr("ceil_mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::max_pool2d(std::move(peek(stack, 0, 1)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool2d", 1, 2);
  }},
  {"max_pool2d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool2d");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto kernel_size_tensor = peek(stack, 1, 6);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 2, 6);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 3, 6);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 4, 6);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(2);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto ceil_mode = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::max_pool2d(std::move(peek(stack, 0, 6)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool2d", 6, 2);
  }},
  {"max_pool2d_backward-3-ceil_mode_i-dilation_is-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto ceil_mode = bool(node->i(Symbol::attr("ceil_mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::max_pool2d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, stride, padding, dilation, ceil_mode, std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool2d_backward", 3, 1);
  }},
  {"max_pool2d_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto kernel_size_tensor = peek(stack, 2, 8);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 3, 8);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 4, 8);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 5, 8);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(2);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto ceil_mode = tensor_as<bool>(std::move(peek(stack, 6, 8)));
      auto result = at::max_pool2d_backward(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), kernel_size, stride, padding, dilation, ceil_mode, std::move(peek(stack, 7, 8)));
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool2d_backward", 8, 1);
  }},
  {"max_pool2d_forward-1-ceil_mode_i-dilation_is-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto ceil_mode = bool(node->i(Symbol::attr("ceil_mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::max_pool2d_forward(std::move(peek(stack, 0, 1)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool2d_forward", 1, 2);
  }},
  {"max_pool2d_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto kernel_size_tensor = peek(stack, 1, 6);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 2, 6);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 3, 6);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 4, 6);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(2);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto ceil_mode = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::max_pool2d_forward(std::move(peek(stack, 0, 6)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool2d_forward", 6, 2);
  }},
  {"max_pool3d-1-ceil_mode_i-dilation_is-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto ceil_mode = bool(node->i(Symbol::attr("ceil_mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::max_pool3d(std::move(peek(stack, 0, 1)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool3d", 1, 2);
  }},
  {"max_pool3d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool3d");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto kernel_size_tensor = peek(stack, 1, 6);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 2, 6);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 3, 6);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 4, 6);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(3);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto ceil_mode = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::max_pool3d(std::move(peek(stack, 0, 6)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool3d", 6, 2);
  }},
  {"max_pool3d_backward-3-ceil_mode_i-dilation_is-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto ceil_mode = bool(node->i(Symbol::attr("ceil_mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::max_pool3d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, stride, padding, dilation, ceil_mode, std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool3d_backward", 3, 1);
  }},
  {"max_pool3d_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto kernel_size_tensor = peek(stack, 2, 8);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 3, 8);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 4, 8);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 5, 8);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(3);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto ceil_mode = tensor_as<bool>(std::move(peek(stack, 6, 8)));
      auto result = at::max_pool3d_backward(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), kernel_size, stride, padding, dilation, ceil_mode, std::move(peek(stack, 7, 8)));
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool3d_backward", 8, 1);
  }},
  {"max_pool3d_forward-1-ceil_mode_i-dilation_is-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto ceil_mode = bool(node->i(Symbol::attr("ceil_mode")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::max_pool3d_forward(std::move(peek(stack, 0, 1)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool3d_forward", 1, 2);
  }},
  {"max_pool3d_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_pool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto kernel_size_tensor = peek(stack, 1, 6);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 2, 6);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 3, 6);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 4, 6);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(3);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto ceil_mode = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::max_pool3d_forward(std::move(peek(stack, 0, 6)), kernel_size, stride, padding, dilation, ceil_mode);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "max_pool3d_forward", 6, 2);
  }},
  {"max_unpool2d-2-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool2d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::max_unpool2d(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool2d", 2, 1);
  }},
  {"max_unpool2d-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool2d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto output_size_tensor = peek(stack, 2, 3);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(2);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::max_unpool2d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), output_size);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool2d", 3, 1);
  }},
  {"max_unpool2d_backward-3-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::max_unpool2d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), output_size);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool2d_backward", 3, 1);
  }},
  {"max_unpool2d_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto output_size_tensor = peek(stack, 3, 4);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(2);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::max_unpool2d_backward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), std::move(peek(stack, 2, 4)), output_size);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool2d_backward", 4, 1);
  }},
  {"max_unpool2d_forward-2-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::max_unpool2d_forward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), output_size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool2d_forward", 2, 1);
  }},
  {"max_unpool2d_forward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto output_size_tensor = peek(stack, 2, 3);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(2);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto result = at::max_unpool2d_forward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), output_size);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool2d_forward", 3, 1);
  }},
  {"max_unpool3d-2-output_size_is-padding_is-stride_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool3d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::max_unpool3d(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), output_size, stride, padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool3d", 2, 1);
  }},
  {"max_unpool3d-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool3d");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto output_size_tensor = peek(stack, 2, 5);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(3);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto stride_tensor = peek(stack, 3, 5);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 4, 5);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::max_unpool3d(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), output_size, stride, padding);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool3d", 5, 1);
  }},
  {"max_unpool3d_backward-3-output_size_is-padding_is-stride_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::max_unpool3d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), output_size, stride, padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool3d_backward", 3, 1);
  }},
  {"max_unpool3d_backward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto output_size_tensor = peek(stack, 3, 6);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(3);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto stride_tensor = peek(stack, 4, 6);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 6);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::max_unpool3d_backward(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), output_size, stride, padding);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool3d_backward", 6, 1);
  }},
  {"max_unpool3d_forward-2-output_size_is-padding_is-stride_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::max_unpool3d_forward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), output_size, stride, padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool3d_forward", 2, 1);
  }},
  {"max_unpool3d_forward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_unpool3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto output_size_tensor = peek(stack, 2, 5);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(3);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto stride_tensor = peek(stack, 3, 5);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 4, 5);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::max_unpool3d_forward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), output_size, stride, padding);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "max_unpool3d_forward", 5, 1);
  }},
  {"max_values-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_values");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::max_values(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "max_values", 1, 1);
  }},
  {"max_values-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("max_values");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::max_values(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "max_values", 3, 1);
  }},
  {"mean-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mean");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::mean(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "mean", 1, 1);
  }},
  {"mean-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mean");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::mean(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "mean", 1, 1);
  }},
  {"mean-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mean");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::mean(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "mean", 3, 1);
  }},
  {"median-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("median");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::median(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "median", 1, 1);
  }},
  {"median-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("median");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::median(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "median", 1, 2);
  }},
  {"median-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("median");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::median(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "median", 3, 2);
  }},
  {"min-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("min");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::min(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "min", 1, 1);
  }},
  {"min-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("min");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::min(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "min", 1, 2);
  }},
  {"min-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("min");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::min(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "min", 2, 1);
  }},
  {"min-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("min");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::min(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "min", 3, 2);
  }},
  {"min_values-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("min_values");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::min_values(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "min_values", 1, 1);
  }},
  {"min_values-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("min_values");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::min_values(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "min_values", 3, 1);
  }},
  {"mkldnn_convolution-3-dilation_is-padding_is-stride_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mkldnn_convolution");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::mkldnn_convolution(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), padding, stride, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "mkldnn_convolution", 3, 1);
  }},
  {"mkldnn_convolution-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mkldnn_convolution");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 3, 6)));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 4, 6)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 5, 6)));
      auto result = at::mkldnn_convolution(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), padding, stride, dilation);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "mkldnn_convolution", 6, 1);
  }},
  {"mkldnn_convolution_backward-3-dilation_is-output_mask_is-padding_is-stride_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto output_mask = as_bool_array<3>(node->is(Symbol::attr("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mkldnn_convolution_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::mkldnn_convolution_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), padding, stride, dilation, output_mask);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "mkldnn_convolution_backward", 3, 3);
  }},
  {"mkldnn_convolution_backward-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mkldnn_convolution_backward");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 3, 7)));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 4, 7)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 5, 7)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(peek(stack, 6, 7)));
      auto result = at::mkldnn_convolution_backward(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), std::move(peek(stack, 2, 7)), padding, stride, dilation, output_mask);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "mkldnn_convolution_backward", 7, 3);
  }},
  {"mkldnn_convolution_backward_input-2-bias_defined_i-dilation_is-padding_is-self_size_is-stride_is", [](Node *node) {
    auto self_size = std::vector<int64_t>(node->is(Symbol::attr("self_size")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto bias_defined = bool(node->i(Symbol::attr("bias_defined")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mkldnn_convolution_backward_input");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::mkldnn_convolution_backward_input(self_size, std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), padding, stride, dilation, bias_defined);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "mkldnn_convolution_backward_input", 2, 1);
  }},
  {"mkldnn_convolution_backward_input-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mkldnn_convolution_backward_input");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto self_size = tensor_as<IntList>(std::move(peek(stack, 0, 7)));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 3, 7)));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 4, 7)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 5, 7)));
      auto bias_defined = tensor_as<bool>(std::move(peek(stack, 6, 7)));
      auto result = at::mkldnn_convolution_backward_input(self_size, std::move(peek(stack, 1, 7)), std::move(peek(stack, 2, 7)), padding, stride, dilation, bias_defined);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "mkldnn_convolution_backward_input", 7, 1);
  }},
  {"mkldnn_convolution_backward_weights-2-bias_defined_i-dilation_is-padding_is-stride_is-weight_size_is", [](Node *node) {
    auto weight_size = std::vector<int64_t>(node->is(Symbol::attr("weight_size")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto bias_defined = bool(node->i(Symbol::attr("bias_defined")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mkldnn_convolution_backward_weights");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::mkldnn_convolution_backward_weights(weight_size, std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), padding, stride, dilation, bias_defined);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "mkldnn_convolution_backward_weights", 2, 2);
  }},
  {"mkldnn_convolution_backward_weights-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mkldnn_convolution_backward_weights");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto weight_size = tensor_as<IntList>(std::move(peek(stack, 0, 7)));
      auto padding = tensor_as<IntList>(std::move(peek(stack, 3, 7)));
      auto stride = tensor_as<IntList>(std::move(peek(stack, 4, 7)));
      auto dilation = tensor_as<IntList>(std::move(peek(stack, 5, 7)));
      auto bias_defined = tensor_as<bool>(std::move(peek(stack, 6, 7)));
      auto result = at::mkldnn_convolution_backward_weights(weight_size, std::move(peek(stack, 1, 7)), std::move(peek(stack, 2, 7)), padding, stride, dilation, bias_defined);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "mkldnn_convolution_backward_weights", 7, 2);
  }},
  {"mm-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mm");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::mm(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "mm", 2, 1);
  }},
  {"mode-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mode");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::mode(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "mode", 1, 2);
  }},
  {"mode-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mode");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::mode(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "mode", 3, 2);
  }},
  {"mse_loss-2-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mse_loss");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::mse_loss(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "mse_loss", 2, 1);
  }},
  {"mse_loss-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mse_loss");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::mse_loss(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "mse_loss", 4, 1);
  }},
  {"mse_loss_backward-3-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mse_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::mse_loss_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "mse_loss_backward", 3, 1);
  }},
  {"mse_loss_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mse_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 3, 5)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::mse_loss_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "mse_loss_backward", 5, 1);
  }},
  {"mse_loss_forward-2-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mse_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::mse_loss_forward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "mse_loss_forward", 2, 1);
  }},
  {"mse_loss_forward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mse_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::mse_loss_forward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "mse_loss_forward", 4, 1);
  }},
  {"mul-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mul");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::mul(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "mul", 1, 1);
  }},
  {"mul-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mul");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::mul(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "mul", 2, 1);
  }},
  {"multi_margin_loss-3-margin_t-p_t-reduce_i-size_average_i", [](Node *node) {
    auto p = Scalar(node->t(Symbol::attr("p")));
    auto margin = Scalar(node->t(Symbol::attr("margin")));
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multi_margin_loss");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::multi_margin_loss(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), p, margin, std::move(peek(stack, 2, 3)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "multi_margin_loss", 3, 1);
  }},
  {"multi_margin_loss-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multi_margin_loss");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto p = tensor_as<Scalar>(std::move(peek(stack, 2, 7)));
      auto margin = tensor_as<Scalar>(std::move(peek(stack, 3, 7)));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 5, 7)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 6, 7)));
      auto result = at::multi_margin_loss(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), p, margin, std::move(peek(stack, 4, 7)), size_average, reduce);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "multi_margin_loss", 7, 1);
  }},
  {"multi_margin_loss_backward-4-margin_t-p_t-reduce_i-size_average_i", [](Node *node) {
    auto p = Scalar(node->t(Symbol::attr("p")));
    auto margin = Scalar(node->t(Symbol::attr("margin")));
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multi_margin_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
  
      auto result = at::multi_margin_loss_backward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), std::move(peek(stack, 2, 4)), p, margin, std::move(peek(stack, 3, 4)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "multi_margin_loss_backward", 4, 1);
  }},
  {"multi_margin_loss_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multi_margin_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto p = tensor_as<Scalar>(std::move(peek(stack, 3, 8)));
      auto margin = tensor_as<Scalar>(std::move(peek(stack, 4, 8)));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 6, 8)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 7, 8)));
      auto result = at::multi_margin_loss_backward(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), std::move(peek(stack, 2, 8)), p, margin, std::move(peek(stack, 5, 8)), size_average, reduce);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "multi_margin_loss_backward", 8, 1);
  }},
  {"multi_margin_loss_forward-3-margin_t-p_t-reduce_i-size_average_i", [](Node *node) {
    auto p = Scalar(node->t(Symbol::attr("p")));
    auto margin = Scalar(node->t(Symbol::attr("margin")));
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multi_margin_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::multi_margin_loss_forward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), p, margin, std::move(peek(stack, 2, 3)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "multi_margin_loss_forward", 3, 1);
  }},
  {"multi_margin_loss_forward-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multi_margin_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto p = tensor_as<Scalar>(std::move(peek(stack, 2, 7)));
      auto margin = tensor_as<Scalar>(std::move(peek(stack, 3, 7)));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 5, 7)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 6, 7)));
      auto result = at::multi_margin_loss_forward(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), p, margin, std::move(peek(stack, 4, 7)), size_average, reduce);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "multi_margin_loss_forward", 7, 1);
  }},
  {"multilabel_margin_loss-2-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multilabel_margin_loss");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::multilabel_margin_loss(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "multilabel_margin_loss", 2, 1);
  }},
  {"multilabel_margin_loss-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multilabel_margin_loss");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::multilabel_margin_loss(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "multilabel_margin_loss", 4, 1);
  }},
  {"multilabel_margin_loss_backward-4-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multilabel_margin_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
  
      auto result = at::multilabel_margin_loss_backward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), std::move(peek(stack, 2, 4)), size_average, reduce, std::move(peek(stack, 3, 4)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "multilabel_margin_loss_backward", 4, 1);
  }},
  {"multilabel_margin_loss_backward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multilabel_margin_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 3, 6)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 4, 6)));
      auto result = at::multilabel_margin_loss_backward(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), size_average, reduce, std::move(peek(stack, 5, 6)));
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "multilabel_margin_loss_backward", 6, 1);
  }},
  {"multilabel_margin_loss_forward-2-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multilabel_margin_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::multilabel_margin_loss_forward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "multilabel_margin_loss_forward", 2, 2);
  }},
  {"multilabel_margin_loss_forward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("multilabel_margin_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::multilabel_margin_loss_forward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "multilabel_margin_loss_forward", 4, 2);
  }},
  {"mv-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("mv");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::mv(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "mv", 2, 1);
  }},
  {"narrow-1-dim_i-length_i-start_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto start = int64_t(node->i(Symbol::attr("start")));
    auto length = int64_t(node->i(Symbol::attr("length")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("narrow");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::narrow(std::move(peek(stack, 0, 1)), dim, start, length);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "narrow", 1, 1);
  }},
  {"narrow-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("narrow");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 4)));
      auto start = tensor_as<int64_t>(std::move(peek(stack, 2, 4)));
      auto length = tensor_as<int64_t>(std::move(peek(stack, 3, 4)));
      auto result = at::narrow(std::move(peek(stack, 0, 4)), dim, start, length);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "narrow", 4, 1);
  }},
  {"ne-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ne");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::ne(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "ne", 1, 1);
  }},
  {"ne-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ne");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::ne(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "ne", 2, 1);
  }},
  {"neg-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("neg");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::neg(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "neg", 1, 1);
  }},
  {"nll_loss-3-ignore_index_i-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto ignore_index = int64_t(node->i(Symbol::attr("ignore_index")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::nll_loss(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), size_average, ignore_index, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss", 3, 1);
  }},
  {"nll_loss-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 3, 6)));
      auto ignore_index = tensor_as<int64_t>(std::move(peek(stack, 4, 6)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::nll_loss(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), size_average, ignore_index, reduce);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss", 6, 1);
  }},
  {"nll_loss2d-3-ignore_index_i-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto ignore_index = int64_t(node->i(Symbol::attr("ignore_index")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss2d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::nll_loss2d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), size_average, ignore_index, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss2d", 3, 1);
  }},
  {"nll_loss2d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss2d");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 3, 6)));
      auto ignore_index = tensor_as<int64_t>(std::move(peek(stack, 4, 6)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::nll_loss2d(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), size_average, ignore_index, reduce);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss2d", 6, 1);
  }},
  {"nll_loss2d_backward-5-ignore_index_i-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto ignore_index = int64_t(node->i(Symbol::attr("ignore_index")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
  
      auto result = at::nll_loss2d_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), std::move(peek(stack, 3, 5)), size_average, ignore_index, reduce, std::move(peek(stack, 4, 5)));
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss2d_backward", 5, 1);
  }},
  {"nll_loss2d_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 4, 8)));
      auto ignore_index = tensor_as<int64_t>(std::move(peek(stack, 5, 8)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 6, 8)));
      auto result = at::nll_loss2d_backward(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), std::move(peek(stack, 2, 8)), std::move(peek(stack, 3, 8)), size_average, ignore_index, reduce, std::move(peek(stack, 7, 8)));
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss2d_backward", 8, 1);
  }},
  {"nll_loss2d_forward-3-ignore_index_i-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto ignore_index = int64_t(node->i(Symbol::attr("ignore_index")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::nll_loss2d_forward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), size_average, ignore_index, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss2d_forward", 3, 2);
  }},
  {"nll_loss2d_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 3, 6)));
      auto ignore_index = tensor_as<int64_t>(std::move(peek(stack, 4, 6)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::nll_loss2d_forward(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), size_average, ignore_index, reduce);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss2d_forward", 6, 2);
  }},
  {"nll_loss_backward-5-ignore_index_i-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto ignore_index = int64_t(node->i(Symbol::attr("ignore_index")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
  
      auto result = at::nll_loss_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), std::move(peek(stack, 3, 5)), size_average, ignore_index, reduce, std::move(peek(stack, 4, 5)));
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss_backward", 5, 1);
  }},
  {"nll_loss_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 4, 8)));
      auto ignore_index = tensor_as<int64_t>(std::move(peek(stack, 5, 8)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 6, 8)));
      auto result = at::nll_loss_backward(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), std::move(peek(stack, 2, 8)), std::move(peek(stack, 3, 8)), size_average, ignore_index, reduce, std::move(peek(stack, 7, 8)));
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss_backward", 8, 1);
  }},
  {"nll_loss_forward-3-ignore_index_i-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto ignore_index = int64_t(node->i(Symbol::attr("ignore_index")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::nll_loss_forward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), size_average, ignore_index, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss_forward", 3, 2);
  }},
  {"nll_loss_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nll_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 3, 6)));
      auto ignore_index = tensor_as<int64_t>(std::move(peek(stack, 4, 6)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::nll_loss_forward(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), size_average, ignore_index, reduce);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "nll_loss_forward", 6, 2);
  }},
  {"nonzero-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("nonzero");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::nonzero(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "nonzero", 1, 1);
  }},
  {"norm-1-dim_i-keepdim_i-p_t", [](Node *node) {
    auto p = Scalar(node->t(Symbol::attr("p")));
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("norm");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::norm(std::move(peek(stack, 0, 1)), p, dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "norm", 1, 1);
  }},
  {"norm-1-p_t", [](Node *node) {
    auto p = Scalar(node->t(Symbol::attr("p")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("norm");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::norm(std::move(peek(stack, 0, 1)), p);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "norm", 1, 1);
  }},
  {"norm-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("norm");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto p = tensor_as<Scalar>(std::move(peek(stack, 1, 2)));
      auto result = at::norm(std::move(peek(stack, 0, 2)), p);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "norm", 2, 1);
  }},
  {"norm-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("norm");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto p = tensor_as<Scalar>(std::move(peek(stack, 1, 4)));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 2, 4)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::norm(std::move(peek(stack, 0, 4)), p, dim, keepdim);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "norm", 4, 1);
  }},
  {"numel-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("numel");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::numel(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "numel", 1, 1);
  }},
  {"ones_like-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ones_like");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::ones_like(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "ones_like", 1, 1);
  }},
  {"orgqr-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("orgqr");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::orgqr(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "orgqr", 2, 1);
  }},
  {"ormqr-3-left_i-transpose_i", [](Node *node) {
    auto left = bool(node->i(Symbol::attr("left")));
    auto transpose = bool(node->i(Symbol::attr("transpose")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ormqr");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::ormqr(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), left, transpose);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "ormqr", 3, 1);
  }},
  {"ormqr-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("ormqr");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto left = tensor_as<bool>(std::move(peek(stack, 3, 5)));
      auto transpose = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::ormqr(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), left, transpose);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "ormqr", 5, 1);
  }},
  {"pairwise_distance-2-eps_f-keepdim_i-p_f", [](Node *node) {
    auto p = double(node->f(Symbol::attr("p")));
    auto eps = double(node->f(Symbol::attr("eps")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("pairwise_distance");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::pairwise_distance(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), p, eps, keepdim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "pairwise_distance", 2, 1);
  }},
  {"pairwise_distance-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("pairwise_distance");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto p = tensor_as<double>(std::move(peek(stack, 2, 5)));
      auto eps = tensor_as<double>(std::move(peek(stack, 3, 5)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::pairwise_distance(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), p, eps, keepdim);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "pairwise_distance", 5, 1);
  }},
  {"permute-1-dims_is", [](Node *node) {
    auto dims = std::vector<int64_t>(node->is(Symbol::attr("dims")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("permute");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).permute(dims);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "permute", 1, 1);
  }},
  {"permute-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("permute");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto dims = tensor_as<IntList>(std::move(peek(stack, 1, 2)));
      auto result = (std::move(peek(stack, 0, 2))).permute(dims);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "permute", 2, 1);
  }},
  {"pin_memory-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("pin_memory");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::pin_memory(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "pin_memory", 1, 1);
  }},
  {"polygamma-1-n_i", [](Node *node) {
    auto n = int64_t(node->i(Symbol::attr("n")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("polygamma");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::polygamma(n, std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "polygamma", 1, 1);
  }},
  {"polygamma-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("polygamma");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto n = tensor_as<int64_t>(std::move(peek(stack, 0, 2)));
      auto result = at::polygamma(n, std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "polygamma", 2, 1);
  }},
  {"potrf-1-upper_i", [](Node *node) {
    auto upper = bool(node->i(Symbol::attr("upper")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("potrf");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::potrf(std::move(peek(stack, 0, 1)), upper);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "potrf", 1, 1);
  }},
  {"potrf-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("potrf");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto upper = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::potrf(std::move(peek(stack, 0, 2)), upper);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "potrf", 2, 1);
  }},
  {"potri-1-upper_i", [](Node *node) {
    auto upper = bool(node->i(Symbol::attr("upper")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("potri");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::potri(std::move(peek(stack, 0, 1)), upper);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "potri", 1, 1);
  }},
  {"potri-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("potri");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto upper = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::potri(std::move(peek(stack, 0, 2)), upper);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "potri", 2, 1);
  }},
  {"potrs-2-upper_i", [](Node *node) {
    auto upper = bool(node->i(Symbol::attr("upper")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("potrs");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::potrs(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), upper);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "potrs", 2, 1);
  }},
  {"potrs-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("potrs");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto upper = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::potrs(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), upper);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "potrs", 3, 1);
  }},
  {"pow-1-base_t", [](Node *node) {
    auto base = Scalar(node->t(Symbol::attr("base")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("pow");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::pow(base, std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "pow", 1, 1);
  }},
  {"pow-1-exponent_t", [](Node *node) {
    auto exponent = Scalar(node->t(Symbol::attr("exponent")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("pow");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::pow(std::move(peek(stack, 0, 1)), exponent);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "pow", 1, 1);
  }},
  {"pow-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("pow");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::pow(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "pow", 2, 1);
  }},
  {"prelu-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("prelu");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::prelu(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "prelu", 2, 1);
  }},
  {"prelu_backward-3-output_mask_is", [](Node *node) {
    auto output_mask = as_bool_array<2>(node->is(Symbol::attr("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("prelu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::prelu_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), output_mask);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "prelu_backward", 3, 2);
  }},
  {"prelu_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("prelu_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto output_mask = tensor_as<std::array<bool,2>>(std::move(peek(stack, 3, 4)));
      auto result = at::prelu_backward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), std::move(peek(stack, 2, 4)), output_mask);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "prelu_backward", 4, 2);
  }},
  {"prelu_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("prelu_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::prelu_forward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "prelu_forward", 2, 1);
  }},
  {"prod-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("prod");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::prod(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "prod", 1, 1);
  }},
  {"prod-1-dim_i-keepdim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("prod");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::prod(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "prod", 1, 1);
  }},
  {"prod-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("prod");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::prod(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "prod", 3, 1);
  }},
  {"pstrf-1-tol_t-upper_i", [](Node *node) {
    auto upper = bool(node->i(Symbol::attr("upper")));
    auto tol = Scalar(node->t(Symbol::attr("tol")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("pstrf");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::pstrf(std::move(peek(stack, 0, 1)), upper, tol);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "pstrf", 1, 2);
  }},
  {"pstrf-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("pstrf");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto upper = tensor_as<bool>(std::move(peek(stack, 1, 3)));
      auto tol = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::pstrf(std::move(peek(stack, 0, 3)), upper, tol);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "pstrf", 3, 2);
  }},
  {"qr-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("qr");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::qr(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "qr", 1, 2);
  }},
  {"rand_like-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("rand_like");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::rand_like(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "rand_like", 1, 1);
  }},
  {"randint_like-1-high_i", [](Node *node) {
    auto high = int64_t(node->i(Symbol::attr("high")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("randint_like");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::randint_like(std::move(peek(stack, 0, 1)), high);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "randint_like", 1, 1);
  }},
  {"randint_like-1-high_i-low_i", [](Node *node) {
    auto low = int64_t(node->i(Symbol::attr("low")));
    auto high = int64_t(node->i(Symbol::attr("high")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("randint_like");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::randint_like(std::move(peek(stack, 0, 1)), low, high);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "randint_like", 1, 1);
  }},
  {"randint_like-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("randint_like");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto high = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::randint_like(std::move(peek(stack, 0, 2)), high);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "randint_like", 2, 1);
  }},
  {"randint_like-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("randint_like");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto low = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto high = tensor_as<int64_t>(std::move(peek(stack, 2, 3)));
      auto result = at::randint_like(std::move(peek(stack, 0, 3)), low, high);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "randint_like", 3, 1);
  }},
  {"randn_like-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("randn_like");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::randn_like(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "randn_like", 1, 1);
  }},
  {"reciprocal-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reciprocal");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::reciprocal(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "reciprocal", 1, 1);
  }},
  {"reflection_pad1d-1-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad1d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::reflection_pad1d(std::move(peek(stack, 0, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad1d", 1, 1);
  }},
  {"reflection_pad1d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad1d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto padding_tensor = peek(stack, 1, 2);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::reflection_pad1d(std::move(peek(stack, 0, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad1d", 2, 1);
  }},
  {"reflection_pad1d_backward-2-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::reflection_pad1d_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad1d_backward", 2, 1);
  }},
  {"reflection_pad1d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto padding_tensor = peek(stack, 2, 3);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::reflection_pad1d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad1d_backward", 3, 1);
  }},
  {"reflection_pad1d_forward-1-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::reflection_pad1d_forward(std::move(peek(stack, 0, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad1d_forward", 1, 1);
  }},
  {"reflection_pad1d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto padding_tensor = peek(stack, 1, 2);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::reflection_pad1d_forward(std::move(peek(stack, 0, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad1d_forward", 2, 1);
  }},
  {"reflection_pad2d-1-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad2d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::reflection_pad2d(std::move(peek(stack, 0, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad2d", 1, 1);
  }},
  {"reflection_pad2d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad2d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto padding_tensor = peek(stack, 1, 2);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(4);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::reflection_pad2d(std::move(peek(stack, 0, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad2d", 2, 1);
  }},
  {"reflection_pad2d_backward-2-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::reflection_pad2d_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad2d_backward", 2, 1);
  }},
  {"reflection_pad2d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto padding_tensor = peek(stack, 2, 3);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(4);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::reflection_pad2d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad2d_backward", 3, 1);
  }},
  {"reflection_pad2d_forward-1-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::reflection_pad2d_forward(std::move(peek(stack, 0, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad2d_forward", 1, 1);
  }},
  {"reflection_pad2d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reflection_pad2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto padding_tensor = peek(stack, 1, 2);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(4);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::reflection_pad2d_forward(std::move(peek(stack, 0, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "reflection_pad2d_forward", 2, 1);
  }},
  {"relu-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("relu");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::relu(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "relu", 1, 1);
  }},
  {"remainder-1-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("remainder");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::remainder(std::move(peek(stack, 0, 1)), other);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "remainder", 1, 1);
  }},
  {"remainder-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("remainder");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::remainder(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "remainder", 2, 1);
  }},
  {"renorm-1-dim_i-maxnorm_t-p_t", [](Node *node) {
    auto p = Scalar(node->t(Symbol::attr("p")));
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto maxnorm = Scalar(node->t(Symbol::attr("maxnorm")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("renorm");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::renorm(std::move(peek(stack, 0, 1)), p, dim, maxnorm);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "renorm", 1, 1);
  }},
  {"renorm-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("renorm");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto p = tensor_as<Scalar>(std::move(peek(stack, 1, 4)));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 2, 4)));
      auto maxnorm = tensor_as<Scalar>(std::move(peek(stack, 3, 4)));
      auto result = at::renorm(std::move(peek(stack, 0, 4)), p, dim, maxnorm);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "renorm", 4, 1);
  }},
  {"repeat-1-repeats_is", [](Node *node) {
    auto repeats = std::vector<int64_t>(node->is(Symbol::attr("repeats")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("repeat");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).repeat(repeats);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "repeat", 1, 1);
  }},
  {"repeat-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("repeat");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto repeats = tensor_as<IntList>(std::move(peek(stack, 1, 2)));
      auto result = (std::move(peek(stack, 0, 2))).repeat(repeats);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "repeat", 2, 1);
  }},
  {"replication_pad1d-1-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad1d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::replication_pad1d(std::move(peek(stack, 0, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad1d", 1, 1);
  }},
  {"replication_pad1d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad1d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto padding_tensor = peek(stack, 1, 2);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::replication_pad1d(std::move(peek(stack, 0, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad1d", 2, 1);
  }},
  {"replication_pad1d_backward-2-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::replication_pad1d_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad1d_backward", 2, 1);
  }},
  {"replication_pad1d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto padding_tensor = peek(stack, 2, 3);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::replication_pad1d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad1d_backward", 3, 1);
  }},
  {"replication_pad1d_forward-1-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::replication_pad1d_forward(std::move(peek(stack, 0, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad1d_forward", 1, 1);
  }},
  {"replication_pad1d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto padding_tensor = peek(stack, 1, 2);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::replication_pad1d_forward(std::move(peek(stack, 0, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad1d_forward", 2, 1);
  }},
  {"replication_pad2d-1-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad2d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::replication_pad2d(std::move(peek(stack, 0, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad2d", 1, 1);
  }},
  {"replication_pad2d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad2d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto padding_tensor = peek(stack, 1, 2);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(4);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::replication_pad2d(std::move(peek(stack, 0, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad2d", 2, 1);
  }},
  {"replication_pad2d_backward-2-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::replication_pad2d_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad2d_backward", 2, 1);
  }},
  {"replication_pad2d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto padding_tensor = peek(stack, 2, 3);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(4);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::replication_pad2d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad2d_backward", 3, 1);
  }},
  {"replication_pad2d_forward-1-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::replication_pad2d_forward(std::move(peek(stack, 0, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad2d_forward", 1, 1);
  }},
  {"replication_pad2d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto padding_tensor = peek(stack, 1, 2);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(4);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::replication_pad2d_forward(std::move(peek(stack, 0, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad2d_forward", 2, 1);
  }},
  {"replication_pad3d-1-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad3d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::replication_pad3d(std::move(peek(stack, 0, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad3d", 1, 1);
  }},
  {"replication_pad3d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad3d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto padding_tensor = peek(stack, 1, 2);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(6);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::replication_pad3d(std::move(peek(stack, 0, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad3d", 2, 1);
  }},
  {"replication_pad3d_backward-2-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::replication_pad3d_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad3d_backward", 2, 1);
  }},
  {"replication_pad3d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto padding_tensor = peek(stack, 2, 3);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(6);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::replication_pad3d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad3d_backward", 3, 1);
  }},
  {"replication_pad3d_forward-1-padding_is", [](Node *node) {
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::replication_pad3d_forward(std::move(peek(stack, 0, 1)), padding);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad3d_forward", 1, 1);
  }},
  {"replication_pad3d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("replication_pad3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto padding_tensor = peek(stack, 1, 2);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(6);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::replication_pad3d_forward(std::move(peek(stack, 0, 2)), padding);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "replication_pad3d_forward", 2, 1);
  }},
  {"reshape-1-shape_is", [](Node *node) {
    auto shape = std::vector<int64_t>(node->is(Symbol::attr("shape")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reshape");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::reshape(std::move(peek(stack, 0, 1)), shape);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "reshape", 1, 1);
  }},
  {"reshape-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("reshape");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto shape = tensor_as<IntList>(std::move(peek(stack, 1, 2)));
      auto result = at::reshape(std::move(peek(stack, 0, 2)), shape);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "reshape", 2, 1);
  }},
  {"rfft-1-normalized_i-onesided_i-signal_ndim_i", [](Node *node) {
    auto signal_ndim = int64_t(node->i(Symbol::attr("signal_ndim")));
    auto normalized = bool(node->i(Symbol::attr("normalized")));
    auto onesided = bool(node->i(Symbol::attr("onesided")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("rfft");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::rfft(std::move(peek(stack, 0, 1)), signal_ndim, normalized, onesided);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "rfft", 1, 1);
  }},
  {"rfft-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("rfft");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto signal_ndim = tensor_as<int64_t>(std::move(peek(stack, 1, 4)));
      auto normalized = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto onesided = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::rfft(std::move(peek(stack, 0, 4)), signal_ndim, normalized, onesided);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "rfft", 4, 1);
  }},
  {"round-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("round");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::round(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "round", 1, 1);
  }},
  {"rrelu_with_noise_backward-3-lower_t-training_i-upper_t", [](Node *node) {
    auto lower = Scalar(node->t(Symbol::attr("lower")));
    auto upper = Scalar(node->t(Symbol::attr("upper")));
    auto training = bool(node->i(Symbol::attr("training")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("rrelu_with_noise_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::rrelu_with_noise_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), lower, upper, training);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "rrelu_with_noise_backward", 3, 1);
  }},
  {"rrelu_with_noise_backward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("rrelu_with_noise_backward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto lower = tensor_as<Scalar>(std::move(peek(stack, 3, 6)));
      auto upper = tensor_as<Scalar>(std::move(peek(stack, 4, 6)));
      auto training = tensor_as<bool>(std::move(peek(stack, 5, 6)));
      auto result = at::rrelu_with_noise_backward(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), std::move(peek(stack, 2, 6)), lower, upper, training);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "rrelu_with_noise_backward", 6, 1);
  }},
  {"rsqrt-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("rsqrt");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::rsqrt(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "rsqrt", 1, 1);
  }},
  {"select-1-dim_i-index_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto index = int64_t(node->i(Symbol::attr("index")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("select");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::select(std::move(peek(stack, 0, 1)), dim, index);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "select", 1, 1);
  }},
  {"select-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("select");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto index = tensor_as<int64_t>(std::move(peek(stack, 2, 3)));
      auto result = at::select(std::move(peek(stack, 0, 3)), dim, index);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "select", 3, 1);
  }},
  {"selu-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("selu");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::selu(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "selu", 1, 1);
  }},
  {"sigmoid-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sigmoid");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::sigmoid(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sigmoid", 1, 1);
  }},
  {"sign-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sign");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::sign(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sign", 1, 1);
  }},
  {"sin-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sin");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::sin(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sin", 1, 1);
  }},
  {"sinh-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sinh");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::sinh(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sinh", 1, 1);
  }},
  {"size-1-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("size");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::size(std::move(peek(stack, 0, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "size", 1, 1);
  }},
  {"size-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("size");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::size(std::move(peek(stack, 0, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "size", 2, 1);
  }},
  {"sizes-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sizes");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).sizes();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sizes", 1, 1);
  }},
  {"slice-1-dim_i-end_i-start_i-step_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto start = int64_t(node->i(Symbol::attr("start")));
    auto end = int64_t(node->i(Symbol::attr("end")));
    auto step = int64_t(node->i(Symbol::attr("step")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("slice");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::slice(std::move(peek(stack, 0, 1)), dim, start, end, step);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "slice", 1, 1);
  }},
  {"slice-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("slice");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 5)));
      auto start = tensor_as<int64_t>(std::move(peek(stack, 2, 5)));
      auto end = tensor_as<int64_t>(std::move(peek(stack, 3, 5)));
      auto step = tensor_as<int64_t>(std::move(peek(stack, 4, 5)));
      auto result = at::slice(std::move(peek(stack, 0, 5)), dim, start, end, step);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "slice", 5, 1);
  }},
  {"slogdet-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("slogdet");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::slogdet(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "slogdet", 1, 2);
  }},
  {"smm-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("smm");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::smm(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "smm", 2, 1);
  }},
  {"smooth_l1_loss-2-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("smooth_l1_loss");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::smooth_l1_loss(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "smooth_l1_loss", 2, 1);
  }},
  {"smooth_l1_loss-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("smooth_l1_loss");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::smooth_l1_loss(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "smooth_l1_loss", 4, 1);
  }},
  {"smooth_l1_loss_backward-3-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("smooth_l1_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::smooth_l1_loss_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "smooth_l1_loss_backward", 3, 1);
  }},
  {"smooth_l1_loss_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("smooth_l1_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 3, 5)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::smooth_l1_loss_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "smooth_l1_loss_backward", 5, 1);
  }},
  {"smooth_l1_loss_forward-2-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("smooth_l1_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::smooth_l1_loss_forward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "smooth_l1_loss_forward", 2, 1);
  }},
  {"smooth_l1_loss_forward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("smooth_l1_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::smooth_l1_loss_forward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "smooth_l1_loss_forward", 4, 1);
  }},
  {"soft_margin_loss-2-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("soft_margin_loss");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::soft_margin_loss(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "soft_margin_loss", 2, 1);
  }},
  {"soft_margin_loss-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("soft_margin_loss");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::soft_margin_loss(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "soft_margin_loss", 4, 1);
  }},
  {"soft_margin_loss_backward-3-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("soft_margin_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::soft_margin_loss_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "soft_margin_loss_backward", 3, 1);
  }},
  {"soft_margin_loss_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("soft_margin_loss_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 3, 5)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::soft_margin_loss_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), size_average, reduce);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "soft_margin_loss_backward", 5, 1);
  }},
  {"soft_margin_loss_forward-2-reduce_i-size_average_i", [](Node *node) {
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("soft_margin_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::soft_margin_loss_forward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), size_average, reduce);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "soft_margin_loss_forward", 2, 1);
  }},
  {"soft_margin_loss_forward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("soft_margin_loss_forward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::soft_margin_loss_forward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), size_average, reduce);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "soft_margin_loss_forward", 4, 1);
  }},
  {"softmax-1-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softmax");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::softmax(std::move(peek(stack, 0, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "softmax", 1, 1);
  }},
  {"softmax-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softmax");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::softmax(std::move(peek(stack, 0, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "softmax", 2, 1);
  }},
  {"softmax_backward_data-3-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softmax_backward_data");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::softmax_backward_data(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), dim, std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "softmax_backward_data", 3, 1);
  }},
  {"softmax_backward_data-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softmax_backward_data");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 2, 4)));
      auto result = at::softmax_backward_data(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), dim, std::move(peek(stack, 3, 4)));
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "softmax_backward_data", 4, 1);
  }},
  {"softplus-1-beta_t-threshold_t", [](Node *node) {
    auto beta = Scalar(node->t(Symbol::attr("beta")));
    auto threshold = Scalar(node->t(Symbol::attr("threshold")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softplus");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::softplus(std::move(peek(stack, 0, 1)), beta, threshold);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "softplus", 1, 1);
  }},
  {"softplus-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softplus");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto beta = tensor_as<Scalar>(std::move(peek(stack, 1, 3)));
      auto threshold = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::softplus(std::move(peek(stack, 0, 3)), beta, threshold);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "softplus", 3, 1);
  }},
  {"softplus_backward-3-beta_t-threshold_t", [](Node *node) {
    auto beta = Scalar(node->t(Symbol::attr("beta")));
    auto threshold = Scalar(node->t(Symbol::attr("threshold")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softplus_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::softplus_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), beta, threshold, std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "softplus_backward", 3, 1);
  }},
  {"softplus_backward-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softplus_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto beta = tensor_as<Scalar>(std::move(peek(stack, 2, 5)));
      auto threshold = tensor_as<Scalar>(std::move(peek(stack, 3, 5)));
      auto result = at::softplus_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), beta, threshold, std::move(peek(stack, 4, 5)));
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "softplus_backward", 5, 1);
  }},
  {"softplus_forward-1-beta_t-threshold_t", [](Node *node) {
    auto beta = Scalar(node->t(Symbol::attr("beta")));
    auto threshold = Scalar(node->t(Symbol::attr("threshold")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softplus_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::softplus_forward(std::move(peek(stack, 0, 1)), beta, threshold);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "softplus_forward", 1, 1);
  }},
  {"softplus_forward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softplus_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto beta = tensor_as<Scalar>(std::move(peek(stack, 1, 3)));
      auto threshold = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::softplus_forward(std::move(peek(stack, 0, 3)), beta, threshold);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "softplus_forward", 3, 1);
  }},
  {"softshrink-1-lambd_t", [](Node *node) {
    auto lambd = Scalar(node->t(Symbol::attr("lambd")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softshrink");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::softshrink(std::move(peek(stack, 0, 1)), lambd);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "softshrink", 1, 1);
  }},
  {"softshrink-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softshrink");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto lambd = tensor_as<Scalar>(std::move(peek(stack, 1, 2)));
      auto result = at::softshrink(std::move(peek(stack, 0, 2)), lambd);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "softshrink", 2, 1);
  }},
  {"softshrink_backward-2-lambd_t", [](Node *node) {
    auto lambd = Scalar(node->t(Symbol::attr("lambd")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softshrink_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::softshrink_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), lambd);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "softshrink_backward", 2, 1);
  }},
  {"softshrink_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softshrink_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto lambd = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::softshrink_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), lambd);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "softshrink_backward", 3, 1);
  }},
  {"softshrink_forward-1-lambd_t", [](Node *node) {
    auto lambd = Scalar(node->t(Symbol::attr("lambd")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softshrink_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::softshrink_forward(std::move(peek(stack, 0, 1)), lambd);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "softshrink_forward", 1, 1);
  }},
  {"softshrink_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("softshrink_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto lambd = tensor_as<Scalar>(std::move(peek(stack, 1, 2)));
      auto result = at::softshrink_forward(std::move(peek(stack, 0, 2)), lambd);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "softshrink_forward", 2, 1);
  }},
  {"sort-1-descending_i-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto descending = bool(node->i(Symbol::attr("descending")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sort");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::sort(std::move(peek(stack, 0, 1)), dim, descending);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sort", 1, 2);
  }},
  {"sort-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sort");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto descending = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::sort(std::move(peek(stack, 0, 3)), dim, descending);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "sort", 3, 2);
  }},
  {"sparse_coo_tensor-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sparse_coo_tensor");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::sparse_coo_tensor(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "sparse_coo_tensor", 2, 1);
  }},
  {"sparse_coo_tensor-2-size_is", [](Node *node) {
    auto size = std::vector<int64_t>(node->is(Symbol::attr("size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sparse_coo_tensor");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::sparse_coo_tensor(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "sparse_coo_tensor", 2, 1);
  }},
  {"sparse_coo_tensor-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sparse_coo_tensor");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto size = tensor_as<IntList>(std::move(peek(stack, 2, 3)));
      auto result = at::sparse_coo_tensor(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), size);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "sparse_coo_tensor", 3, 1);
  }},
  {"split-1-dim_i-split_size_i", [](Node *node) {
    auto split_size = int64_t(node->i(Symbol::attr("split_size")));
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("split");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::split(std::move(peek(stack, 0, 1)), split_size, dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "split", 1, UNKNOWN_OUTPUTS);
  }},
  {"split-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("split");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto split_size = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 2, 3)));
      auto result = at::split(std::move(peek(stack, 0, 3)), split_size, dim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "split", 3, UNKNOWN_OUTPUTS);
  }},
  {"split_with_sizes-1-dim_i-split_sizes_is", [](Node *node) {
    auto split_sizes = std::vector<int64_t>(node->is(Symbol::attr("split_sizes")));
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("split_with_sizes");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::split_with_sizes(std::move(peek(stack, 0, 1)), split_sizes, dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "split_with_sizes", 1, UNKNOWN_OUTPUTS);
  }},
  {"split_with_sizes-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("split_with_sizes");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto split_sizes = tensor_as<IntList>(std::move(peek(stack, 1, 3)));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 2, 3)));
      auto result = at::split_with_sizes(std::move(peek(stack, 0, 3)), split_sizes, dim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "split_with_sizes", 3, UNKNOWN_OUTPUTS);
  }},
  {"sqrt-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sqrt");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::sqrt(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sqrt", 1, 1);
  }},
  {"squeeze-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("squeeze");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::squeeze(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "squeeze", 1, 1);
  }},
  {"squeeze-1-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("squeeze");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::squeeze(std::move(peek(stack, 0, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "squeeze", 1, 1);
  }},
  {"squeeze-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("squeeze");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::squeeze(std::move(peek(stack, 0, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "squeeze", 2, 1);
  }},
  {"sspaddmm-3-alpha_t-beta_t", [](Node *node) {
    auto beta = Scalar(node->t(Symbol::attr("beta")));
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sspaddmm");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::sspaddmm(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), beta, alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "sspaddmm", 3, 1);
  }},
  {"sspaddmm-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sspaddmm");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto beta = tensor_as<Scalar>(std::move(peek(stack, 3, 5)));
      auto alpha = tensor_as<Scalar>(std::move(peek(stack, 4, 5)));
      auto result = at::sspaddmm(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), beta, alpha);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "sspaddmm", 5, 1);
  }},
  {"stack-*", [](Node *node) {
    size_t varargs_length = node->inputs().size();
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("stack");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 0, 1)));
      auto result = at::stack(peekSlice(stack, 0, varargs_length - 1, varargs_length), dim);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "stack", varargs_length, 1);
  }},
  {"stack-*-dim_i", [](Node *node) {
    size_t varargs_length = node->inputs().size();
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("stack");
      AutoGPU device_guard(deviceForInputs(stack, varargs_length));
  
      auto result = at::stack(peekSlice(stack, 0, varargs_length - 0, varargs_length), dim);
      drop(stack, varargs_length);
      pack(stack, std::move(result));
      return 0;
    }, "stack", varargs_length, 1);
  }},
  {"std-1-dim_i-keepdim_i-unbiased_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto unbiased = bool(node->i(Symbol::attr("unbiased")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("std");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::std(std::move(peek(stack, 0, 1)), dim, unbiased, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "std", 1, 1);
  }},
  {"std-1-unbiased_i", [](Node *node) {
    auto unbiased = bool(node->i(Symbol::attr("unbiased")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("std");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::std(std::move(peek(stack, 0, 1)), unbiased);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "std", 1, 1);
  }},
  {"std-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("std");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto unbiased = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::std(std::move(peek(stack, 0, 2)), unbiased);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "std", 2, 1);
  }},
  {"std-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("std");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 4)));
      auto unbiased = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::std(std::move(peek(stack, 0, 4)), dim, unbiased, keepdim);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "std", 4, 1);
  }},
  {"stft-2-fft_size_i-frame_length_i-hop_i-normalized_i-onesided_i-pad_end_i", [](Node *node) {
    auto frame_length = int64_t(node->i(Symbol::attr("frame_length")));
    auto hop = int64_t(node->i(Symbol::attr("hop")));
    auto fft_size = int64_t(node->i(Symbol::attr("fft_size")));
    auto normalized = bool(node->i(Symbol::attr("normalized")));
    auto onesided = bool(node->i(Symbol::attr("onesided")));
    auto pad_end = int64_t(node->i(Symbol::attr("pad_end")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("stft");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::stft(std::move(peek(stack, 0, 2)), frame_length, hop, fft_size, normalized, onesided, std::move(peek(stack, 1, 2)), pad_end);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "stft", 2, 1);
  }},
  {"stft-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("stft");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto frame_length = tensor_as<int64_t>(std::move(peek(stack, 1, 8)));
      auto hop = tensor_as<int64_t>(std::move(peek(stack, 2, 8)));
      auto fft_size = tensor_as<int64_t>(std::move(peek(stack, 3, 8)));
      auto normalized = tensor_as<bool>(std::move(peek(stack, 4, 8)));
      auto onesided = tensor_as<bool>(std::move(peek(stack, 5, 8)));
      auto pad_end = tensor_as<int64_t>(std::move(peek(stack, 7, 8)));
      auto result = at::stft(std::move(peek(stack, 0, 8)), frame_length, hop, fft_size, normalized, onesided, std::move(peek(stack, 6, 8)), pad_end);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "stft", 8, 1);
  }},
  {"storage_offset-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("storage_offset");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).storage_offset();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "storage_offset", 1, 1);
  }},
  {"stride-1-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("stride");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::stride(std::move(peek(stack, 0, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "stride", 1, 1);
  }},
  {"stride-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("stride");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::stride(std::move(peek(stack, 0, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "stride", 2, 1);
  }},
  {"strides-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("strides");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).strides();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "strides", 1, 1);
  }},
  {"sub-1-alpha_t-other_t", [](Node *node) {
    auto other = Scalar(node->t(Symbol::attr("other")));
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sub");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::sub(std::move(peek(stack, 0, 1)), other, alpha);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sub", 1, 1);
  }},
  {"sub-2-alpha_t", [](Node *node) {
    auto alpha = Scalar(node->t(Symbol::attr("alpha")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sub");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::sub(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), alpha);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "sub", 2, 1);
  }},
  {"sub-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sub");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto alpha = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::sub(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), alpha);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "sub", 3, 1);
  }},
  {"sum-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sum");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::sum(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sum", 1, 1);
  }},
  {"sum-1-dim_is-keepdim_i", [](Node *node) {
    auto dim = std::vector<int64_t>(node->is(Symbol::attr("dim")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sum");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::sum(std::move(peek(stack, 0, 1)), dim, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "sum", 1, 1);
  }},
  {"sum-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("sum");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim_tensor = peek(stack, 1, 3);
      if (dim_tensor.dim() == 0)
          dim_tensor = dim_tensor.expand(1);
      auto dim = tensor_as<at::IntList>(std::move(dim_tensor));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::sum(std::move(peek(stack, 0, 3)), dim, keepdim);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "sum", 3, 1);
  }},
  {"svd-1-some_i", [](Node *node) {
    auto some = bool(node->i(Symbol::attr("some")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("svd");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::svd(std::move(peek(stack, 0, 1)), some);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "svd", 1, 3);
  }},
  {"svd-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("svd");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto some = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::svd(std::move(peek(stack, 0, 2)), some);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "svd", 2, 3);
  }},
  {"symeig-1-eigenvectors_i-upper_i", [](Node *node) {
    auto eigenvectors = bool(node->i(Symbol::attr("eigenvectors")));
    auto upper = bool(node->i(Symbol::attr("upper")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("symeig");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::symeig(std::move(peek(stack, 0, 1)), eigenvectors, upper);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "symeig", 1, 2);
  }},
  {"symeig-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("symeig");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto eigenvectors = tensor_as<bool>(std::move(peek(stack, 1, 3)));
      auto upper = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::symeig(std::move(peek(stack, 0, 3)), eigenvectors, upper);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "symeig", 3, 2);
  }},
  {"t-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("t");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::t(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "t", 1, 1);
  }},
  {"take-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("take");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::take(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "take", 2, 1);
  }},
  {"tan-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("tan");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::tan(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "tan", 1, 1);
  }},
  {"tanh-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("tanh");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::tanh(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "tanh", 1, 1);
  }},
  {"thnn_batch_norm-5-eps_f-momentum_f-training_i", [](Node *node) {
    auto training = bool(node->i(Symbol::attr("training")));
    auto momentum = double(node->f(Symbol::attr("momentum")));
    auto eps = double(node->f(Symbol::attr("eps")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_batch_norm");
      AutoGPU device_guard(deviceForInputs(stack, 5));
  
      auto result = at::thnn_batch_norm(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), std::move(peek(stack, 3, 5)), std::move(peek(stack, 4, 5)), training, momentum, eps);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_batch_norm", 5, 1);
  }},
  {"thnn_batch_norm-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_batch_norm");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto training = tensor_as<bool>(std::move(peek(stack, 5, 8)));
      auto momentum = tensor_as<double>(std::move(peek(stack, 6, 8)));
      auto eps = tensor_as<double>(std::move(peek(stack, 7, 8)));
      auto result = at::thnn_batch_norm(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), std::move(peek(stack, 2, 8)), std::move(peek(stack, 3, 8)), std::move(peek(stack, 4, 8)), training, momentum, eps);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_batch_norm", 8, 1);
  }},
  {"thnn_batch_norm_backward-10", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_batch_norm_backward");
      AutoGPU device_guard(deviceForInputs(stack, 10));
      auto training = tensor_as<bool>(std::move(peek(stack, 5, 10)));
      auto eps = tensor_as<double>(std::move(peek(stack, 6, 10)));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(peek(stack, 9, 10)));
      auto result = at::thnn_batch_norm_backward(std::move(peek(stack, 0, 10)), std::move(peek(stack, 1, 10)), std::move(peek(stack, 2, 10)), std::move(peek(stack, 3, 10)), std::move(peek(stack, 4, 10)), training, eps, std::move(peek(stack, 7, 10)), std::move(peek(stack, 8, 10)), output_mask);
      drop(stack, 10);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_batch_norm_backward", 10, 3);
  }},
  {"thnn_batch_norm_backward-7-eps_f-output_mask_is-training_i", [](Node *node) {
    auto training = bool(node->i(Symbol::attr("training")));
    auto eps = double(node->f(Symbol::attr("eps")));
    auto output_mask = as_bool_array<3>(node->is(Symbol::attr("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_batch_norm_backward");
      AutoGPU device_guard(deviceForInputs(stack, 7));
  
      auto result = at::thnn_batch_norm_backward(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), std::move(peek(stack, 2, 7)), std::move(peek(stack, 3, 7)), std::move(peek(stack, 4, 7)), training, eps, std::move(peek(stack, 5, 7)), std::move(peek(stack, 6, 7)), output_mask);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_batch_norm_backward", 7, 3);
  }},
  {"thnn_batch_norm_forward-5-eps_f-momentum_f-training_i", [](Node *node) {
    auto training = bool(node->i(Symbol::attr("training")));
    auto momentum = double(node->f(Symbol::attr("momentum")));
    auto eps = double(node->f(Symbol::attr("eps")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_batch_norm_forward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
  
      auto result = at::thnn_batch_norm_forward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), std::move(peek(stack, 3, 5)), std::move(peek(stack, 4, 5)), training, momentum, eps);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_batch_norm_forward", 5, 3);
  }},
  {"thnn_batch_norm_forward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_batch_norm_forward");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto training = tensor_as<bool>(std::move(peek(stack, 5, 8)));
      auto momentum = tensor_as<double>(std::move(peek(stack, 6, 8)));
      auto eps = tensor_as<double>(std::move(peek(stack, 7, 8)));
      auto result = at::thnn_batch_norm_forward(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), std::move(peek(stack, 2, 8)), std::move(peek(stack, 3, 8)), std::move(peek(stack, 4, 8)), training, momentum, eps);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_batch_norm_forward", 8, 3);
  }},
  {"thnn_conv2d-3-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv2d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv2d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, std::move(peek(stack, 2, 3)), stride, padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv2d", 3, 1);
  }},
  {"thnn_conv2d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv2d");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto kernel_size_tensor = peek(stack, 2, 6);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 6);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 6);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::thnn_conv2d(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), kernel_size, std::move(peek(stack, 3, 6)), stride, padding);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv2d", 6, 1);
  }},
  {"thnn_conv2d_backward-5-kernel_size_is-output_mask_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto output_mask = as_bool_array<3>(node->is(Symbol::attr("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
  
      auto result = at::thnn_conv2d_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), kernel_size, stride, padding, std::move(peek(stack, 3, 5)), std::move(peek(stack, 4, 5)), output_mask);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv2d_backward", 5, 3);
  }},
  {"thnn_conv2d_backward-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 9));
      auto kernel_size_tensor = peek(stack, 3, 9);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 9);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 9);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(peek(stack, 8, 9)));
      auto result = at::thnn_conv2d_backward(std::move(peek(stack, 0, 9)), std::move(peek(stack, 1, 9)), std::move(peek(stack, 2, 9)), kernel_size, stride, padding, std::move(peek(stack, 6, 9)), std::move(peek(stack, 7, 9)), output_mask);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv2d_backward", 9, 3);
  }},
  {"thnn_conv2d_forward-3-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv2d_forward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, std::move(peek(stack, 2, 3)), stride, padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv2d_forward", 3, 3);
  }},
  {"thnn_conv2d_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto kernel_size_tensor = peek(stack, 2, 6);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 6);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 6);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::thnn_conv2d_forward(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), kernel_size, std::move(peek(stack, 3, 6)), stride, padding);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv2d_forward", 6, 3);
  }},
  {"thnn_conv3d-3-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv3d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv3d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, std::move(peek(stack, 2, 3)), stride, padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv3d", 3, 1);
  }},
  {"thnn_conv3d-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv3d");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto kernel_size_tensor = peek(stack, 2, 6);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 6);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 6);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::thnn_conv3d(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), kernel_size, std::move(peek(stack, 3, 6)), stride, padding);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv3d", 6, 1);
  }},
  {"thnn_conv3d_backward-5-kernel_size_is-output_mask_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto output_mask = as_bool_array<3>(node->is(Symbol::attr("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
  
      auto result = at::thnn_conv3d_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), kernel_size, stride, padding, std::move(peek(stack, 3, 5)), std::move(peek(stack, 4, 5)), output_mask);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv3d_backward", 5, 3);
  }},
  {"thnn_conv3d_backward-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 9));
      auto kernel_size_tensor = peek(stack, 3, 9);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 9);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 9);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(peek(stack, 8, 9)));
      auto result = at::thnn_conv3d_backward(std::move(peek(stack, 0, 9)), std::move(peek(stack, 1, 9)), std::move(peek(stack, 2, 9)), kernel_size, stride, padding, std::move(peek(stack, 6, 9)), std::move(peek(stack, 7, 9)), output_mask);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv3d_backward", 9, 3);
  }},
  {"thnn_conv3d_forward-3-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv3d_forward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, std::move(peek(stack, 2, 3)), stride, padding);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv3d_forward", 3, 3);
  }},
  {"thnn_conv3d_forward-6", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 6));
      auto kernel_size_tensor = peek(stack, 2, 6);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 6);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 6);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto result = at::thnn_conv3d_forward(std::move(peek(stack, 0, 6)), std::move(peek(stack, 1, 6)), kernel_size, std::move(peek(stack, 3, 6)), stride, padding);
      drop(stack, 6);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv3d_forward", 6, 3);
  }},
  {"thnn_conv_depthwise2d-3-dilation_is-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_depthwise2d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv_depthwise2d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, std::move(peek(stack, 2, 3)), stride, padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_depthwise2d", 3, 1);
  }},
  {"thnn_conv_depthwise2d-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_depthwise2d");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto kernel_size_tensor = peek(stack, 2, 7);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 7);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 7);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 6, 7);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(2);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto result = at::thnn_conv_depthwise2d(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), kernel_size, std::move(peek(stack, 3, 7)), stride, padding, dilation);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_depthwise2d", 7, 1);
  }},
  {"thnn_conv_depthwise2d_backward-3-dilation_is-kernel_size_is-output_mask_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto output_mask = as_bool_array<2>(node->is(Symbol::attr("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_depthwise2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv_depthwise2d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), kernel_size, stride, padding, dilation, output_mask);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_depthwise2d_backward", 3, 2);
  }},
  {"thnn_conv_depthwise2d_backward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_depthwise2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto kernel_size_tensor = peek(stack, 3, 8);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 8);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 8);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 6, 8);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(2);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto output_mask = tensor_as<std::array<bool,2>>(std::move(peek(stack, 7, 8)));
      auto result = at::thnn_conv_depthwise2d_backward(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), std::move(peek(stack, 2, 8)), kernel_size, stride, padding, dilation, output_mask);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_depthwise2d_backward", 8, 2);
  }},
  {"thnn_conv_depthwise2d_forward-3-dilation_is-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_depthwise2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv_depthwise2d_forward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, std::move(peek(stack, 2, 3)), stride, padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_depthwise2d_forward", 3, 1);
  }},
  {"thnn_conv_depthwise2d_forward-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_depthwise2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto kernel_size_tensor = peek(stack, 2, 7);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 7);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 7);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 6, 7);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(2);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto result = at::thnn_conv_depthwise2d_forward(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), kernel_size, std::move(peek(stack, 3, 7)), stride, padding, dilation);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_depthwise2d_forward", 7, 1);
  }},
  {"thnn_conv_dilated2d-3-dilation_is-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated2d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv_dilated2d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, std::move(peek(stack, 2, 3)), stride, padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated2d", 3, 1);
  }},
  {"thnn_conv_dilated2d-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated2d");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto kernel_size_tensor = peek(stack, 2, 7);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 7);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 7);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 6, 7);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(2);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto result = at::thnn_conv_dilated2d(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), kernel_size, std::move(peek(stack, 3, 7)), stride, padding, dilation);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated2d", 7, 1);
  }},
  {"thnn_conv_dilated2d_backward-10", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 10));
      auto kernel_size_tensor = peek(stack, 3, 10);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 10);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 10);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 6, 10);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(2);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(peek(stack, 9, 10)));
      auto result = at::thnn_conv_dilated2d_backward(std::move(peek(stack, 0, 10)), std::move(peek(stack, 1, 10)), std::move(peek(stack, 2, 10)), kernel_size, stride, padding, dilation, std::move(peek(stack, 7, 10)), std::move(peek(stack, 8, 10)), output_mask);
      drop(stack, 10);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated2d_backward", 10, 3);
  }},
  {"thnn_conv_dilated2d_backward-5-dilation_is-kernel_size_is-output_mask_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto output_mask = as_bool_array<3>(node->is(Symbol::attr("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
  
      auto result = at::thnn_conv_dilated2d_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), kernel_size, stride, padding, dilation, std::move(peek(stack, 3, 5)), std::move(peek(stack, 4, 5)), output_mask);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated2d_backward", 5, 3);
  }},
  {"thnn_conv_dilated2d_forward-3-dilation_is-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv_dilated2d_forward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, std::move(peek(stack, 2, 3)), stride, padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated2d_forward", 3, 3);
  }},
  {"thnn_conv_dilated2d_forward-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto kernel_size_tensor = peek(stack, 2, 7);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 7);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 7);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 6, 7);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(2);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto result = at::thnn_conv_dilated2d_forward(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), kernel_size, std::move(peek(stack, 3, 7)), stride, padding, dilation);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated2d_forward", 7, 3);
  }},
  {"thnn_conv_dilated3d-3-dilation_is-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated3d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv_dilated3d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, std::move(peek(stack, 2, 3)), stride, padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated3d", 3, 1);
  }},
  {"thnn_conv_dilated3d-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated3d");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto kernel_size_tensor = peek(stack, 2, 7);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 7);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 7);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 6, 7);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(3);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto result = at::thnn_conv_dilated3d(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), kernel_size, std::move(peek(stack, 3, 7)), stride, padding, dilation);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated3d", 7, 1);
  }},
  {"thnn_conv_dilated3d_backward-10", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 10));
      auto kernel_size_tensor = peek(stack, 3, 10);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 10);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 10);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 6, 10);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(3);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(peek(stack, 9, 10)));
      auto result = at::thnn_conv_dilated3d_backward(std::move(peek(stack, 0, 10)), std::move(peek(stack, 1, 10)), std::move(peek(stack, 2, 10)), kernel_size, stride, padding, dilation, std::move(peek(stack, 7, 10)), std::move(peek(stack, 8, 10)), output_mask);
      drop(stack, 10);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated3d_backward", 10, 3);
  }},
  {"thnn_conv_dilated3d_backward-5-dilation_is-kernel_size_is-output_mask_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto output_mask = as_bool_array<3>(node->is(Symbol::attr("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
  
      auto result = at::thnn_conv_dilated3d_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), kernel_size, stride, padding, dilation, std::move(peek(stack, 3, 5)), std::move(peek(stack, 4, 5)), output_mask);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated3d_backward", 5, 3);
  }},
  {"thnn_conv_dilated3d_forward-3-dilation_is-kernel_size_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv_dilated3d_forward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, std::move(peek(stack, 2, 3)), stride, padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated3d_forward", 3, 3);
  }},
  {"thnn_conv_dilated3d_forward-7", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_dilated3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 7));
      auto kernel_size_tensor = peek(stack, 2, 7);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 7);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 7);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto dilation_tensor = peek(stack, 6, 7);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(3);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto result = at::thnn_conv_dilated3d_forward(std::move(peek(stack, 0, 7)), std::move(peek(stack, 1, 7)), kernel_size, std::move(peek(stack, 3, 7)), stride, padding, dilation);
      drop(stack, 7);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_dilated3d_forward", 7, 3);
  }},
  {"thnn_conv_transpose2d-3-dilation_is-kernel_size_is-output_padding_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose2d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv_transpose2d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, std::move(peek(stack, 2, 3)), stride, padding, output_padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose2d", 3, 1);
  }},
  {"thnn_conv_transpose2d-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose2d");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto kernel_size_tensor = peek(stack, 2, 8);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 8);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 8);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto output_padding_tensor = peek(stack, 6, 8);
      if (output_padding_tensor.dim() == 0)
          output_padding_tensor = output_padding_tensor.expand(2);
      auto output_padding = tensor_as<at::IntList>(std::move(output_padding_tensor));
      auto dilation_tensor = peek(stack, 7, 8);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(2);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto result = at::thnn_conv_transpose2d(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), kernel_size, std::move(peek(stack, 3, 8)), stride, padding, output_padding, dilation);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose2d", 8, 1);
  }},
  {"thnn_conv_transpose2d_backward-11", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 11));
      auto kernel_size_tensor = peek(stack, 3, 11);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 11);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 11);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto output_padding_tensor = peek(stack, 6, 11);
      if (output_padding_tensor.dim() == 0)
          output_padding_tensor = output_padding_tensor.expand(2);
      auto output_padding = tensor_as<at::IntList>(std::move(output_padding_tensor));
      auto dilation_tensor = peek(stack, 7, 11);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(2);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(peek(stack, 10, 11)));
      auto result = at::thnn_conv_transpose2d_backward(std::move(peek(stack, 0, 11)), std::move(peek(stack, 1, 11)), std::move(peek(stack, 2, 11)), kernel_size, stride, padding, output_padding, dilation, std::move(peek(stack, 8, 11)), std::move(peek(stack, 9, 11)), output_mask);
      drop(stack, 11);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose2d_backward", 11, 3);
  }},
  {"thnn_conv_transpose2d_backward-5-dilation_is-kernel_size_is-output_mask_is-output_padding_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto output_mask = as_bool_array<3>(node->is(Symbol::attr("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
  
      auto result = at::thnn_conv_transpose2d_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), kernel_size, stride, padding, output_padding, dilation, std::move(peek(stack, 3, 5)), std::move(peek(stack, 4, 5)), output_mask);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose2d_backward", 5, 3);
  }},
  {"thnn_conv_transpose2d_forward-3-dilation_is-kernel_size_is-output_padding_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv_transpose2d_forward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, std::move(peek(stack, 2, 3)), stride, padding, output_padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose2d_forward", 3, 3);
  }},
  {"thnn_conv_transpose2d_forward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto kernel_size_tensor = peek(stack, 2, 8);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(2);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 8);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(2);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 8);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(2);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto output_padding_tensor = peek(stack, 6, 8);
      if (output_padding_tensor.dim() == 0)
          output_padding_tensor = output_padding_tensor.expand(2);
      auto output_padding = tensor_as<at::IntList>(std::move(output_padding_tensor));
      auto dilation_tensor = peek(stack, 7, 8);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(2);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto result = at::thnn_conv_transpose2d_forward(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), kernel_size, std::move(peek(stack, 3, 8)), stride, padding, output_padding, dilation);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose2d_forward", 8, 3);
  }},
  {"thnn_conv_transpose3d-3-dilation_is-kernel_size_is-output_padding_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose3d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv_transpose3d(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, std::move(peek(stack, 2, 3)), stride, padding, output_padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose3d", 3, 1);
  }},
  {"thnn_conv_transpose3d-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose3d");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto kernel_size_tensor = peek(stack, 2, 8);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 8);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 8);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto output_padding_tensor = peek(stack, 6, 8);
      if (output_padding_tensor.dim() == 0)
          output_padding_tensor = output_padding_tensor.expand(3);
      auto output_padding = tensor_as<at::IntList>(std::move(output_padding_tensor));
      auto dilation_tensor = peek(stack, 7, 8);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(3);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto result = at::thnn_conv_transpose3d(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), kernel_size, std::move(peek(stack, 3, 8)), stride, padding, output_padding, dilation);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose3d", 8, 1);
  }},
  {"thnn_conv_transpose3d_backward-11", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 11));
      auto kernel_size_tensor = peek(stack, 3, 11);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 11);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 11);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto output_padding_tensor = peek(stack, 6, 11);
      if (output_padding_tensor.dim() == 0)
          output_padding_tensor = output_padding_tensor.expand(3);
      auto output_padding = tensor_as<at::IntList>(std::move(output_padding_tensor));
      auto dilation_tensor = peek(stack, 7, 11);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(3);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto output_mask = tensor_as<std::array<bool,3>>(std::move(peek(stack, 10, 11)));
      auto result = at::thnn_conv_transpose3d_backward(std::move(peek(stack, 0, 11)), std::move(peek(stack, 1, 11)), std::move(peek(stack, 2, 11)), kernel_size, stride, padding, output_padding, dilation, std::move(peek(stack, 8, 11)), std::move(peek(stack, 9, 11)), output_mask);
      drop(stack, 11);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose3d_backward", 11, 3);
  }},
  {"thnn_conv_transpose3d_backward-5-dilation_is-kernel_size_is-output_mask_is-output_padding_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    auto output_mask = as_bool_array<3>(node->is(Symbol::attr("output_mask")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 5));
  
      auto result = at::thnn_conv_transpose3d_backward(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), std::move(peek(stack, 2, 5)), kernel_size, stride, padding, output_padding, dilation, std::move(peek(stack, 3, 5)), std::move(peek(stack, 4, 5)), output_mask);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose3d_backward", 5, 3);
  }},
  {"thnn_conv_transpose3d_forward-3-dilation_is-kernel_size_is-output_padding_is-padding_is-stride_is", [](Node *node) {
    auto kernel_size = std::vector<int64_t>(node->is(Symbol::attr("kernel_size")));
    auto stride = std::vector<int64_t>(node->is(Symbol::attr("stride")));
    auto padding = std::vector<int64_t>(node->is(Symbol::attr("padding")));
    auto output_padding = std::vector<int64_t>(node->is(Symbol::attr("output_padding")));
    auto dilation = std::vector<int64_t>(node->is(Symbol::attr("dilation")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::thnn_conv_transpose3d_forward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), kernel_size, std::move(peek(stack, 2, 3)), stride, padding, output_padding, dilation);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose3d_forward", 3, 3);
  }},
  {"thnn_conv_transpose3d_forward-8", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("thnn_conv_transpose3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 8));
      auto kernel_size_tensor = peek(stack, 2, 8);
      if (kernel_size_tensor.dim() == 0)
          kernel_size_tensor = kernel_size_tensor.expand(3);
      auto kernel_size = tensor_as<at::IntList>(std::move(kernel_size_tensor));
      auto stride_tensor = peek(stack, 4, 8);
      if (stride_tensor.dim() == 0)
          stride_tensor = stride_tensor.expand(3);
      auto stride = tensor_as<at::IntList>(std::move(stride_tensor));
      auto padding_tensor = peek(stack, 5, 8);
      if (padding_tensor.dim() == 0)
          padding_tensor = padding_tensor.expand(3);
      auto padding = tensor_as<at::IntList>(std::move(padding_tensor));
      auto output_padding_tensor = peek(stack, 6, 8);
      if (output_padding_tensor.dim() == 0)
          output_padding_tensor = output_padding_tensor.expand(3);
      auto output_padding = tensor_as<at::IntList>(std::move(output_padding_tensor));
      auto dilation_tensor = peek(stack, 7, 8);
      if (dilation_tensor.dim() == 0)
          dilation_tensor = dilation_tensor.expand(3);
      auto dilation = tensor_as<at::IntList>(std::move(dilation_tensor));
      auto result = at::thnn_conv_transpose3d_forward(std::move(peek(stack, 0, 8)), std::move(peek(stack, 1, 8)), kernel_size, std::move(peek(stack, 3, 8)), stride, padding, output_padding, dilation);
      drop(stack, 8);
      pack(stack, std::move(result));
      return 0;
    }, "thnn_conv_transpose3d_forward", 8, 3);
  }},
  {"threshold-1-threshold_t-value_t", [](Node *node) {
    auto threshold = Scalar(node->t(Symbol::attr("threshold")));
    auto value = Scalar(node->t(Symbol::attr("value")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("threshold");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::threshold(std::move(peek(stack, 0, 1)), threshold, value);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "threshold", 1, 1);
  }},
  {"threshold-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("threshold");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto threshold = tensor_as<Scalar>(std::move(peek(stack, 1, 3)));
      auto value = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::threshold(std::move(peek(stack, 0, 3)), threshold, value);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "threshold", 3, 1);
  }},
  {"threshold_backward-2-threshold_t-value_t", [](Node *node) {
    auto threshold = Scalar(node->t(Symbol::attr("threshold")));
    auto value = Scalar(node->t(Symbol::attr("value")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("threshold_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::threshold_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), threshold, value);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "threshold_backward", 2, 1);
  }},
  {"threshold_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("threshold_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto threshold = tensor_as<Scalar>(std::move(peek(stack, 2, 4)));
      auto value = tensor_as<Scalar>(std::move(peek(stack, 3, 4)));
      auto result = at::threshold_backward(std::move(peek(stack, 0, 4)), std::move(peek(stack, 1, 4)), threshold, value);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "threshold_backward", 4, 1);
  }},
  {"threshold_forward-1-threshold_t-value_t", [](Node *node) {
    auto threshold = Scalar(node->t(Symbol::attr("threshold")));
    auto value = Scalar(node->t(Symbol::attr("value")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("threshold_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::threshold_forward(std::move(peek(stack, 0, 1)), threshold, value);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "threshold_forward", 1, 1);
  }},
  {"threshold_forward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("threshold_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto threshold = tensor_as<Scalar>(std::move(peek(stack, 1, 3)));
      auto value = tensor_as<Scalar>(std::move(peek(stack, 2, 3)));
      auto result = at::threshold_forward(std::move(peek(stack, 0, 3)), threshold, value);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "threshold_forward", 3, 1);
  }},
  {"to_dense-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("to_dense");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).to_dense();
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "to_dense", 1, 1);
  }},
  {"topk-1-dim_i-k_i-largest_i-sorted_i", [](Node *node) {
    auto k = int64_t(node->i(Symbol::attr("k")));
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto largest = bool(node->i(Symbol::attr("largest")));
    auto sorted = bool(node->i(Symbol::attr("sorted")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("topk");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::topk(std::move(peek(stack, 0, 1)), k, dim, largest, sorted);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "topk", 1, 2);
  }},
  {"topk-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("topk");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto k = tensor_as<int64_t>(std::move(peek(stack, 1, 5)));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 2, 5)));
      auto largest = tensor_as<bool>(std::move(peek(stack, 3, 5)));
      auto sorted = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::topk(std::move(peek(stack, 0, 5)), k, dim, largest, sorted);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "topk", 5, 2);
  }},
  {"trace-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("trace");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::trace(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "trace", 1, 1);
  }},
  {"transpose-1-dim0_i-dim1_i", [](Node *node) {
    auto dim0 = int64_t(node->i(Symbol::attr("dim0")));
    auto dim1 = int64_t(node->i(Symbol::attr("dim1")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("transpose");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::transpose(std::move(peek(stack, 0, 1)), dim0, dim1);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "transpose", 1, 1);
  }},
  {"transpose-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("transpose");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto dim0 = tensor_as<int64_t>(std::move(peek(stack, 1, 3)));
      auto dim1 = tensor_as<int64_t>(std::move(peek(stack, 2, 3)));
      auto result = at::transpose(std::move(peek(stack, 0, 3)), dim0, dim1);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "transpose", 3, 1);
  }},
  {"tril-1-diagonal_i", [](Node *node) {
    auto diagonal = int64_t(node->i(Symbol::attr("diagonal")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("tril");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::tril(std::move(peek(stack, 0, 1)), diagonal);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "tril", 1, 1);
  }},
  {"tril-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("tril");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto diagonal = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::tril(std::move(peek(stack, 0, 2)), diagonal);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "tril", 2, 1);
  }},
  {"triplet_margin_loss-3-eps_f-margin_f-p_f-reduce_i-size_average_i-swap_i", [](Node *node) {
    auto margin = double(node->f(Symbol::attr("margin")));
    auto p = double(node->f(Symbol::attr("p")));
    auto eps = double(node->f(Symbol::attr("eps")));
    auto swap = bool(node->i(Symbol::attr("swap")));
    auto size_average = bool(node->i(Symbol::attr("size_average")));
    auto reduce = bool(node->i(Symbol::attr("reduce")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("triplet_margin_loss");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::triplet_margin_loss(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)), margin, p, eps, swap, size_average, reduce);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "triplet_margin_loss", 3, 1);
  }},
  {"triplet_margin_loss-9", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("triplet_margin_loss");
      AutoGPU device_guard(deviceForInputs(stack, 9));
      auto margin = tensor_as<double>(std::move(peek(stack, 3, 9)));
      auto p = tensor_as<double>(std::move(peek(stack, 4, 9)));
      auto eps = tensor_as<double>(std::move(peek(stack, 5, 9)));
      auto swap = tensor_as<bool>(std::move(peek(stack, 6, 9)));
      auto size_average = tensor_as<bool>(std::move(peek(stack, 7, 9)));
      auto reduce = tensor_as<bool>(std::move(peek(stack, 8, 9)));
      auto result = at::triplet_margin_loss(std::move(peek(stack, 0, 9)), std::move(peek(stack, 1, 9)), std::move(peek(stack, 2, 9)), margin, p, eps, swap, size_average, reduce);
      drop(stack, 9);
      pack(stack, std::move(result));
      return 0;
    }, "triplet_margin_loss", 9, 1);
  }},
  {"triu-1-diagonal_i", [](Node *node) {
    auto diagonal = int64_t(node->i(Symbol::attr("diagonal")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("triu");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::triu(std::move(peek(stack, 0, 1)), diagonal);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "triu", 1, 1);
  }},
  {"triu-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("triu");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto diagonal = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::triu(std::move(peek(stack, 0, 2)), diagonal);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "triu", 2, 1);
  }},
  {"trtrs-2-transpose_i-unitriangular_i-upper_i", [](Node *node) {
    auto upper = bool(node->i(Symbol::attr("upper")));
    auto transpose = bool(node->i(Symbol::attr("transpose")));
    auto unitriangular = bool(node->i(Symbol::attr("unitriangular")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("trtrs");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::trtrs(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), upper, transpose, unitriangular);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "trtrs", 2, 2);
  }},
  {"trtrs-5", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("trtrs");
      AutoGPU device_guard(deviceForInputs(stack, 5));
      auto upper = tensor_as<bool>(std::move(peek(stack, 2, 5)));
      auto transpose = tensor_as<bool>(std::move(peek(stack, 3, 5)));
      auto unitriangular = tensor_as<bool>(std::move(peek(stack, 4, 5)));
      auto result = at::trtrs(std::move(peek(stack, 0, 5)), std::move(peek(stack, 1, 5)), upper, transpose, unitriangular);
      drop(stack, 5);
      pack(stack, std::move(result));
      return 0;
    }, "trtrs", 5, 2);
  }},
  {"trunc-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("trunc");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::trunc(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "trunc", 1, 1);
  }},
  {"type_as-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("type_as");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = (std::move(peek(stack, 0, 2))).type_as(std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "type_as", 2, 1);
  }},
  {"unfold-1-dimension_i-size_i-step_i", [](Node *node) {
    auto dimension = int64_t(node->i(Symbol::attr("dimension")));
    auto size = int64_t(node->i(Symbol::attr("size")));
    auto step = int64_t(node->i(Symbol::attr("step")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("unfold");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).unfold(dimension, size, step);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "unfold", 1, 1);
  }},
  {"unfold-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("unfold");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto dimension = tensor_as<int64_t>(std::move(peek(stack, 1, 4)));
      auto size = tensor_as<int64_t>(std::move(peek(stack, 2, 4)));
      auto step = tensor_as<int64_t>(std::move(peek(stack, 3, 4)));
      auto result = (std::move(peek(stack, 0, 4))).unfold(dimension, size, step);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "unfold", 4, 1);
  }},
  {"unsqueeze-1-dim_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("unsqueeze");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::unsqueeze(std::move(peek(stack, 0, 1)), dim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "unsqueeze", 1, 1);
  }},
  {"unsqueeze-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("unsqueeze");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::unsqueeze(std::move(peek(stack, 0, 2)), dim);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "unsqueeze", 2, 1);
  }},
  {"upsample_bilinear2d-1-align_corners_i-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    auto align_corners = bool(node->i(Symbol::attr("align_corners")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_bilinear2d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_bilinear2d(std::move(peek(stack, 0, 1)), output_size, align_corners);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_bilinear2d", 1, 1);
  }},
  {"upsample_bilinear2d-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_bilinear2d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto output_size_tensor = peek(stack, 1, 3);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(2);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto align_corners = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::upsample_bilinear2d(std::move(peek(stack, 0, 3)), output_size, align_corners);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_bilinear2d", 3, 1);
  }},
  {"upsample_bilinear2d_backward-1-align_corners_i-input_size_is-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    auto input_size = std::vector<int64_t>(node->is(Symbol::attr("input_size")));
    auto align_corners = bool(node->i(Symbol::attr("align_corners")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_bilinear2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_bilinear2d_backward(std::move(peek(stack, 0, 1)), output_size, input_size, align_corners);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_bilinear2d_backward", 1, 1);
  }},
  {"upsample_bilinear2d_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_bilinear2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto output_size_tensor = peek(stack, 1, 4);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(2);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto input_size_tensor = peek(stack, 2, 4);
      if (input_size_tensor.dim() == 0)
          input_size_tensor = input_size_tensor.expand(4);
      auto input_size = tensor_as<at::IntList>(std::move(input_size_tensor));
      auto align_corners = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::upsample_bilinear2d_backward(std::move(peek(stack, 0, 4)), output_size, input_size, align_corners);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_bilinear2d_backward", 4, 1);
  }},
  {"upsample_bilinear2d_forward-1-align_corners_i-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    auto align_corners = bool(node->i(Symbol::attr("align_corners")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_bilinear2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_bilinear2d_forward(std::move(peek(stack, 0, 1)), output_size, align_corners);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_bilinear2d_forward", 1, 1);
  }},
  {"upsample_bilinear2d_forward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_bilinear2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto output_size_tensor = peek(stack, 1, 3);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(2);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto align_corners = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::upsample_bilinear2d_forward(std::move(peek(stack, 0, 3)), output_size, align_corners);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_bilinear2d_forward", 3, 1);
  }},
  {"upsample_linear1d-1-align_corners_i-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    auto align_corners = bool(node->i(Symbol::attr("align_corners")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_linear1d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_linear1d(std::move(peek(stack, 0, 1)), output_size, align_corners);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_linear1d", 1, 1);
  }},
  {"upsample_linear1d-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_linear1d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto output_size_tensor = peek(stack, 1, 3);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(1);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto align_corners = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::upsample_linear1d(std::move(peek(stack, 0, 3)), output_size, align_corners);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_linear1d", 3, 1);
  }},
  {"upsample_linear1d_backward-1-align_corners_i-input_size_is-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    auto input_size = std::vector<int64_t>(node->is(Symbol::attr("input_size")));
    auto align_corners = bool(node->i(Symbol::attr("align_corners")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_linear1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_linear1d_backward(std::move(peek(stack, 0, 1)), output_size, input_size, align_corners);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_linear1d_backward", 1, 1);
  }},
  {"upsample_linear1d_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_linear1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto output_size_tensor = peek(stack, 1, 4);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(1);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto input_size_tensor = peek(stack, 2, 4);
      if (input_size_tensor.dim() == 0)
          input_size_tensor = input_size_tensor.expand(3);
      auto input_size = tensor_as<at::IntList>(std::move(input_size_tensor));
      auto align_corners = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::upsample_linear1d_backward(std::move(peek(stack, 0, 4)), output_size, input_size, align_corners);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_linear1d_backward", 4, 1);
  }},
  {"upsample_linear1d_forward-1-align_corners_i-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    auto align_corners = bool(node->i(Symbol::attr("align_corners")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_linear1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_linear1d_forward(std::move(peek(stack, 0, 1)), output_size, align_corners);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_linear1d_forward", 1, 1);
  }},
  {"upsample_linear1d_forward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_linear1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto output_size_tensor = peek(stack, 1, 3);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(1);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto align_corners = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::upsample_linear1d_forward(std::move(peek(stack, 0, 3)), output_size, align_corners);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_linear1d_forward", 3, 1);
  }},
  {"upsample_nearest1d-1-scale_factor_i", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol::attr("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest1d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_nearest1d(std::move(peek(stack, 0, 1)), scale_factor);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest1d", 1, 1);
  }},
  {"upsample_nearest1d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest1d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto scale_factor = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::upsample_nearest1d(std::move(peek(stack, 0, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest1d", 2, 1);
  }},
  {"upsample_nearest1d_backward-2-scale_factor_i", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol::attr("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::upsample_nearest1d_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest1d_backward", 2, 1);
  }},
  {"upsample_nearest1d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest1d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto scale_factor = tensor_as<int64_t>(std::move(peek(stack, 2, 3)));
      auto result = at::upsample_nearest1d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), scale_factor);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest1d_backward", 3, 1);
  }},
  {"upsample_nearest1d_forward-1-scale_factor_i", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol::attr("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_nearest1d_forward(std::move(peek(stack, 0, 1)), scale_factor);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest1d_forward", 1, 1);
  }},
  {"upsample_nearest1d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest1d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto scale_factor = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::upsample_nearest1d_forward(std::move(peek(stack, 0, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest1d_forward", 2, 1);
  }},
  {"upsample_nearest2d-1-scale_factor_i", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol::attr("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest2d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_nearest2d(std::move(peek(stack, 0, 1)), scale_factor);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest2d", 1, 1);
  }},
  {"upsample_nearest2d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest2d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto scale_factor = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::upsample_nearest2d(std::move(peek(stack, 0, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest2d", 2, 1);
  }},
  {"upsample_nearest2d_backward-2-scale_factor_i", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol::attr("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::upsample_nearest2d_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest2d_backward", 2, 1);
  }},
  {"upsample_nearest2d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest2d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto scale_factor = tensor_as<int64_t>(std::move(peek(stack, 2, 3)));
      auto result = at::upsample_nearest2d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), scale_factor);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest2d_backward", 3, 1);
  }},
  {"upsample_nearest2d_forward-1-scale_factor_i", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol::attr("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_nearest2d_forward(std::move(peek(stack, 0, 1)), scale_factor);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest2d_forward", 1, 1);
  }},
  {"upsample_nearest2d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest2d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto scale_factor = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::upsample_nearest2d_forward(std::move(peek(stack, 0, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest2d_forward", 2, 1);
  }},
  {"upsample_nearest3d-1-scale_factor_i", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol::attr("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest3d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_nearest3d(std::move(peek(stack, 0, 1)), scale_factor);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest3d", 1, 1);
  }},
  {"upsample_nearest3d-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest3d");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto scale_factor = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::upsample_nearest3d(std::move(peek(stack, 0, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest3d", 2, 1);
  }},
  {"upsample_nearest3d_backward-2-scale_factor_i", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol::attr("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = at::upsample_nearest3d_backward(std::move(peek(stack, 0, 2)), std::move(peek(stack, 1, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest3d_backward", 2, 1);
  }},
  {"upsample_nearest3d_backward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto scale_factor = tensor_as<int64_t>(std::move(peek(stack, 2, 3)));
      auto result = at::upsample_nearest3d_backward(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), scale_factor);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest3d_backward", 3, 1);
  }},
  {"upsample_nearest3d_forward-1-scale_factor_i", [](Node *node) {
    auto scale_factor = int64_t(node->i(Symbol::attr("scale_factor")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_nearest3d_forward(std::move(peek(stack, 0, 1)), scale_factor);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest3d_forward", 1, 1);
  }},
  {"upsample_nearest3d_forward-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_nearest3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto scale_factor = tensor_as<int64_t>(std::move(peek(stack, 1, 2)));
      auto result = at::upsample_nearest3d_forward(std::move(peek(stack, 0, 2)), scale_factor);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_nearest3d_forward", 2, 1);
  }},
  {"upsample_trilinear3d-1-align_corners_i-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    auto align_corners = bool(node->i(Symbol::attr("align_corners")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_trilinear3d");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_trilinear3d(std::move(peek(stack, 0, 1)), output_size, align_corners);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_trilinear3d", 1, 1);
  }},
  {"upsample_trilinear3d-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_trilinear3d");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto output_size_tensor = peek(stack, 1, 3);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(3);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto align_corners = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::upsample_trilinear3d(std::move(peek(stack, 0, 3)), output_size, align_corners);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_trilinear3d", 3, 1);
  }},
  {"upsample_trilinear3d_backward-1-align_corners_i-input_size_is-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    auto input_size = std::vector<int64_t>(node->is(Symbol::attr("input_size")));
    auto align_corners = bool(node->i(Symbol::attr("align_corners")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_trilinear3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_trilinear3d_backward(std::move(peek(stack, 0, 1)), output_size, input_size, align_corners);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_trilinear3d_backward", 1, 1);
  }},
  {"upsample_trilinear3d_backward-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_trilinear3d_backward");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto output_size_tensor = peek(stack, 1, 4);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(3);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto input_size_tensor = peek(stack, 2, 4);
      if (input_size_tensor.dim() == 0)
          input_size_tensor = input_size_tensor.expand(5);
      auto input_size = tensor_as<at::IntList>(std::move(input_size_tensor));
      auto align_corners = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::upsample_trilinear3d_backward(std::move(peek(stack, 0, 4)), output_size, input_size, align_corners);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_trilinear3d_backward", 4, 1);
  }},
  {"upsample_trilinear3d_forward-1-align_corners_i-output_size_is", [](Node *node) {
    auto output_size = std::vector<int64_t>(node->is(Symbol::attr("output_size")));
    auto align_corners = bool(node->i(Symbol::attr("align_corners")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_trilinear3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::upsample_trilinear3d_forward(std::move(peek(stack, 0, 1)), output_size, align_corners);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_trilinear3d_forward", 1, 1);
  }},
  {"upsample_trilinear3d_forward-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("upsample_trilinear3d_forward");
      AutoGPU device_guard(deviceForInputs(stack, 3));
      auto output_size_tensor = peek(stack, 1, 3);
      if (output_size_tensor.dim() == 0)
          output_size_tensor = output_size_tensor.expand(3);
      auto output_size = tensor_as<at::IntList>(std::move(output_size_tensor));
      auto align_corners = tensor_as<bool>(std::move(peek(stack, 2, 3)));
      auto result = at::upsample_trilinear3d_forward(std::move(peek(stack, 0, 3)), output_size, align_corners);
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "upsample_trilinear3d_forward", 3, 1);
  }},
  {"var-1-dim_i-keepdim_i-unbiased_i", [](Node *node) {
    auto dim = int64_t(node->i(Symbol::attr("dim")));
    auto unbiased = bool(node->i(Symbol::attr("unbiased")));
    auto keepdim = bool(node->i(Symbol::attr("keepdim")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("var");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::var(std::move(peek(stack, 0, 1)), dim, unbiased, keepdim);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "var", 1, 1);
  }},
  {"var-1-unbiased_i", [](Node *node) {
    auto unbiased = bool(node->i(Symbol::attr("unbiased")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("var");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::var(std::move(peek(stack, 0, 1)), unbiased);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "var", 1, 1);
  }},
  {"var-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("var");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto unbiased = tensor_as<bool>(std::move(peek(stack, 1, 2)));
      auto result = at::var(std::move(peek(stack, 0, 2)), unbiased);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "var", 2, 1);
  }},
  {"var-4", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("var");
      AutoGPU device_guard(deviceForInputs(stack, 4));
      auto dim = tensor_as<int64_t>(std::move(peek(stack, 1, 4)));
      auto unbiased = tensor_as<bool>(std::move(peek(stack, 2, 4)));
      auto keepdim = tensor_as<bool>(std::move(peek(stack, 3, 4)));
      auto result = at::var(std::move(peek(stack, 0, 4)), dim, unbiased, keepdim);
      drop(stack, 4);
      pack(stack, std::move(result));
      return 0;
    }, "var", 4, 1);
  }},
  {"view-1-size_is", [](Node *node) {
    auto size = std::vector<int64_t>(node->is(Symbol::attr("size")));
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("view");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = (std::move(peek(stack, 0, 1))).view(size);
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "view", 1, 1);
  }},
  {"view-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("view");
      AutoGPU device_guard(deviceForInputs(stack, 2));
      auto size = tensor_as<IntList>(std::move(peek(stack, 1, 2)));
      auto result = (std::move(peek(stack, 0, 2))).view(size);
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "view", 2, 1);
  }},
  {"view_as-2", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("view_as");
      AutoGPU device_guard(deviceForInputs(stack, 2));
  
      auto result = (std::move(peek(stack, 0, 2))).view_as(std::move(peek(stack, 1, 2)));
      drop(stack, 2);
      pack(stack, std::move(result));
      return 0;
    }, "view_as", 2, 1);
  }},
  {"where-3", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("where");
      AutoGPU device_guard(deviceForInputs(stack, 3));
  
      auto result = at::where(std::move(peek(stack, 0, 3)), std::move(peek(stack, 1, 3)), std::move(peek(stack, 2, 3)));
      drop(stack, 3);
      pack(stack, std::move(result));
      return 0;
    }, "where", 3, 1);
  }},
  {"zeros_like-1", [](Node *node) {
  
    return TensorOp([=](Stack & stack) {
      autograd::profiler::RecordFunction record("zeros_like");
      AutoGPU device_guard(deviceForInputs(stack, 1));
  
      auto result = at::zeros_like(std::move(peek(stack, 0, 1)));
      drop(stack, 1);
      pack(stack, std::move(result));
      return 0;
    }, "zeros_like", 1, 1);
  }},
};

std::string getDescriptor(jit::Node* n) {
  std::stringstream s;
  JIT_ASSERTM(n->kind().is_aten(), "%s is not an ATen op", n->kind().toDisplayString());
  s << n->kind().toUnqualString();
  if (tensor_vararg_fns.count(n->kind()) == 0)
    s << "-" << n->inputs().size();
  else
    s << "-*";
  std::vector<std::string> attr_names = fmap(n->attributeNames(), [&](Symbol x) {
    std::stringstream ss;
    ss << x.toUnqualString() << "_" << toString(n->kindOf(x));
    return ss.str();
  });
  std::sort(attr_names.begin(), attr_names.end());

  for (const auto & name : attr_names)
    s << "-" << name;
  return s.str();
}

} // anonymous namespace

at::optional<TensorOp> findTensorOp(jit::Node* n) {
  auto signature = getDescriptor(n);
  auto it = constructors.find(signature);
  if(it == constructors.end()) {
    return at::nullopt;
  }
  return it->second(n);
}
TensorOp getTensorOp(jit::Node* n) {
  auto op = findTensorOp(n);
  if (!op) {
    throw std::runtime_error(
        "Unsupported op descriptor: " + getDescriptor(n) +
        ". "
        "File a bug report.");
  }
  return op.value();
}

}} // namespace torch::jit

// @generated
#pragma once

#include <THC/THC.h>
#include <THC/THCTensor.hpp>
#include <THCUNN/THCUNN.h>
#undef THNN_
#undef THCIndexTensor_
#include <THCS/THCS.h>
#include <THCS/THCSTensor.hpp>
#undef THCIndexTensor_

#include "ATen/Tensor.h"
#include "ATen/TensorImpl.h"
#include "ATen/Context.h"
#include "ATen/TensorMethods.h"

namespace at {

struct CUDAByteTensor final : public TensorImpl {
public:
  explicit CUDAByteTensor(Context* context);
  CUDAByteTensor(Context* context, THCudaByteTensor * tensor);
  virtual ~CUDAByteTensor();
  virtual const char * toString() const override;
  virtual IntList sizes() const override;
  virtual IntList strides() const override;
  virtual int64_t dim() const override;
  virtual Scalar localScalar() override;
  virtual void * unsafeGetTH(bool retain) override;
  virtual std::unique_ptr<Storage> storage() override;
  static const char * typeString();

//TODO(zach): sort of friend permissions later so this
// can be protected
public:
  THCudaByteTensor * tensor;
  Context* context;
  friend struct CUDAByteType;
};

} // namespace at

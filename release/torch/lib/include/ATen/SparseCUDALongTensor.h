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

struct SparseCUDALongTensor final : public TensorImpl {
public:
  explicit SparseCUDALongTensor(Context* context);
  SparseCUDALongTensor(Context* context, THCSLongTensor * tensor);
  virtual ~SparseCUDALongTensor();
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
  THCSLongTensor * tensor;
  Context* context;
  friend struct SparseCUDALongType;
};

} // namespace at

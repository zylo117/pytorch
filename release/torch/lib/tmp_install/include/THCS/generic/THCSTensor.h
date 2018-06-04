#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensor.h"
#else

typedef struct THCSTensor THCSTensor;

/**** access methods ****/
THC_API int THCSTensor_(nDimension)(THCState *state, const THCSTensor *self);
THC_API int THCSTensor_(nDimensionI)(THCState *state, const THCSTensor *self);
THC_API int THCSTensor_(nDimensionV)(THCState *state, const THCSTensor *self);
THC_API int64_t THCSTensor_(size)(THCState *state, const THCSTensor *self, int dim);
THC_API ptrdiff_t THCSTensor_(nnz)(THCState *state, const THCSTensor *self);
THC_API THLongStorage *THCSTensor_(newSizeOf)(THCState *state, THCSTensor *self);
THC_API THCIndexTensor *THCSTensor_(newIndices)(THCState *state, const THCSTensor *self);
THC_API THCTensor *THCSTensor_(newValues)(THCState *state, const THCSTensor *self);

/**** creation methods ****/
THC_API THCSTensor *THCSTensor_(new)(THCState *state);
THC_API THCSTensor *THCSTensor_(newWithTensor)(THCState *state, THCIndexTensor *indices, THCTensor *values);
THC_API THCSTensor *THCSTensor_(newWithTensorAndSizeUnsafe)(THCState *state, THCIndexTensor *indices, THCTensor *values, THLongStorage *sizes);
THC_API THCSTensor *THCSTensor_(newWithTensorAndSize)(THCState *state, THCIndexTensor *indices, THCTensor *values, THLongStorage *sizes);

THC_API THCSTensor *THCSTensor_(newWithSize)(THCState *state, THLongStorage *size_, THLongStorage *_ignored);
THC_API THCSTensor *THCSTensor_(newWithSize1d)(THCState *state, int64_t size0_);
THC_API THCSTensor *THCSTensor_(newWithSize2d)(THCState *state, int64_t size0_, int64_t size1_);
THC_API THCSTensor *THCSTensor_(newWithSize3d)(THCState *state, int64_t size0_, int64_t size1_, int64_t size2_);
THC_API THCSTensor *THCSTensor_(newWithSize4d)(THCState *state, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_);

THC_API THCSTensor *THCSTensor_(newClone)(THCState *state, THCSTensor *self);
THC_API THCSTensor *THCSTensor_(newTranspose)(THCState *state, THCSTensor *self, int dimension1_, int dimension2_);

/**** reshaping methods ***/
THC_API int THCSTensor_(isSameSizeAs)(THCState *state, const THCSTensor *self, const THCSTensor* src);
THC_API int THCSTensor_(isSameSizeAsDense)(THCState *state, const THCSTensor *self, const THCTensor* src);
THC_API THCSTensor *THCSTensor_(resize)(THCState *state, THCSTensor *self, THLongStorage *size);
THC_API THCSTensor *THCSTensor_(resizeAs)(THCState *state, THCSTensor *self, THCSTensor *src);
THC_API THCSTensor *THCSTensor_(resize1d)(THCState *state, THCSTensor *self, int64_t size0);
THC_API THCSTensor *THCSTensor_(resize2d)(THCState *state, THCSTensor *self, int64_t size0, int64_t size1);
THC_API THCSTensor *THCSTensor_(resize3d)(THCState *state, THCSTensor *self, int64_t size0, int64_t size1, int64_t size2);
THC_API THCSTensor *THCSTensor_(resize4d)(THCState *state, THCSTensor *self, int64_t size0, int64_t size1, int64_t size2, int64_t size3);

THC_API THCTensor *THCSTensor_(toDense)(THCState *state, THCSTensor *self);
THC_API void THCSTensor_(copy)(THCState *state, THCSTensor *self, THCSTensor *src);

THC_API void THCSTensor_(transpose)(THCState *state, THCSTensor *self, int dimension1_, int dimension2_);
THC_API int THCSTensor_(isCoalesced)(THCState *state, const THCSTensor *self);
THC_API THCSTensor *THCSTensor_(newCoalesce)(THCState *state, THCSTensor *self);

THC_API void THCTensor_(sparseMask)(THCState *state, THCSTensor *r_, THCTensor *t, THCSTensor *mask);

THC_API void THCSTensor_(free)(THCState *state, THCSTensor *self);
THC_API void THCSTensor_(retain)(THCState *state, THCSTensor *self);

/* CUDA-specific functions */
THC_API int THCSTensor_(getDevice)(THCState *state, const THCSTensor *self);
// NB: nTensors is the number of TOTAL tensors, not the number of dense tensors.
// That is to say, nSparseTensors + nDenseTensors == nTensors
THC_API int THCSTensor_(checkGPU)(THCState *state, unsigned int nSparseTensors, unsigned int nTensors, ...);

/* internal methods */
THC_API THCSTensor* THCSTensor_(rawResize)(THCState *state, THCSTensor *self, int nDimI, int nDimV, int64_t *size);
THC_API THCTensor *THCSTensor_(newValuesWithSizeOf)(THCState *state, THCTensor *values, int64_t nnz);
THC_API THCSTensor* THCSTensor_(_move)(THCState *state, THCSTensor *self, THCIndexTensor *indices, THCTensor *values);
THC_API THCSTensor* THCSTensor_(_set)(THCState *state, THCSTensor *self, THCIndexTensor *indices, THCTensor *values);
// forceClone is intended to use as a boolean
THC_API THCIndexTensor* THCSTensor_(newFlattenedIndices)(THCState *state, THCSTensor *self, int forceClone);

#endif

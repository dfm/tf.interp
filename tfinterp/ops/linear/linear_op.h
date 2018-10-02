#ifndef _TFINTERP_LINEAR_OP_H_
#define _TFINTERP_LINEAR_OP_H_

#include <Eigen/Core>
#include "interp.h"

#if GOOGLE_CUDA
template <typename T>
struct LinearInterpCUDAFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, int M, const T* const x, const T* const y, int N, const T* const t, T* v, int* inds);
};
#endif

#endif

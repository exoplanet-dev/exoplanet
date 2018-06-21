#ifndef _TRANSIT_TRANSIT_OP_H_
#define _TRANSIT_TRANSIT_OP_H_

#include <Eigen/Core>
#include "transit.h"

template <typename Device, typename T>
struct TransitFunctor {
  void operator()(const Device& d, int grid_size, const T* const grid,
                  int size, const T* const z, const T* const r, T* delta);
};

#if GOOGLE_CUDA
template <typename T>
struct TransitFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int grid_size, const T* const grid,
                  int size, const T* const z, const T* const r, T* delta);
};
#endif

#endif

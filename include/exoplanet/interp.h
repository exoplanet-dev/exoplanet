#ifndef _EXOPLANET_INTERP_H_
#define _EXOPLANET_INTERP_H_

#include <cmath>
#include "exoplanet/cuda_utils.h"

namespace exoplanet {
  namespace interp {

    // Adapted from https://academy.realm.io/posts/how-we-beat-cpp-stl-binary-search/
    template <typename T>
    EXOPLANET_CUDA_CALLABLE
    inline int search_sorted (int N, const T* const x, T value) {
      int low = -1;
      int high = N;
      while (high - low > 1) {
        int probe = (low + high) / 2;
        T v = x[probe];
        if (v > value)
          high = probe;
        else
          low = probe;
      }
      return high;
    }

    template <typename T>
    EXOPLANET_CUDA_CALLABLE
    inline int interp1d (int M, const T* const x, const T* const y, T value, T* v) {
      bool low = value <= x[0];
      bool high = value >= x[M-1];
      if (!low && !high) {
        int ind = search_sorted(M, x, value);
        T a = (value - x[ind-1]) / (x[ind] - x[ind-1]);
        *v = a * y[ind] + (1.0 - a) * y[ind-1];
        return ind;
      } else if (low) {
        *v = y[0];
        return 0;
      }
      *v = y[M-1];
      return M;
    }

  };
};

#endif

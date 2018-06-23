#ifndef _EXOPLANET_CUDA_UTILS_H_
#define _EXOPLANET_CUDA_UTILS_H_

#ifdef __CUDACC__
#define EXOPLANET_CUDA_CALLABLE __host__ __device__
#else
#define EXOPLANET_CUDA_CALLABLE
#endif

#endif

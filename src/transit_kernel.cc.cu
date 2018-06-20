#include <cmath>

template <typename T>
__device__ inline void step_advance (T* __restrict__ const step_scale,
                                     T* __restrict__ const x1,
                                     T* __restrict__       x2)
{
  *x2 = *x1 + *step_scale * acos(*x1);
}

template <typename T>
__device__ void compute_area (T* __restrict__ const x,
                              T* __restrict__ const z,
                              T* __restrict__ const r,
                              T* __restrict__       area)
{
    T z2 = (*z)*(*z), x2 = (*x)*(*x), r2 = (*r)*(*r);

    if (*x <= *r - *z) {
      *area = M_PI * x2;
      return;
    } else if (*x >= *z + *r) {
      *area = M_PI * r2;
      return;
    }

    T u = 0.5 * (z2 + x2 - r2) / ((*z) * (*x));
    T v = 0.5 * (z2 + r2 - x2) / ((*z) * (*r));
    T w = (-*z + *x + *r) * (*z + *x - *r) * (*z - *x + *r) * (*z + *x + *r);
    if (w < 0.0) w = 0.0;
    *area = x2 * acos(u) + r2 * acos(v) - 0.5 * sqrt(w);
}

template <typename T>
__global__ void transit_kernel(int size,
                               T* __restrict__ const z,
                               T* __restrict__ const r,
                               T* __restrict__       delta)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  T x = 1.0;
  for (int i = index; i < size; i += stride) {
    compute_area<T> (&x, z[i], r[i], delta[i]);
  }
}

int main(void)
{
  int N = 70000;
  float *z, *r, *delta;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&z,     N*sizeof(float));
  cudaMallocManaged(&r,     N*sizeof(float));
  cudaMallocManaged(&delta, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    z[i] = 1.0 - 2*float(i) / N;
    r[i] = 0.01;
  }

  transit_kernel<<<1, 1>>>(N, z, r, delta);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  /*float maxError = 0.0f;*/
  /*for (int i = 0; i < N; i++)*/
  /*  maxError = fmax(maxError, fabs(y[i]-3.0f));*/
  /*std::cout << "Max error: " << maxError << std::endl;*/

  // Free memory
  cudaFree(z);
  cudaFree(r);
  cudaFree(delta);

  return 0;
}

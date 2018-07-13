#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

#include "transit_op.h"

using namespace tensorflow;
using namespace exoplanet;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("TransitDepth")
  .Attr("T: {float, double}")
  .Input("radius: T")
  .Input("intensity: T")
  .Input("n_min: int32")
  .Input("n_max: int32")
  .Input("z: T")
  .Input("r: T")
  .Input("direction: T")
  .Output("delta: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle shape;

    TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &shape));
    TF_RETURN_IF_ERROR(c->Merge(c->input(2), c->input(3), &shape));
    TF_RETURN_IF_ERROR(c->Merge(shape, c->input(4), &shape));
    TF_RETURN_IF_ERROR(c->Merge(shape, c->input(5), &shape));
    TF_RETURN_IF_ERROR(c->Merge(shape, c->input(6), &shape));

    c->set_output(0, c->input(4));
    return Status::OK();
  });

///
/// A custom TensorFlow op to compute a transit light curve
///
/// The key assumption is that the limb darkening profile defined by the
/// ``intensity`` parameter is a function ``I(x)`` that integrates to one over
/// a disk of radius one
///
template <typename T>
class TransitDepthOpBase : public OpKernel {
  public:
    explicit TransitDepthOpBase (OpKernelConstruction* context) : OpKernel(context) {}

    virtual void DoCompute (OpKernelContext* ctx,
      int64 N,
      const T* const radius,
      const T* const intensity,
      int64 size,
      const int* const n_min,
      const int* const n_max,
      const T* const z,
      const T* const r,
      const T* const direction,
      T*             delta
    ) = 0;

    void Compute(OpKernelContext* context) override {
      // Inputs
      const Tensor& radius_tensor    = context->input(0);
      const Tensor& intensity_tensor = context->input(1);
      const Tensor& n_min_tensor     = context->input(2);
      const Tensor& n_max_tensor     = context->input(3);
      const Tensor& z_tensor         = context->input(4);
      const Tensor& r_tensor         = context->input(5);
      const Tensor& direction_tensor = context->input(6);

      // Dimensions
      const int64 N = radius_tensor.NumElements();
      OP_REQUIRES(context, N <= tensorflow::kint32max,
          errors::InvalidArgument("too many elements in tensor"));
      OP_REQUIRES(context, intensity_tensor.NumElements() == N,
          errors::InvalidArgument("radius and intensity must have the same number of elements"));

      const int64 size = z_tensor.NumElements();
      OP_REQUIRES(context, size <= tensorflow::kint32max,
          errors::InvalidArgument("too many elements in tensor"));
      OP_REQUIRES(context, n_min_tensor.NumElements() == size,
          errors::InvalidArgument("z and n_min must have the same number of elements"));
      OP_REQUIRES(context, n_max_tensor.NumElements() == size,
          errors::InvalidArgument("z and n_max must have the same number of elements"));
      OP_REQUIRES(context, r_tensor.NumElements() == size,
          errors::InvalidArgument("z and r must have the same number of elements"));
      OP_REQUIRES(context, direction_tensor.NumElements() == size,
          errors::InvalidArgument("z and direction must have the same number of elements"));

      // Output
      Tensor* delta_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, z_tensor.shape(), &delta_tensor));

      // Access the data
      const auto radius    = radius_tensor.template flat<T>();
      const auto intensity = intensity_tensor.template flat<T>();
      const auto n_min     = n_min_tensor.flat<int>();
      const auto n_max     = n_max_tensor.flat<int>();
      const auto z         = z_tensor.template flat<T>();
      const auto r         = r_tensor.template flat<T>();
      const auto direction = direction_tensor.template flat<T>();
      auto delta           = delta_tensor->template flat<T>();

      // Perform the calculation on the selected device
      DoCompute(context,
        N,
        radius.data(),
        intensity.data(),
        size,
        n_min.data(),
        n_max.data(),
        z.data(),
        r.data(),
        direction.data(),
        delta.data()
      );
    }

};

template <class Device, typename T>
class TransitDepthOp;

template <typename T>
class TransitDepthOp<CPUDevice, T> : public TransitDepthOpBase<T> {

  public:
    explicit TransitDepthOp (OpKernelConstruction* context) : TransitDepthOpBase<T>(context) {}

    void DoCompute (OpKernelContext* ctx,
      int64 N,
      const T* const radius,
      const T* const intensity,
      int64 size,
      const int* const n_min,
      const int* const n_max,
      const T* const z,
      const T* const r,
      const T* const direction,
      T*             delta
    ) override {
      auto work = [&](int64 begin, int64 end) {
        for (int i = begin; i < end; ++i) {
          if (direction[i] > T(0))
            delta[i] = transit::compute_transit_depth<T>(
                N, radius, intensity, n_min[i], n_max[i], z[i], r[i]);
          else delta[i] = T(0);
        }
      };
      auto worker_threads = *ctx->device()->tensorflow_cpu_worker_threads();
      int64 cost = 5 * N;
      Shard(worker_threads.num_threads, worker_threads.workers, size, cost, work);
    }

};


#define REGISTER_CPU(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("TransitDepth").Device(DEVICE_CPU).TypeConstraint<type>("T"),        \
      TransitDepthOp<CPUDevice, type>)

REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU

#ifdef GOOGLE_CUDA
template <typename T>
class TransitDepthOp<GPUDevice, T> : public TransitDepthOpBase<T> {

  public:
    explicit TransitDepthOp (OpKernelConstruction* context) : TransitDepthOpBase<T>(context) {}

    void DoCompute (OpKernelContext* ctx,
      int64 N,
      const T* const radius,
      const T* const intensity,
      int64 size,
      const int* const n_min,
      const int* const n_max,
      const T* const z,
      const T* const r,
      const T* const direction,
      T*             delta
    ) override {
      TransitDepthCUDAFunctor<T>()(
        ctx->eigen_device<GPUDevice>(),
        static_cast<int>(N), radius, intensity,
        static_cast<int>(size), n_min, n_max, z, r,
        direction, delta);
    }

};

#define REGISTER_GPU(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("TransitDepth").Device(DEVICE_GPU).TypeConstraint<type>("T"),         \
      TransitDepthOp<GPUDevice, type>)

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

#endif

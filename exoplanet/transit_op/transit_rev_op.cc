#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "transit_op.h"

using namespace tensorflow;
using namespace exoplanet;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
struct TransitDepthRevFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int N, const T* const radius, const T* const intensity,
                  int size, const int* const n_min, const int* const n_max, const T* const z, const T* const r,
                  const T* const direction, const T* const b_delta, T* b_intensity, T* b_z, T* b_r) {
    for (int i = 0; i < size; ++i) {
      if (direction[i] > 0.0)
        transit::compute_transit_depth_rev<T>(N, radius, intensity, n_min[i], n_max[i], z[i], r[i],
                                              b_delta[i], b_intensity, &(b_z[i]), &(b_r[i]));
    }
  }
};

REGISTER_OP("TransitDepthRev")
  .Attr("T: {float, double}")
  .Input("radius: T")
  .Input("intensity: T")
  .Input("n_min: int32")
  .Input("n_max: int32")
  .Input("z: T")
  .Input("r: T")
  .Input("direction: T")
  .Input("b_delta: T")
  .Output("b_grid: T")
  .Output("b_z: T")
  .Output("b_r: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle shape;
    TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &shape));
    TF_RETURN_IF_ERROR(c->Merge(c->input(2), c->input(3), &shape));
    TF_RETURN_IF_ERROR(c->Merge(shape, c->input(4), &shape));
    TF_RETURN_IF_ERROR(c->Merge(shape, c->input(5), &shape));
    TF_RETURN_IF_ERROR(c->Merge(shape, c->input(6), &shape));
    TF_RETURN_IF_ERROR(c->Merge(shape, c->input(7), &shape));

    c->set_output(0, c->input(0));
    c->set_output(1, c->input(4));
    c->set_output(2, c->input(5));
    return Status::OK();
  });

template <typename Device, typename T>
class TransitDepthRevOp : public OpKernel {
 public:
  explicit TransitDepthRevOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& radius_tensor    = context->input(0);
    const Tensor& intensity_tensor = context->input(1);
    const Tensor& n_min_tensor     = context->input(2);
    const Tensor& n_max_tensor     = context->input(3);
    const Tensor& z_tensor         = context->input(4);
    const Tensor& r_tensor         = context->input(5);
    const Tensor& direction_tensor = context->input(6);
    const Tensor& b_delta_tensor   = context->input(7);

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
    OP_REQUIRES(context, b_delta_tensor.NumElements() == size,
        errors::InvalidArgument("z and b_delta must have the same number of elements"));

    // Output
    Tensor* b_intensity_tensor = NULL,
          * b_z_tensor = NULL,
          * b_r_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, intensity_tensor.shape(), &b_intensity_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, z_tensor.shape(), &b_z_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, r_tensor.shape(), &b_r_tensor));

    // Access the data
    const auto radius    = radius_tensor.template flat<T>();
    const auto intensity = intensity_tensor.template flat<T>();
    const auto n_min     = n_min_tensor.flat<int>();
    const auto n_max     = n_max_tensor.flat<int>();
    const auto z         = z_tensor.template flat<T>();
    const auto r         = r_tensor.template flat<T>();
    const auto direction = direction_tensor.template flat<T>();
    const auto b_delta   = b_delta_tensor.template flat<T>();
    auto b_intensity     = b_intensity_tensor->template flat<T>();
    auto b_z             = b_z_tensor->template flat<T>();
    auto b_r             = b_r_tensor->template flat<T>();

    b_intensity.setZero();
    b_z.setZero();
    b_r.setZero();

    TransitDepthRevFunctor<Device, T>()(context->eigen_device<Device>(),
        static_cast<int>(N), radius.data(), intensity.data(),
        static_cast<int>(size), n_min.data(), n_max.data(), z.data(), r.data(),
        direction.data(),
        b_delta.data(), b_intensity.data(), b_z.data(), b_r.data());
  }
};

#define REGISTER_CPU(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("TransitDepthRev").Device(DEVICE_CPU).TypeConstraint<type>("T"),        \
      TransitDepthRevOp<CPUDevice, type>)

REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("TransitDepthRev").Device(DEVICE_GPU).TypeConstraint<type>("T"),         \
      TransitDepthRevOp<GPUDevice, type>)

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

#endif

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <limits>

#include "transit_op.h"

using namespace tensorflow;
using namespace exoplanet;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
struct TransitDepthFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int N, const T* const radius, const T* const intensity,
                  int size, const int* const n_min, const int* const n_max, const T* const z, T r, T* delta) {
    for (int i = 0; i < size; ++i) {
      delta[i] = transit::compute_transit_depth<T>(N, radius, intensity, n_min[i], n_max[i], z[i], r);
    }
  }
};

REGISTER_OP("TransitDepth")
  .Attr("T: {float, double}")
  .Input("radius: T")
  .Input("intensity: T")
  .Input("n_min: int32")
  .Input("n_max: int32")
  .Input("z: T")
  .Input("r: T")
  .Output("delta: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle shape;

    TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &shape));
    TF_RETURN_IF_ERROR(c->Merge(c->input(2), c->input(3), &shape));
    TF_RETURN_IF_ERROR(c->Merge(shape, c->input(4), &shape));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &shape));

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
template <typename Device, typename T>
class TransitDepthOp : public OpKernel {
 public:
  explicit TransitDepthOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& radius_tensor    = context->input(0);
    const Tensor& intensity_tensor = context->input(1);
    const Tensor& n_min_tensor     = context->input(2);
    const Tensor& n_max_tensor     = context->input(3);
    const Tensor& z_tensor         = context->input(4);
    const Tensor& r_tensor         = context->input(5);

    // Dimensions
    const int64 N = radius_tensor.NumElements();
    OP_REQUIRES(context, N <= tensorflow::kint32max,
                errors::InvalidArgument("too many elements in tensor"));
    OP_REQUIRES(context, intensity_tensor.NumElements() == N,
        errors::InvalidArgument("z and r must have the same number of elements"));

    const int64 size = z_tensor.NumElements();
    OP_REQUIRES(context, size <= tensorflow::kint32max,
                errors::InvalidArgument("too many elements in tensor"));
    OP_REQUIRES(context, n_min_tensor.NumElements() == size,
        errors::InvalidArgument("z and n_min must have the same number of elements"));
    OP_REQUIRES(context, n_max_tensor.NumElements() == size,
        errors::InvalidArgument("z and n_max must have the same number of elements"));
    OP_REQUIRES(context, r_tensor.NumElements() == 1,
        errors::InvalidArgument("r must be a scalar"));

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
    auto delta           = delta_tensor->template flat<T>();

    TransitDepthFunctor<Device, T>()(context->eigen_device<Device>(),
        static_cast<int>(N), radius.data(), intensity.data(),
        static_cast<int>(size), n_min.data(), n_max.data(), z.data(), r(0), delta.data());
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

#define REGISTER_GPU(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("TransitDepth").Device(DEVICE_GPU).TypeConstraint<type>("T"),         \
      TransitDepthOp<GPUDevice, type>)

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

#endif

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "transit_op.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
struct TransitFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int grid_size, const T* const x, const T* const grid,
                  int size, const int* const indmin, const int* const indmax, const T* const z, const T* const r, T* delta) {
    for (int i = 0; i < size; ++i) {
      delta[i] = transit::compute_delta<T>(grid_size, x, grid, indmin[i], indmax[i], z[i], r[i]);
    }
  }
};

REGISTER_OP("Transit")
  .Attr("T: {float, double}")
  .Input("grid: T")
  .Input("z: T")
  .Input("r: T")
  .Output("delta: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &shape));
    TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->input(2), &shape));
    c->set_output(0, c->input(1));
    return Status::OK();
  });

template <typename Device, typename T>
class TransitOp : public OpKernel {
 public:
  explicit TransitOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    typedef Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> ConstantArray;

    // Inputs
    const Tensor& grid_tensor = context->input(0);
    const Tensor& z_tensor = context->input(1);
    const Tensor& r_tensor = context->input(2);

    // Dimensions
    const int64 grid_size = grid_tensor.NumElements();
    const int64 size = z_tensor.NumElements();
    OP_REQUIRES(context, r_tensor.NumElements() == size,
        errors::InvalidArgument("z and r must have the same number of elements"));
    OP_REQUIRES(context, size <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    // Output
    Tensor* delta_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, z_tensor.shape(), &delta_tensor));

    // Access the data
    const auto grid = grid_tensor.template flat<T>();
    const auto z = ConstantArray(z_tensor.template flat<T>().data(), size, 1);
    const auto r = ConstantArray(r_tensor.template flat<T>().data(), size, 1);
    auto delta = delta_tensor->template flat<T>();

    auto val = 1.0 - Eigen::Array<T, Eigen::Dynamic, 1>::LinSpaced(grid_size, 0.0, 1.0);
    Eigen::Array<T, Eigen::Dynamic, 1> x = 1.0 - val * val;

    Eigen::Array<int, Eigen::Dynamic, 1> indmin = ((grid_size-1) * (1.0 - sqrt(1.0 - (z - r).max(0.0)))).floor().template cast<int>();
    Eigen::Array<int, Eigen::Dynamic, 1> indmax = ((grid_size-1) * (1.0 - sqrt(1.0 - (z + r).min(1.0)))).ceil().template cast<int>().min(grid_size - 1);

    TransitFunctor<Device, T>()(context->eigen_device<Device>(),
        static_cast<int>(grid_size), x.data(), grid.data(),
        static_cast<int>(size), indmin.data(), indmax.data(), z.data(), r.data(), delta.data());
  }
};

#define REGISTER_CPU(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Transit").Device(DEVICE_CPU).TypeConstraint<type>("T"),        \
      TransitOp<CPUDevice, type>)

REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Transit").Device(DEVICE_GPU).TypeConstraint<type>("T"),         \
      TransitOp<GPUDevice, type>)

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

#endif

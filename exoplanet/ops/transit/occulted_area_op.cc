#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <limits>

#include "transit_op.h"

using namespace tensorflow;
using namespace exoplanet;

REGISTER_OP("OccultedArea")
  .Attr("T: {float, double}")
  .Input("x: T")
  .Input("r: T")
  .Input("z: T")
  .Output("area: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle shape;

    TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &shape));
    TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->input(2), &shape));
    c->set_output(0, c->input(0));
    return Status::OK();
  });

template <typename T>
class OccultedAreaOp : public OpKernel {
 public:
  explicit OccultedAreaOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& x_tensor = context->input(0);
    const Tensor& r_tensor = context->input(1);
    const Tensor& z_tensor = context->input(2);

    // Dimensions
    const int64 N = x_tensor.NumElements();
    OP_REQUIRES(context, (r_tensor.NumElements() == N && z_tensor.NumElements() == N),
        errors::InvalidArgument("x, r, and z must have the same number of elements"));

    // Output
    Tensor* area_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_tensor.shape(), &area_tensor));

    // Access the data
    const auto x = x_tensor.template flat<T>();
    const auto r = r_tensor.template flat<T>();
    const auto z = z_tensor.template flat<T>();
    auto area    = area_tensor->template flat<T>();

    for (int i = 0; i < N; ++i) {
      area(i) = transit::compute_area<T>(x(i), r(i), z(i));
    }
  }
};

#define REGISTER_CPU(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("OccultedArea").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      OccultedAreaOp<type>)

REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU

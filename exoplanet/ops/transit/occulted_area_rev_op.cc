#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <limits>

#include "transit_op.h"

using namespace tensorflow;
using namespace exoplanet;

REGISTER_OP("OccultedAreaRev")
  .Attr("T: {float, double}")
  .Input("x: T")
  .Input("r: T")
  .Input("z: T")
  .Input("barea: T")
  .Output("br: T")
  .Output("bz: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle shape;

    TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &shape));
    TF_RETURN_IF_ERROR(c->Merge(c->input(1), c->input(2), &shape));
    TF_RETURN_IF_ERROR(c->Merge(c->input(2), c->input(3), &shape));
    c->set_output(0, c->input(0));
    c->set_output(1, c->input(0));
    return Status::OK();
  });

template <typename T>
class OccultedAreaRevOp : public OpKernel {
 public:
  explicit OccultedAreaRevOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& x_tensor = context->input(0);
    const Tensor& r_tensor = context->input(1);
    const Tensor& z_tensor = context->input(2);
    const Tensor& barea_tensor = context->input(3);

    // Dimensions
    const int64 N = x_tensor.NumElements();
    OP_REQUIRES(context, (r_tensor.NumElements() == N &&
                          z_tensor.NumElements() == N &&
                          barea_tensor.NumElements() == N),
        errors::InvalidArgument("x, r, z, and barea must have the same number of elements"));

    // Output
    Tensor* br_tensor = NULL;
    Tensor* bz_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_tensor.shape(), &br_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, x_tensor.shape(), &bz_tensor));

    // Access the data
    const auto x     = x_tensor.template flat<T>();
    const auto r     = r_tensor.template flat<T>();
    const auto z     = z_tensor.template flat<T>();
    const auto barea = barea_tensor.template flat<T>();
    auto br          = br_tensor->template flat<T>();
    auto bz          = bz_tensor->template flat<T>();

    for (int i = 0; i < N; ++i) {
      br(i) = 0.0;
      bz(i) = 0.0;
      transit::compute_area_fwd<T>(x(i), r(i), z(i), &(br(i)), &(bz(i)));
      br(i) *= barea(i);
      bz(i) *= barea(i);
    }
  }
};

#define REGISTER_CPU(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("OccultedAreaRev").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      OccultedAreaRevOp<type>)

REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU

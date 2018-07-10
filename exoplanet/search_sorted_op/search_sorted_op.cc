#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("SearchSorted")
  .Attr("T: {float, double}")
  .Input("a: T")
  .Input("v: T")
  .Output("inds: int64")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle a, v;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &a));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &v));
    c->set_output(0, c->input(1));
    return Status::OK();
  });

template <typename T>
class SearchSortedOp : public OpKernel {
 public:
  explicit SearchSortedOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& a_tensor = context->input(0);
    const Tensor& v_tensor = context->input(1);

    // Dimensions
    const int64 N  = a_tensor.NumElements();
    const int64 Nv = v_tensor.dim_size(0);
    const int64 M  = v_tensor.dim_size(1);

    // Output
    Tensor* inds_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, v_tensor.shape(), &inds_tensor));

    // Access the data
    const auto a = a_tensor.template flat<T>();
    const auto v = v_tensor.template matrix<T>();
    auto inds = inds_tensor->matrix<int64>();

    for (int64 k = 0; k < Nv; ++k) {
      int64 m = 0;

      while ((m < M) && (v(k, m) <= a(0))) {
        inds(k, m) = 0;
        m++;
      }
      if (m >= M) continue;

      for (int64 n = 0; n < N-1; ++n) {
        while (v(k, m) <= a(n+1)) {
          inds(k, m) = n+1;
          m++;
          if (m >= M) break;
        }
        if (m >= M) break;
      }
      if (m >= M) continue;

      while (m < M) {
        inds(k, m) = N;
        m++;
      }
    }
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("SearchSorted").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      SearchSortedOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL

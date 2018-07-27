#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <algorithm>

using namespace tensorflow;

REGISTER_OP("TriDiagSolve")
  .Attr("T: {float, double}")
  .Input("diag: T")
  .Input("upper: T")
  .Input("lower: T")
  .Input("y: T")
  .Output("x: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle diag, upper, lower, y;
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &diag));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &upper));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &lower));
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(3), 1, &y));

    // The shapes of the upper and lower diagonals must match
    TF_RETURN_IF_ERROR(c->Merge(upper, lower, &upper));

    // Make sure that the diagonal is one longer than the off-diagonals
    shape_inference::DimensionHandle upper_d = c->Dim(upper, -1);
    TF_RETURN_IF_ERROR(c->Add(upper_d, 1, &upper_d));
    TF_RETURN_IF_ERROR(c->Merge(upper_d, c->Dim(diag, -1), &upper_d));

    // Make sure that the inner dimensions of y match
    TF_RETURN_IF_ERROR(c->MergePrefix(y, diag, &y, &diag));

    c->set_output(0, c->input(3));

    return Status::OK();
  });

template <typename T>
class TriDiagSolveOp : public OpKernel {
 public:
  explicit TriDiagSolveOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ConstMatrix;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Matrix;

    // Inputs
    const Tensor& diag_tensor = context->input(0);
    const Tensor& upper_tensor = context->input(1);
    const Tensor& lower_tensor = context->input(2);
    const Tensor& y_tensor = context->input(3);

    // Check the dimensions
    const int64 n_dim_in = diag_tensor.dims();
    const int64 n_inner = diag_tensor.dim_size(n_dim_in-1);
    const int64 n_dim_out = y_tensor.dims();

    OP_REQUIRES(context, upper_tensor.shape() == lower_tensor.shape(),
        errors::InvalidArgument("upper and lower must be the same shape"));
    OP_REQUIRES(context, diag_tensor.dims() == n_dim_in,
        errors::InvalidArgument("upper, lower, and diag must be the same number of dimensions"));
    int64 n_in = 1;
    for (int64 n = 0; n < n_dim_in-1; ++n) {
      n_in *= diag_tensor.dim_size(n);
      OP_REQUIRES(context, upper_tensor.dim_size(n) == diag_tensor.dim_size(n),
          errors::InvalidArgument("upper, lower, and diag must have compatible dimensions"));
      OP_REQUIRES(context, y_tensor.dim_size(n) == diag_tensor.dim_size(n),
          errors::InvalidArgument("diag and y must have compatible dimensions"));
    }
    OP_REQUIRES(context, n_inner == upper_tensor.dim_size(n_dim_in-1)+1,
        errors::InvalidArgument("upper, lower, and diag must have compatible dimensions"));
    OP_REQUIRES(context, n_inner == y_tensor.dim_size(n_dim_in-1),
        errors::InvalidArgument("diag and y must have compatible dimensions"));
    int64 n_out = 1;
    for (int64 n = n_dim_in; n < n_dim_out; ++n) {
      n_out *= y_tensor.dim_size(n);
    }

    // Output
    Tensor* x_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, y_tensor.shape(), &x_tensor));

    // Access the data
    const auto diag  = ConstMatrix(diag_tensor.template flat<T>().data(), n_in, n_inner);
    const auto upper = ConstMatrix(upper_tensor.template flat<T>().data(), n_in, n_inner-1);
    const auto lower = ConstMatrix(lower_tensor.template flat<T>().data(), n_in, n_inner-1);

    Eigen::Matrix<T, Eigen::Dynamic, 1> c(n_inner);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> d(n_inner, n_out);

    for (int64 n = 0; n < n_in; ++n) {
      auto offset = n*n_inner*n_out;
      const auto y = ConstMatrix(y_tensor.template flat<T>().data()  + offset, n_inner, n_out);
      auto       x =      Matrix(x_tensor->template flat<T>().data() + offset, n_inner, n_out);

      // Ref: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
      c(0) = upper(n, 0) / diag(n, 0);
      d.row(0).noalias() = y.row(0) / diag(n, 0);

      for (int64 i = 1; i < n_inner-1; ++i) {
        T a = lower(n, i-1), b = diag(n, i), denom = b - a * c(i-1);
        c(i) = upper(n, i) / denom;
        d.row(i) = (y.row(i) - a * d.row(i-1)) / denom;
      }

      int64 i = n_inner - 1;
      T a = lower(n, i-1), denom = diag(n, i) - a * c(i-1);
      d.row(i) = (y.row(i) - a * d.row(i-1)) / denom;

      x.row(i) = d.row(i);
      for (int64 i = n_inner-2; i >= 0; --i) {
        x.row(i) = d.row(i) - c(i) * x.row(i+1);
      }
    }
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("TriDiagSolve").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      TriDiagSolveOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL

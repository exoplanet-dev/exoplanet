#section support_code

template <typename T1, typename T2, typename T3>
void do_diag_dot(const Eigen::MatrixBase<T1>& A, const Eigen::MatrixBase<T2>& B,
                 Eigen::MatrixBase<T3>& r) {
  for (int n = 0; n < A.rows(); ++n) {
    r(n) = A.row(n) * B.col(n);
  }
}

#section support_code_apply

// Apply-specific main function
int APPLY_SPECIFIC(diag_dot)(PyArrayObject* input0, PyArrayObject* input1,
                             PyArrayObject** output0) {
  using namespace exoplanet;

  int success = 0;
  npy_intp N, J;
  auto A_in = get_matrix_input<DTYPE_INPUT_0>(&N, &J, input0, &success, false);
  if (success) return 1;
  npy_intp shape_in[] = {J, N};
  auto B_in = get_input<DTYPE_INPUT_1>(2, shape_in, input1, &success, false);
  if (success) return 1;

  npy_intp shape_out[] = {N};
  auto r_out = allocate_output<DTYPE_OUTPUT_0>(1, shape_out, TYPENUM_OUTPUT_0,
                                               output0, &success);

  auto A_c = PyArray_IS_C_CONTIGUOUS(input0);
  auto A_f = PyArray_IS_F_CONTIGUOUS(input0);
  auto B_c = PyArray_IS_C_CONTIGUOUS(input1);
  auto B_f = PyArray_IS_F_CONTIGUOUS(input1);

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_0, Eigen::Dynamic, 1> > r(r_out, N);
  if (A_c && B_c) {
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor> >
        A(A_in, N, J);
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor> >
        B(B_in, J, N);
    do_diag_dot(A, B, r);
  } else if (A_c && B_f) {
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor> >
        A(A_in, N, J);
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor> >
        B(B_in, J, N);
    do_diag_dot(A, B, r);
  } else if (A_f && B_c) {
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor> >
        A(A_in, N, J);
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor> >
        B(B_in, J, N);
    do_diag_dot(A, B, r);
  } else if (A_f && B_f) {
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor> >
        A(A_in, N, J);
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::ColMajor> >
        B(B_in, J, N);
    do_diag_dot(A, B, r);
  } else {
    PyErr_Format(PyExc_ValueError, "invalid strides");
    return 1;
  }

  return 0;
}

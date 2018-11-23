#section support_code

template <typename T1, typename T2, typename T3>
void do_diag_dot (const Eigen::MatrixBase<T1>& A, const Eigen::MatrixBase<T2>& B, Eigen::MatrixBase<T3>& r) {
  for (int n = 0; n < A.rows(); ++n) {
    r(n) = A.row(n) * B.col(n);
  }
}

#section support_code_apply

// Apply-specific main function
int APPLY_SPECIFIC(diag_dot)(
    PyArrayObject* input0,
    PyArrayObject* input1,
    PyArrayObject** output0)
{

  if (input0 == NULL || PyArray_NDIM(input0) != 2) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }
  npy_intp N = PyArray_DIMS(input0)[0];
  npy_intp J = PyArray_DIMS(input0)[1];
  if (input1 == NULL || PyArray_NDIM(input1) != 2 || PyArray_DIMS(input1)[0] != J || PyArray_DIMS(input1)[1] != N) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }

  npy_intp shape0[] = {N};
  if (allocate_output(1, shape0, TYPENUM_OUTPUT_0, output0)) {
    return 1;
  }

  auto A_c = PyArray_IS_C_CONTIGUOUS(input0);
  auto A_f = PyArray_IS_F_CONTIGUOUS(input0);
  auto B_c = PyArray_IS_C_CONTIGUOUS(input1);
  auto B_f = PyArray_IS_F_CONTIGUOUS(input1);

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_0, Eigen::Dynamic, 1> > r((DTYPE_OUTPUT_0*)PyArray_DATA(*output0), N);
  if (A_c && B_c) {
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > A((DTYPE_INPUT_0*)PyArray_DATA(input0), N, J);
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > B((DTYPE_INPUT_1*)PyArray_DATA(input1), J, N);
    do_diag_dot(A, B, r);
  } else if (A_c && B_f) {
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > A((DTYPE_INPUT_0*)PyArray_DATA(input0), N, J);
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > B((DTYPE_INPUT_1*)PyArray_DATA(input1), J, N);
    do_diag_dot(A, B, r);
  } else if (A_f && B_c) {
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > A((DTYPE_INPUT_0*)PyArray_DATA(input0), N, J);
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > B((DTYPE_INPUT_1*)PyArray_DATA(input1), J, N);
    do_diag_dot(A, B, r);
  } else if (A_f && B_f) {
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > A((DTYPE_INPUT_0*)PyArray_DATA(input0), N, J);
    Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > B((DTYPE_INPUT_1*)PyArray_DATA(input1), J, N);
    do_diag_dot(A, B, r);
  } else {
    PyErr_Format(PyExc_ValueError, "invalid strides");
    return 1;
  }

  return 0;
}

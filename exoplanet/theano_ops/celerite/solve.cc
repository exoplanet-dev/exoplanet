#section support_code_apply

// Apply-specific main function
int APPLY_SPECIFIC(solve)(
    PyArrayObject* input0,
    PyArrayObject* input1,
    PyArrayObject* input2,
    PyArrayObject* input3,
    PyArrayObject* input4,
    PyArrayObject** output0,
    PyArrayObject** output1,
    PyArrayObject** output2)
{
  npy_intp N, J, Nrhs;
  int success = get_dimensions(input0, &N, &J);
  success += get_dimensions(input4, &N, &Nrhs);
  if (success) return 1;
  if (CELERITE_J != Eigen::Dynamic && J != CELERITE_J) {
    PyErr_Format(PyExc_ValueError, "runtime value of J does not match compiled value");
    return 1;
  }
  if (CELERITE_NRHS != Eigen::Dynamic && Nrhs != CELERITE_NRHS) {
    PyErr_Format(PyExc_ValueError, "runtime value of n_rhs does not match compiled value");
    return 1;
  }
  npy_intp input0_shape[] = {N, J};
  success += check_input(2, input0_shape, input0);
  npy_intp input1_shape[] = {N-1, J};
  success += check_input(2, input1_shape, input1);
  npy_intp input2_shape[] = {N};
  success += check_input(1, input2_shape, input2);
  npy_intp input3_shape[] = {N, J};
  success += check_input(2, input3_shape, input3);
  if (success) return 1;

  npy_intp shape0[] = {N, Nrhs};
  success += allocate_output(2, shape0, TYPENUM_OUTPUT_0, output0);
  npy_intp shape1[] = {N, J*Nrhs};
  success += allocate_output(2, shape1, TYPENUM_OUTPUT_1, output1);
  npy_intp shape2[] = {N, J*Nrhs};
  success += allocate_output(2, shape2, TYPENUM_OUTPUT_2, output2);
  if (success) {
    return 1;
  }

  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER> > U((DTYPE_INPUT_0*)PyArray_DATA(input0), N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER> > P((DTYPE_INPUT_1*)PyArray_DATA(input1), N-1, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_2, Eigen::Dynamic, 1> > d((DTYPE_INPUT_2*)PyArray_DATA(input2), N);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_3, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER> > W((DTYPE_INPUT_3*)PyArray_DATA(input3), N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_4, Eigen::Dynamic, CELERITE_NRHS, CELERITE_NRHS_ORDER> > Y((DTYPE_INPUT_4*)PyArray_DATA(input4), N, Nrhs);

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_0, Eigen::Dynamic, CELERITE_NRHS, CELERITE_NRHS_ORDER> > Z((DTYPE_OUTPUT_0*)PyArray_DATA(*output0), N, Nrhs);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_1, Eigen::Dynamic, CELERITE_JNRHS, CELERITE_JNRHS_ORDER> > F((DTYPE_OUTPUT_1*)PyArray_DATA(*output1), N, J*Nrhs);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_2, Eigen::Dynamic, CELERITE_JNRHS, CELERITE_JNRHS_ORDER> > G((DTYPE_OUTPUT_2*)PyArray_DATA(*output2), N, J*Nrhs);

  Z = Y;
  celerite::solve(U, P, d, W, Z, F, G);

  return 0;
}

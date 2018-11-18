#section support_code_apply

// Apply-specific main function
int APPLY_SPECIFIC(factor)(
    PyArrayObject* input0,
    PyArrayObject* input1,
    PyArrayObject* input2,
    PyArrayObject* input3,
    PyArrayObject** output0,
    PyArrayObject** output1,
    PyArrayObject** output2,
    PyArrayObject** output3)
{
  npy_intp N, J;
  int success = get_dimensions(input1, &N, &J);
  if (success) return 1;
  if (CELERITE_J != Eigen::Dynamic && J != CELERITE_J) {
    PyErr_Format(PyExc_ValueError, "runtime value of J does not match compiled value");
    return 1;
  }
  npy_intp input0_shape[] = {N};
  success += check_input(1, input0_shape, input0);
  npy_intp input2_shape[] = {N, J};
  success += check_input(2, input2_shape, input2);
  npy_intp input3_shape[] = {N-1, J};
  success += check_input(2, input3_shape, input3);
  if (success) return 1;

  npy_intp shape0[] = {N};
  success += allocate_output(1, shape0, TYPENUM_OUTPUT_0, output0);
  npy_intp shape1[] = {N, J};
  success += allocate_output(2, shape1, TYPENUM_OUTPUT_1, output1);
  npy_intp shape2[] = {N, J*J};
  success += allocate_output(2, shape2, TYPENUM_OUTPUT_2, output2);
  npy_intp shape3[] = {};
  success += allocate_output(0, shape3, TYPENUM_OUTPUT_3, output3);
  if (success) {
    return 1;
  }

  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, 1> > a((DTYPE_INPUT_0*)PyArray_DATA(input0), N);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER> > U((DTYPE_INPUT_1*)PyArray_DATA(input1), N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_2, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER> > V((DTYPE_INPUT_2*)PyArray_DATA(input2), N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_3, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER> > P((DTYPE_INPUT_3*)PyArray_DATA(input3), N-1, J);

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_0, Eigen::Dynamic, 1> > d((DTYPE_OUTPUT_0*)PyArray_DATA(*output0), N);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_1, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER> > W((DTYPE_OUTPUT_1*)PyArray_DATA(*output1), N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_2, Eigen::Dynamic, CELERITE_J2, CELERITE_J_ORDER> > S((DTYPE_OUTPUT_2*)PyArray_DATA(*output2), N, J*J);

  DTYPE_OUTPUT_3* flag_out = (DTYPE_OUTPUT_3*)PyArray_DATA(*output3);

  d = a;
  W = V;
  S.setZero();
  *flag_out = celerite::factor(U, P, d, W, S);

  return 0;
}

#section support_code_apply

// Apply-specific main function
int APPLY_SPECIFIC(solve_rev)(
    PyArrayObject* input0,
    PyArrayObject* input1,
    PyArrayObject* input2,
    PyArrayObject* input3,
    PyArrayObject* input4,
    PyArrayObject* input5,
    PyArrayObject* input6,
    PyArrayObject* input7,
    PyArrayObject** output0,
    PyArrayObject** output1,
    PyArrayObject** output2,
    PyArrayObject** output3,
    PyArrayObject** output4)
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
  npy_intp input5_shape[] = {N, J*Nrhs};
  success += check_input(2, input5_shape, input5);
  npy_intp input6_shape[] = {N, J*Nrhs};
  success += check_input(2, input6_shape, input6);
  npy_intp input7_shape[] = {N, Nrhs};
  success += check_input(2, input7_shape, input7);
  if (success) return 1;

  npy_intp shape0[] = {N, J};
  success += allocate_output(2, shape0, TYPENUM_OUTPUT_0, output0);
  npy_intp shape1[] = {N-1, J};
  success += allocate_output(2, shape1, TYPENUM_OUTPUT_1, output1);
  npy_intp shape2[] = {N};
  success += allocate_output(1, shape2, TYPENUM_OUTPUT_2, output2);
  npy_intp shape3[] = {N, J};
  success += allocate_output(2, shape3, TYPENUM_OUTPUT_3, output3);
  npy_intp shape4[] = {N, Nrhs};
  success += allocate_output(2, shape4, TYPENUM_OUTPUT_4, output4);
  if (success) {
    return 1;
  }

  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER> >         U((DTYPE_INPUT_0*)PyArray_DATA(input0), N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER> >         P((DTYPE_INPUT_1*)PyArray_DATA(input1), N-1, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_2, Eigen::Dynamic, 1> >                                    d((DTYPE_INPUT_2*)PyArray_DATA(input2), N);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_3, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER> >         W((DTYPE_INPUT_3*)PyArray_DATA(input3), N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_4, Eigen::Dynamic, CELERITE_NRHS, CELERITE_NRHS_ORDER> >   Z((DTYPE_INPUT_4*)PyArray_DATA(input4), N, Nrhs);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_5, Eigen::Dynamic, CELERITE_JNRHS, CELERITE_JNRHS_ORDER> > F((DTYPE_INPUT_5*)PyArray_DATA(input5), N, J*Nrhs);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_6, Eigen::Dynamic, CELERITE_JNRHS, CELERITE_JNRHS_ORDER> > G((DTYPE_INPUT_6*)PyArray_DATA(input6), N, J*Nrhs);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_7, Eigen::Dynamic, CELERITE_NRHS, CELERITE_NRHS_ORDER> >   bZ((DTYPE_INPUT_7*)PyArray_DATA(input7), N, Nrhs);

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_0, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER> >        bU((DTYPE_OUTPUT_0*)PyArray_DATA(*output0), N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_1, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER> >        bP((DTYPE_OUTPUT_1*)PyArray_DATA(*output1), N-1, J);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_2, Eigen::Dynamic, 1> >                                   bd((DTYPE_OUTPUT_2*)PyArray_DATA(*output2), N);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_3, Eigen::Dynamic, CELERITE_J, CELERITE_J_ORDER> >        bW((DTYPE_OUTPUT_3*)PyArray_DATA(*output3), N, J);
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_4, Eigen::Dynamic, CELERITE_NRHS, CELERITE_NRHS_ORDER> >  bY((DTYPE_OUTPUT_4*)PyArray_DATA(*output4), N, Nrhs);

  bU.setZero();
  bP.setZero();
  bd.setZero();
  bW.setZero();
  bY.setZero();
  celerite::solve_grad(U, P, d, W, Z, F, G, bZ, bU, bP, bd, bW, bY);

  return 0;
}

#section support_code_apply

int APPLY_SPECIFIC(circular_contact_points)(
    PyArrayObject*  input0,
    PyArrayObject*  input1,
    PyArrayObject*  input2,
    PyArrayObject*  input3,
    PyArrayObject** output0,
    PyArrayObject** output1,
    PyArrayObject** output2,
    PyArrayObject** output3,
    PARAMS_TYPE* params)
{
  typedef DTYPE_OUTPUT_0 T;

  long maxiter = params->maxiter;
  double tol = params->tol;

  npy_intp N, M;
  int success = get_size(input0, &N);
  success += get_size(input1, &M);
  if (M != N) success = -100;
  success += get_size(input2, &M);
  if (M != N) success = -100;
  success += get_size(input3, &M);
  if (M != N) success = -100;
  if (success < 0) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }
  if (success) return 1;

  npy_intp ndim = PyArray_NDIM(input0);
  npy_intp* shape = PyArray_DIMS(input0);
  success += allocate_output(ndim, shape, TYPENUM_OUTPUT_0, output0);
  success += allocate_output(ndim, shape, TYPENUM_OUTPUT_1, output1);
  success += allocate_output(ndim, shape, TYPENUM_OUTPUT_2, output2);
  success += allocate_output(ndim, shape, TYPENUM_OUTPUT_3, output3);
  if (success) {
    return 1;
  }

  DTYPE_INPUT_0*  a_in  = (DTYPE_INPUT_0*)PyArray_DATA(input0);
  DTYPE_INPUT_1*  i_in  = (DTYPE_INPUT_1*)PyArray_DATA(input1);
  DTYPE_INPUT_2*  r_in  = (DTYPE_INPUT_2*)PyArray_DATA(input2);
  DTYPE_INPUT_3*  R_in  = (DTYPE_INPUT_3*)PyArray_DATA(input3);

  DTYPE_OUTPUT_0* M1_out = (DTYPE_OUTPUT_0*)PyArray_DATA(*output0);
  DTYPE_OUTPUT_1* M2_out = (DTYPE_OUTPUT_1*)PyArray_DATA(*output1);
  DTYPE_OUTPUT_2* M3_out = (DTYPE_OUTPUT_2*)PyArray_DATA(*output2);
  DTYPE_OUTPUT_3* M4_out = (DTYPE_OUTPUT_3*)PyArray_DATA(*output3);

  for (npy_intp n = 0; n < N; ++n) {
    contact_points::CircularContactPoint<T> solver(a_in[n], i_in[n]);
    solver.find_contacts(R_in[n], r_in[n], &(M1_out[n]), &(M2_out[n]), &(M3_out[n]), &(M4_out[n]), maxiter, tol);
  }

  return 0;
}

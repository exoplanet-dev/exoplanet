#section support_code_apply

int APPLY_SPECIFIC(contact_points)(
    PyArrayObject*  input0,
    PyArrayObject*  input1,
    PyArrayObject*  input2,
    PyArrayObject*  input3,
    PyArrayObject*  input4,
    PyArrayObject*  input5,
    PyArrayObject** output0,
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
  success += get_size(input4, &M);
  if (M != N) success = -100;
  success += get_size(input5, &M);
  if (M != N) success = -100;

  if (success < 0) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }
  if (success) return 1;

  npy_intp ndim = PyArray_NDIM(input0);
  std::vector<npy_intp> shape(ndim+1);
  for (npy_intp n = 0; n < ndim; ++n) shape[n] = PyArray_DIM(input0, n);
  shape[ndim] = 4;
  success += allocate_output(ndim + 1, &(shape[0]), TYPENUM_OUTPUT_0, output0);
  if (success) {
    return 1;
  }

  DTYPE_INPUT_0*  a_in  = (DTYPE_INPUT_0*)PyArray_DATA(input0);
  DTYPE_INPUT_1*  e_in  = (DTYPE_INPUT_1*)PyArray_DATA(input1);
  DTYPE_INPUT_2*  w_in  = (DTYPE_INPUT_2*)PyArray_DATA(input2);
  DTYPE_INPUT_3*  i_in  = (DTYPE_INPUT_3*)PyArray_DATA(input3);
  DTYPE_INPUT_4*  r_in  = (DTYPE_INPUT_4*)PyArray_DATA(input4);
  DTYPE_INPUT_5*  R_in  = (DTYPE_INPUT_5*)PyArray_DATA(input5);

  DTYPE_OUTPUT_0* f_out = (DTYPE_OUTPUT_0*)PyArray_DATA(*output0);

  for (npy_intp n = 0; n < N; ++n) {
    T* f0 = &(f_out[n*4]);
    contact_points::ContactPoint<T> solver(a_in[n], e_in[n], w_in[n], i_in[n]);
    solver.find_contacts(R_in[n], r_in[n], f0, &(f0[1]), &(f0[2]), &(f0[3]), maxiter, tol);
  }

  return 0;
}

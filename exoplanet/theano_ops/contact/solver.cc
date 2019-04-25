#section support_code_apply

int APPLY_SPECIFIC(solver)(
    PyArrayObject*  input0,
    PyArrayObject*  input1,
    PyArrayObject*  input2,
    PyArrayObject*  input3,
    PyArrayObject*  input4,
    PyArrayObject*  input5,
    PyArrayObject*  input6,
    PyArrayObject** output0,
    PyArrayObject** output1,
    PyArrayObject** output2,
    PARAMS_TYPE* params)
{
  typedef DTYPE_OUTPUT_0 T;

  double tol = params->tol;

  npy_intp N, N1, N2, N3, N4, N5, N6;
  int success = get_size(input0, &N);
  success += get_size(input1, &N1);
  success += get_size(input2, &N2);
  success += get_size(input3, &N3);
  success += get_size(input4, &N4);
  success += get_size(input5, &N5);
  success += get_size(input6, &N6);
  if (success) return 1;
  if (N != N1 || N != N2 || N != N3 || N != N4 || N != N5 || N != N6) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }

  success += allocate_output(PyArray_NDIM(input0), PyArray_DIMS(input0), TYPENUM_OUTPUT_0, output0);
  success += allocate_output(PyArray_NDIM(input0), PyArray_DIMS(input0), TYPENUM_OUTPUT_1, output1);
  success += allocate_output(PyArray_NDIM(input0), PyArray_DIMS(input0), TYPENUM_OUTPUT_2, output2);
  if (success) {
    return 1;
  }

  DTYPE_INPUT_0*  a     = (DTYPE_INPUT_0*)PyArray_DATA(input0);
  DTYPE_INPUT_1*  e     = (DTYPE_INPUT_1*)PyArray_DATA(input1);
  DTYPE_INPUT_2*  cosw  = (DTYPE_INPUT_2*)PyArray_DATA(input2);
  DTYPE_INPUT_3*  sinw  = (DTYPE_INPUT_3*)PyArray_DATA(input3);
  DTYPE_INPUT_4*  cosi  = (DTYPE_INPUT_4*)PyArray_DATA(input4);
  DTYPE_INPUT_5*  sini  = (DTYPE_INPUT_5*)PyArray_DATA(input5);
  DTYPE_INPUT_6*  L     = (DTYPE_INPUT_6*)PyArray_DATA(input6);

  DTYPE_OUTPUT_0* M_left  = (DTYPE_OUTPUT_0*)PyArray_DATA(*output0);
  DTYPE_OUTPUT_1* M_right = (DTYPE_OUTPUT_1*)PyArray_DATA(*output1);
  DTYPE_OUTPUT_2* flag    = (DTYPE_OUTPUT_2*)PyArray_DATA(*output2);

  for (npy_intp n = 0; n < N; ++n) {
    auto const solver = contact_points::ContactPointSolver<T>(a[n], e[n], cosw[n], sinw[n], cosi[n], sini[n]);
    auto const roots = solver.find_roots(L[n], tol);
    flag[n] = std::get<0>(roots);
    M_left[n] = std::get<1>(roots);
    M_right[n] = std::get<2>(roots);
  }

  return 0;
}

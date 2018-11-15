#section support_code_apply

int APPLY_SPECIFIC(limbdark)(
    PyArrayObject* input0,  // Array of "cl"
    PyArrayObject* input1,  // Array of impact parameters "b"
    PyArrayObject* input2,  // Array of radius ratios "r"
    PyArrayObject* input3,  // Array of line-of-sight position "los"
    PyArrayObject** output0)
{
  npy_intp Nc, Nb, Nr, Nlos;
  int success = get_size(input0, &Nc);
  success += get_size(input1, &Nb);
  success += get_size(input2, &Nr);
  success += get_size(input3, &Nr);
  if (success) return 1;
  if (Nb != Nr || Nb != Nlos) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }

  success += allocate_output(PyArray_NDIM(input1), PyArray_DIMS(input1), TYPENUM_OUTPUT_0, output0);
  if (success) {
    Py_XDECREF(*output0);
    return 1;
  }

  DTYPE_INPUT_0*  c  = (DTYPE_INPUT_0*) PyArray_DATA(input0);
  DTYPE_INPUT_1*  b  = (DTYPE_INPUT_1*) PyArray_DATA(input1);
  DTYPE_INPUT_2*  r  = (DTYPE_INPUT_2*) PyArray_DATA(input2);
  DTYPE_INPUT_3* los = (DTYPE_INPUT_3*) PyArray_DATA(input3);
  DTYPE_OUTPUT_0* f  = (DTYPE_OUTPUT_0*)PyArray_DATA(*output0);

  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, 1>> cvec(c, Nc);
  starry::limbdark::GreensLimbDark<DTYPE_OUTPUT_0> L(Nc-1);

  for (npy_intp i = 0; i < Nb; ++i) {
    f[i] = 1;
    if (los[i] < 0) {
      auto b_ = std::abs(b[i]);
      auto r_ = std::abs(r[i]);
      if (b_ < 1 + r_) {
        L.compute(b_, r_, false);
        f[i] = L.S.dot(cvec);
      }
    }
  }

  return 0;
}

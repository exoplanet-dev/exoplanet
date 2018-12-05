#section support_code_apply

int APPLY_SPECIFIC(limbdark)(
    PyArrayObject* input0,    // Array of "cl"
    PyArrayObject* input1,    // Array of impact parameters "b"
    PyArrayObject* input2,    // Array of radius ratios "r"
    PyArrayObject* input3,    // Array of line-of-sight position "los"
    PyArrayObject** output0,  // Flux
    PyArrayObject** output1,  // dfdcl
    PyArrayObject** output2,  // dfdb
    PyArrayObject** output3   // dfdr
  )
{
  npy_intp Nc, Nb, Nr, Nlos;
  int success = get_size(input0, &Nc);
  success += get_size(input1, &Nb);
  success += get_size(input2, &Nr);
  success += get_size(input3, &Nlos);
  if (success) return 1;
  if (Nb != Nr || Nb != Nlos) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch %d %d %d", Nb, Nr, Nlos);
    return 1;
  }

  npy_intp ndim = PyArray_NDIM(input1);
  npy_intp* dims = PyArray_DIMS(input1);
  std::vector<npy_intp> shape(ndim + 1);
  shape[0] = Nc;
  for (npy_intp i = 0; i < ndim; ++i) shape[i+1] = dims[i];

  success += allocate_output(ndim, dims, TYPENUM_OUTPUT_0, output0);
  success += allocate_output(ndim+1, &(shape[0]), TYPENUM_OUTPUT_1, output1);
  success += allocate_output(ndim, dims, TYPENUM_OUTPUT_2, output2);
  success += allocate_output(ndim, dims, TYPENUM_OUTPUT_3, output3);
  if (success) {
    return 1;
  }

  DTYPE_INPUT_0* c   = (DTYPE_INPUT_0*) PyArray_DATA(input0);
  DTYPE_INPUT_1* b   = (DTYPE_INPUT_1*) PyArray_DATA(input1);
  DTYPE_INPUT_2* r   = (DTYPE_INPUT_2*) PyArray_DATA(input2);
  DTYPE_INPUT_3* los = (DTYPE_INPUT_3*) PyArray_DATA(input3);

  DTYPE_OUTPUT_0* f     = (DTYPE_OUTPUT_0*)PyArray_DATA(*output0);
  DTYPE_OUTPUT_1* dfdcl = (DTYPE_OUTPUT_1*)PyArray_DATA(*output1);
  DTYPE_OUTPUT_2* dfdb  = (DTYPE_OUTPUT_2*)PyArray_DATA(*output2);
  DTYPE_OUTPUT_3* dfdr  = (DTYPE_OUTPUT_3*)PyArray_DATA(*output3);

  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dfdcl_mat(dfdcl, Nc, Nb);
  dfdcl_mat.setZero();

  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, 1>> cvec(c, Nc);
  starry::limbdark::GreensLimbDark<DTYPE_OUTPUT_0> L(Nc-1);

  for (npy_intp i = 0; i < Nb; ++i) {
    f[i] = 0;
    dfdb[i] = 0;
    dfdr[i] = 0;

    if (los[i] < 0) {
      auto b_ = std::abs(b[i]);
      auto r_ = std::abs(r[i]);
      if (b_ < 1 + r_) {
        L.compute(b_, r_, true);

        // The value of the light curve
        f[i] = L.S.dot(cvec) - 1;

        // The gradients
        dfdcl_mat.col(i) = L.S;
        dfdb[i] = sgn(b[i]) * L.dSdb.dot(cvec);
        dfdr[i] = sgn(r[i]) * L.dSdr.dot(cvec);
      }
    }
  }

  return 0;
}

#section support_code_apply

// Apply-specific main function
int APPLY_SPECIFIC(factor)(
    PyArrayObject* points0_obj,
#ifdef REGULAR_GRID_1
    PyArrayObject* points1_obj,
#endif
#ifdef REGULAR_GRID_2
    PyArrayObject* points2_obj,
#endif
#ifdef REGULAR_GRID_3
    PyArrayObject* points3_obj,
#endif
#ifdef REGULAR_GRID_4
    PyArrayObject* points4_obj,
#endif
    PyArrayObject* values_obj,
    PyArrayObject* xi_obj,
    PyArrayObject** zi_obj,
    PyArrayObject** dz_obj)
{
  npy_intp shape[REGULAR_GRID_NDIM + 1];
  int success = get_dimensions(points0_obj, shape);

#ifdef REGULAR_GRID_1
  success += get_dimensions(points1_obj, &(shape[1]));
#endif
#ifdef REGULAR_GRID_2
  success += get_dimensions(points2_obj, &(shape[2]));
#endif
#ifdef REGULAR_GRID_3
  success += get_dimensions(points3_obj, &(shape[3]));
#endif
#ifdef REGULAR_GRID_4
  success += get_dimensions(points4_obj, &(shape[4]));
#endif
  if (success) return 1;

  // Sort out the values shape
  // must be (nx, ny, ..., nout)
  if (value_obj == NULL || PyArray_NDIM(value_obj) != REGULAR_GRID_NDIM + 1 || !PyArray_CHKFLAGS(values_obj, NPY_ARRAY_C_CONTIGUOUS)) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }
  npy_intp nout = PyArray_DIM(values_obj, REGULAR_GRID_NDIM);
  if (REGULAR_GRID_NOUT != Eigen::Dynamic && REGULAR_GRID_NOUT != nout) {
    PyErr_Format(PyExc_ValueError, "number of outputs does not match compiled number");
    return 1;
  }
  for (npy_intp n = 0; n < REGULAR_GRID_NDIM; ++n) {
    if (shape[n] != PyArray_DIM(values, n)) {
      PyErr_Format(PyExc_ValueError, "size of values dimension %d does not match", n);
      return 1;
    }
  }

  // Sort out the test points
  // must be (ntest, ndim)
  if (xi_obj == NULL || PyArray_NDIM(xi_obj) != 2 || PyArray_DIM(xi_obj, 1) != ndim || !PyArray_CHKFLAGS(values_obj, NPY_ARRAY_C_CONTIGUOUS)) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch");
    return 1;
  }
  npy_intp ntest = PyArray_DIM(xi_obj, 0);

  npy_intp shape0[] = {ntest, nout};
  success += allocate_output(2, shape0, TYPENUM_OUTPUT_0, zi_obj);
  npy_intp shape1[] = {ntest, ndim, nout};
  success += allocate_output(3, shape1, TYPENUM_OUTPUT_1, dz_obj);
  if (success) return 1;

  return 0;
}

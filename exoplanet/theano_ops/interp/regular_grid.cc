#section support_code_apply

// Apply-specific main function
int APPLY_SPECIFIC(regular_grid)(PyArrayObject* xi_obj,
                                 PyArrayObject* values_obj,
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
                                 PyArrayObject** zi_obj, PyArrayObject** dz_obj,
                                 PARAMS_TYPE* params) {
  bool check_sorted = params->check_sorted;
  bool bounds_error = params->bounds_error;

  const npy_intp ndim = REGULAR_GRID_NDIM;
  npy_intp shape[REGULAR_GRID_NDIM];
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
  if (values_obj == NULL ||
      PyArray_NDIM(values_obj) != (REGULAR_GRID_NDIM + 1) ||
      !PyArray_CHKFLAGS(values_obj, NPY_ARRAY_C_CONTIGUOUS)) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch (values)");
    return 1;
  }
  npy_intp nout = PyArray_DIM(values_obj, REGULAR_GRID_NDIM);
  if (REGULAR_GRID_NOUT != Eigen::Dynamic && REGULAR_GRID_NOUT != nout) {
    PyErr_Format(PyExc_ValueError,
                 "number of outputs does not match compiled number");
    return 1;
  }
  npy_intp ngrid = 1;
  for (npy_intp n = 0; n < REGULAR_GRID_NDIM; ++n) {
    ngrid *= shape[n];
    if (shape[n] != PyArray_DIM(values_obj, n)) {
      PyErr_Format(PyExc_ValueError,
                   "size of values dimension %d does not match", n);
      return 1;
    }
  }

  // Sort out the test points
  // must be (ntest, ndim)
  if (xi_obj == NULL || PyArray_NDIM(xi_obj) != 2 ||
      PyArray_DIM(xi_obj, 1) != ndim ||
      !PyArray_CHKFLAGS(xi_obj, NPY_ARRAY_C_CONTIGUOUS)) {
    PyErr_Format(PyExc_ValueError, "dimension mismatch (xi)");
    return 1;
  }
  npy_intp ntest = PyArray_DIM(xi_obj, 0);

  // Allocate the output
  npy_intp shape0[] = {ntest, nout};
  success += allocate_output(2, shape0, TYPENUM_OUTPUT_0, zi_obj);
  npy_intp shape1[] = {ntest, ndim, nout};
  success += allocate_output(3, shape1, TYPENUM_OUTPUT_1, dz_obj);
  if (success) return 1;

  // Cast the values and test points as a matrix with the right shape
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_0, Eigen::Dynamic, REGULAR_GRID_NDIM,
                           REGULAR_GRID_NDIM_ORDER>>
  xi((DTYPE_INPUT_0*)PyArray_DATA(xi_obj), ntest, ndim);
  Eigen::Map<Eigen::Matrix<DTYPE_INPUT_1, Eigen::Dynamic, REGULAR_GRID_NOUT,
                           REGULAR_GRID_NOUT_ORDER>>
  values((DTYPE_INPUT_1*)PyArray_DATA(values_obj), ngrid, nout);

  // Outputs
  Eigen::Map<Eigen::Matrix<DTYPE_OUTPUT_0, Eigen::Dynamic, REGULAR_GRID_NOUT,
                           REGULAR_GRID_NOUT_ORDER>>
  zi((DTYPE_OUTPUT_0*)PyArray_DATA(*zi_obj), ntest, nout);
  Eigen::Map<
      Eigen::Matrix<DTYPE_OUTPUT_1, Eigen::Dynamic, REGULAR_GRID_NDIM_NOUT,
                    REGULAR_GRID_NDIM_NOUT_ORDER>>
  dz((DTYPE_OUTPUT_1*)PyArray_DATA(*dz_obj), ntest, ndim * nout);

  // Allocate temporary arrays to store indices and weights
  typedef DTYPE_OUTPUT_0 T;
  Eigen::Matrix<npy_intp, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> inds(
      ntest, ndim);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> numerator(
      ntest, ndim);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> denominator(
      ntest, ndim);
  Eigen::Matrix<T, Eigen::Dynamic, 1> accumulator(ndim);

  std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>> points_vec(ndim);
  points_vec[0] = get_points_grid<DTYPE_INPUT_2, T>(points0_obj);
#ifdef REGULAR_GRID_1
  points_vec[1] = get_points_grid<DTYPE_INPUT_3, T>(points1_obj);
#endif
#ifdef REGULAR_GRID_2
  points_vec[2] = get_points_grid<DTYPE_INPUT_4, T>(points2_obj);
#endif
#ifdef REGULAR_GRID_3
  points_vec[3] = get_points_grid<DTYPE_INPUT_5, T>(points3_obj);
#endif
#ifdef REGULAR_GRID_4
  points_vec[4] = get_points_grid<DTYPE_INPUT_6, T>(points4_obj);
#endif

  // Loop over dimensions and compute the indices of each test point in each
  // grid
  for (npy_intp dim = 0; dim < ndim; ++dim) {
    auto points = points_vec[dim];
    npy_intp N = shape[dim];
    if (check_sorted) {
      for (npy_intp n = 0; n < N - 1; ++n)
        if (points(n + 1) <= points(n)) {
          PyErr_Format(PyExc_ValueError,
                       "each tensor in 'points' must be sorted");
          return 1;
        }
    }

    // Find where the point should be inserted into the grid
    for (npy_intp n = 0; n < ntest; ++n) {
      bool out_of_bounds = false;
      npy_intp ind = search_sorted<T>(N, (T*)points.data(), xi(n, dim)) - 1;
      if (ind < 0) {
        out_of_bounds = true;
        ind = 0;
      }
      if (ind > N - 2) {
        out_of_bounds = true;
        ind = N - 2;
      }
      if (bounds_error) {
        if (out_of_bounds) {
          PyErr_Format(PyExc_ValueError,
                       "target point out of bounds n=%d dim=%d", n, dim);
          return 1;
        }
      }
      inds(n, dim) = ind;
      numerator(n, dim) = xi(n, dim) - points(ind);
      denominator(n, dim) = points(ind + 1) - points(ind);
    }
  }

  // Loop over test points and compute the interpolation for that point
  unsigned ncorner = pow(2, ndim);
  for (int n = 0; n < ntest; ++n) {
    // Madness to find the coordinates of every corner
    zi.row(n).setZero();
    dz.row(n).setZero();
    for (unsigned corner = 0; corner < ncorner; ++corner) {
      npy_intp factor = 1;
      npy_intp ind = 0;
      T weight = T(1.0);
      for (int dim = ndim - 1; dim >= 0; --dim) {
        unsigned offset = (corner >> unsigned(dim)) & 1;
        ind += factor * (inds(n, dim) + offset);
        factor *= shape[dim];
        if (offset == 1) {
          weight *= numerator(n, dim) / denominator(n, dim);
          accumulator(dim) = numerator(n, dim);
        } else {
          // T norm_dist = T(1.0) - numerator(n, dim) / denominator(n, dim);
          weight *= T(1.0) - numerator(n, dim) / denominator(n, dim);
          accumulator(dim) = (numerator(n, dim) - denominator(n, dim));
        }
      }

      if (std::abs(weight) > std::numeric_limits<T>::epsilon()) {
        zi.row(n).noalias() += weight * values.row(ind);
        for (int dim = 0; dim < ndim; ++dim) {
          dz.block(n, dim * nout, 1, nout).noalias() +=
              (weight / accumulator(dim)) * values.row(ind);
        }
      }
    }
  }

  return 0;
}

// This implementation (By Dan Foreman-Mackey) for the "exoplanet" library is
// based closely on the implementation from the TheiaSfM project
// (https://github.com/sweeneychris/TheiaSfM) which has the following license:
//
// Copyright (C) 2015 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

// These functions are largely based off the the Ceres solver polynomial
// functions which are not available through the public interface. The license
// is below:
//
// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: moll.markus@arcor.de (Markus Moll)
//         sameeragarwal@google.com (Sameer Agarwal)

#ifndef THEIA_MATH_FIND_POLYNOMIAL_ROOTS_COMPANION_MATRIX_H_
#define THEIA_MATH_FIND_POLYNOMIAL_ROOTS_COMPANION_MATRIX_H_

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace theia {

// All polynomials are assumed to be the form
//
//   sum_{i=0}^N polynomial(i) x^{N-i}.
//
// and are given by a vector of coefficients of size N + 1.

// Use the companion matrix eigenvalues to determine the roots of the
// polynomial.
//
// This function returns true on success, false otherwise.
// Failure indicates that the polynomial is invalid (of size 0) or
// that the eigenvalues of the companion matrix could not be computed.
// On failure, a more detailed message will be written to LOG(ERROR).
// If real is not NULL, the real parts of the roots will be returned in it.
// Likewise, if imaginary is not NULL, imaginary parts will be returned in it.

namespace {

// Balancing function as described by B. N. Parlett and C. Reinsch,
// "Balancing a Matrix for Calculation of Eigenvalues and Eigenvectors".
// In: Numerische Mathematik, Volume 13, Number 4 (1969), 293-304,
// Springer Berlin / Heidelberg. DOI: 10.1007/BF02165404
template <typename T1>
void BalanceCompanionMatrix(Eigen::MatrixBase<T1>& companion_matrix) {
  typedef typename T1::Scalar T;

  Eigen::Matrix<T, T1::RowsAtCompileTime, T1::ColsAtCompileTime>
    companion_matrix_offdiagonal = companion_matrix;
  companion_matrix_offdiagonal.diagonal().setZero();

  const int degree = companion_matrix.rows();

  // gamma <= 1 controls how much a change in the scaling has to
  // lower the 1-norm of the companion matrix to be accepted.
  //
  // gamma = 1 seems to lead to cycles (numerical issues?), so
  // we set it slightly lower.
  const T gamma = 0.9;

  // Greedily scale row/column pairs until there is no change.
  bool scaling_has_changed;
  do {
    scaling_has_changed = false;

    for (int i = 0; i < degree; ++i) {
      const T row_norm = companion_matrix_offdiagonal.row(i).template lpNorm<1>();
      const T col_norm = companion_matrix_offdiagonal.col(i).template lpNorm<1>();

      // Decompose row_norm/col_norm into mantissa * 2^exponent,
      // where 0.5 <= mantissa < 1. Discard mantissa (return value
      // of frexp), as only the exponent is needed.
      int exponent = 0;
      std::frexp(row_norm / col_norm, &exponent);
      exponent /= 2;

      if (exponent != 0) {
        const T scaled_col_norm = std::ldexp(col_norm, exponent);
        const T scaled_row_norm = std::ldexp(row_norm, -exponent);
        if (scaled_col_norm + scaled_row_norm < gamma * (col_norm + row_norm)) {
          // Accept the new scaling. (Multiplication by powers of 2 should not
          // introduce rounding errors (ignoring non-normalized numbers and
          // over- or underflow))
          scaling_has_changed = true;
          companion_matrix_offdiagonal.row(i) *= std::ldexp(1.0, -exponent);
          companion_matrix_offdiagonal.col(i) *= std::ldexp(1.0, exponent);
        }
      }
    }
  } while (scaling_has_changed);

  companion_matrix_offdiagonal.diagonal() = companion_matrix.diagonal();
  companion_matrix = companion_matrix_offdiagonal;
}

template <typename T1, typename T2>
void BuildCompanionMatrix(const Eigen::MatrixBase<T1>& polynomial,
                          Eigen::MatrixBase<T2>& companion_matrix) {
  const int degree = polynomial.size() - 1;
  companion_matrix.resize(degree, degree);
  companion_matrix.setZero();
  companion_matrix.diagonal(-1).setOnes();
  companion_matrix.col(degree - 1) = -polynomial.reverse().head(degree);
}

}  // namespace

// All polynomials are assumed to be the form
//
//   sum_{i=0}^N polynomial(i) x^{N-i}.
//
// and are given by a vector of coefficients of size N + 1.

// Use the companion matrix eigenvalues to determine the roots of the
// polynomial.
//
// This function returns true on success, false otherwise.
// Failure indicates that the polynomial is invalid (of size 0) or
// that the eigenvalues of the companion matrix could not be computed.
// On failure, a more detailed message will be written to LOG(ERROR).
// If real is not NULL, the real parts of the roots will be returned in it.
// Likewise, if imaginary is not NULL, imaginary parts will be returned in it.
template <typename T1, typename T2>
bool FindPolynomialRootsCompanionMatrix(const Eigen::MatrixBase<T1>& polynomial_in,
                                        Eigen::MatrixBase<T2>& real,
                                        Eigen::MatrixBase<T2>& imag) {
  typedef typename T1::Scalar T;
  const int Rows = (T1::RowsAtCompileTime == Eigen::Dynamic) ? Eigen::Dynamic : (T1::RowsAtCompileTime - 1);

  const int degree = polynomial_in.size() - 1;

  // Divide by leading term
  Eigen::Matrix<T, T1::RowsAtCompileTime, 1> polynomial = polynomial_in / polynomial_in(0);

  // Build and balance the companion matrix to the polynomial.
  Eigen::Matrix<T, Rows, Rows> companion_matrix(degree, degree);
  BuildCompanionMatrix(polynomial, companion_matrix);
  BalanceCompanionMatrix(companion_matrix);

  // Find its (complex) eigenvalues.
  Eigen::EigenSolver<Eigen::Matrix<T, Rows, Rows>> solver(companion_matrix, false);
  if (solver.info() != Eigen::Success) {
    return false;
  }

  // Output roots
  real << solver.eigenvalues().real();
  imag << solver.eigenvalues().imag();
  return true;
}


}  // namespace theia

#endif  // THEIA_MATH_FIND_POLYNOMIAL_ROOTS_COMPANION_MATRIX_H_

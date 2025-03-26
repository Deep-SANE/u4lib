#ifndef U4LIB_UTILS_INVERSESOLVER_H
#define U4LIB_UTILS_INVERSESOLVER_H

#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "basics.h"

// Finding the inverse of a matrix
/*
i) if the matrix is dense, we can use the standard method included in Eigen
ii) if the matrix is sparse, LU decomposition is performed and solve AX = I
*/
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
inverse(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& mat)
{

  // Create a matrix to store the inverse
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> inverseMatrix;

  // Sparse matrix solver
  if (sparisity(mat) >= 0.7)
  {

    // Create the identity matrix

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Identity(mat.rows(),
                                                                   mat.cols());
    for (int i = 0; i < mat.rows(); i++)
      for (int j = 0; j < mat.cols(); j++)
        Identity(i, j) = (i == j) ? 1.0 : 0.0;

    // Convert Matrix to SparseMatrix
    Eigen::SparseMatrix<double> spar_mat = mat.sparseView();
    spar_mat.makeCompressed();

    // Create a SparseQR decomposition of the matrix
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
        solver;

    // Compute the QR decomposition of the matrix
    solver.analyzePattern(spar_mat);
    solver.setPivotThreshold(1e-6);
    solver.factorize(spar_mat);

    inverseMatrix = solver.solve(Identity);
  }

  else
  {
    inverseMatrix = mat.inverse();
  }

  return inverseMatrix;
}

#endif
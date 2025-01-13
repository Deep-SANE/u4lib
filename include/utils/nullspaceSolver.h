// protection against multiple header file inclusion
#ifndef NULLSPACESOLVER_H
#define NULLSPACESOLVER_H

#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

#include "basics.h"

// A function to return the standard deviation (error) of solving the nullspace,
// it is calculated by going through all elements of (projection_matrix *
// nullspace) and determine the standard deviation
double error(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& mat,
             Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& nullspace)
{

  if (mat.size() <= 1)
    return 0.0;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> zeros = mat * nullspace;

  double error = 0.0;
  for (int i = 0; i < zeros.rows(); i++)
    for (int j = 0; j < zeros.cols(); j++)
      error += abs(zeros(i, j));
  error /= zeros.size();

  return error;
}

// Solving the null space of a rectangular matrix
/*
The main strategy to solve the null space of a matrix A of size m x n is as
follows i) Apply QR decomposition on the matrix: AP = QR, ii) Matrix R has size
m x n, thus it will be resized to n x n, iii) Solve matrix R by backward
substitution to obtain the null space of A, iv) However, columns of A has been
permuted, the last step is to permute the components of the null space vectors
back by multiplying the permutation matrix P with the null space matrix.
*/
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
null_space(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& mat)
{

  if (mat.size() <= 1)
  {
    Eigen::Matrix<double, 1, 1> one;
    one(0, 0) = 1;
    return one;
  }

  else
  {

    // Create a matrix to store the null space
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> nullSpace;

    // std::cout << sparisity(mat) << std::endl;

    // Sparse matrix solver
    if (sparisity(mat) >= 0.7)
    {

      // Convert Matrix to SparseMatrix
      Eigen::SparseMatrix<double> spar_mat = mat.sparseView();
      spar_mat.makeCompressed();

      // Create a SparseQR decomposition of the projection matrix
      Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
          qr;

      // Compute the qR decomposition of the projection matrix
      qr.setPivotThreshold(1e-6);
      qr.compute(spar_mat);

      // Get the upper triangular matrix and resize it to square
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> R(qr.matrixR());
      R.conservativeResize(mat.cols(), mat.cols());

      // Get the nullity -- dimension of the nullspace
      int nullity = mat.cols() - qr.rank();

      nullSpace = backwardSubstitution(R, nullity);

      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> P(
          qr.colsPermutation());

      nullSpace = P * nullSpace;
    }

    else
    {

      Eigen::ColPivHouseholderQR<
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
          qr;

      // Compute the qR decomposition of the projection matrix
      // qr.setThreshold(1e-16);
      qr.compute(mat);

      // Get the upper triangular matrix and resize it to square
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> R(
          qr.matrixR().template triangularView<Eigen::Upper>());
      R.conservativeResize(mat.cols(), mat.cols());

      // Get the nullity -- dimension of the nullspace
      int nullity = qr.dimensionOfKernel();

      nullSpace = backwardSubstitution(R, nullity);

      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> P(
          qr.colsPermutation());

      nullSpace = P * nullSpace;
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ornorNullSpace =
        gramSchmidt(nullSpace);

    // Sign convention
    for (int i = 0; i < ornorNullSpace.cols(); i++)
    {
      int j;
      for (j = 0; j < ornorNullSpace.rows(); j++)
        if (std::abs(ornorNullSpace(j, i)) > 1e-15)
          break;
      if (ornorNullSpace(j, i) < 0)
        ornorNullSpace.col(i) = -ornorNullSpace.col(i);

      for (j = 0; j < ornorNullSpace.rows(); j++)
        if (abs(ornorNullSpace(j, i)) <= 1e-15)
          ornorNullSpace(j, i) = 0.0;
    }

    return ornorNullSpace;
  }
}

#endif

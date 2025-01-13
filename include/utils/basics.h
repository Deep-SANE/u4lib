// protection against multiple header file inclusion
#ifndef BASICS_H
#define BASICS_H

#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

// Determine sparsity of a matrix
float sparisity(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& mat)
{
  float s = 0;
  for (int i = 0; i < mat.rows(); i++)
  {
    for (int j = 0; j < mat.cols(); j++)
      if (mat(i, j) == 0)
        s++;
  }
  return s / (mat.rows() * mat.cols());
}

// Solving an upper square matrix by backward substitution
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
backwardSubstitution(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& mat,
                     int nullity)
{
  int row_num = mat.cols();
  double epsilon = 1e-10;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> result(row_num,
                                                               nullity);

  for (int i = 0; i < nullity; i++)
  {

    for (int j = row_num - i; j < row_num; j++)
      result(j, i) = 0;
    result(row_num - i - 1, i) = 1.0;

    for (int j = row_num - i - 2; j >= 0; j--)
    {
      double sum = 0.0;

      for (int k = j + 1; k < row_num; k++)
        sum += mat(j, k) * result(k, i);

      if (abs(mat(j, j)) < epsilon)
        result(j, i) = 0.0;

      else
        result(j, i) = -sum / mat(j, j);
    }
  }

  return result;
}

// A function to perform the modified Gram-Schmidt orthonormalization -- a more
// stable version of the traditional algorithm
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
gramSchmidt(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& mat)
{

  int row_num = mat.rows();
  int col_num = mat.cols();

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matMGS(row_num,
                                                               col_num);

  for (int i = 0; i < col_num; i++)
  {
    matMGS.col(i) = mat.col(i);
    for (int j = 0; j < i; j++)
    {
      matMGS.col(i) =
          matMGS.col(i) -
          (matMGS.col(j).transpose() * matMGS.col(i)) * matMGS.col(j);
    }
    matMGS.col(i) = matMGS.col(i) / matMGS.col(i).norm();
  }

  return matMGS;

  /* The traditional version
  matMGS = mat;

  for(int j=0; j<matMGS.cols(); j++){
              //Orthogonalization
              for(int k=0; k<j; k++){
                      double projection = 0.0;
                      for(int i=0; i<matMGS.rows(); i++){
                              projection += matMGS(i,j)*matMGS(i,k);
                      }
                      for(int i=0; i<matMGS.rows(); i++){
                              matMGS(i,j) -= projection*matMGS(i,k);
                      }
              }
              matMGS.col(j) /= matMGS.col(j).norm();
      }

      return matMGS;
      */
}

#endif
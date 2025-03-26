#ifndef U4LIB_U4SU2SU2_to_U4U3U2U1_H
#define U4LIB_U4SU2SU2_to_U4U3U2U1_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "U4basis.h"
#include "utils/nullspaceSolver.h"

namespace u4
{
  // construction of the canonical basis: the canonical basis vectors will be
  // stored in an array

  std::vector<int> expand_to_gelfand(int MSx2, int MTx2, std::vector<u4::gelfand>& gArr)
  {
    std::vector<int> indices;

    for (int i = 0; i < gArr.size(); i++)
      if (MSx2 == (gArr[i].e11() + gArr[i].e22() - gArr[i].e33() - gArr[i].e44()) &&
          MTx2 == (gArr[i].e11() - gArr[i].e22() + gArr[i].e33() - gArr[i].e44()))
        indices.push_back(i);

    return indices;
  }

  /*------------------------------------------------------
  Finding the transformation by solving a system of linear equations resulted
  from acting S+ = E13 + E24 and T+ = E12 + E34
  */
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  STprojection(u4::physicalvector& physvec, std::vector<u4::gelfand>& gArr,
               std::vector<int>& indices)
  {

    std::vector<int> new_indices;

    for (int i = 0; i < gArr.size(); i++)
    {
      for (int j = 0; j < indices.size(); j++)
      {
        // action of E24
        if (u4::E24_13_12(gArr[indices[j]], gArr[i]))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E24_13_22(gArr[indices[j]], gArr[i]))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E24_23_12(gArr[indices[j]], gArr[i]))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E24_23_22(gArr[indices[j]], gArr[i]))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E24_33_12(gArr[indices[j]], gArr[i]))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E24_33_22(gArr[indices[j]], gArr[i]))
        {
          new_indices.push_back(i);
          break;
        }

        // action of E13
        else if (u4::E13_12(gArr[indices[j]], gArr[i]))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E13_22(gArr[indices[j]], gArr[i]))
        {
          new_indices.push_back(i);
          break;
        }

        // action of E34
        else if (u4::E34_13(gArr[indices[j]], gArr[i]))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E34_23(gArr[indices[j]], gArr[i]))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E34_33(gArr[indices[j]], gArr[i]))
        {
          new_indices.push_back(i);
          break;
        }

        // action of E12
        else if (u4::E12(gArr[indices[j]], gArr[i]))
        {
          new_indices.push_back(i);
          break;
        }
      }
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> projector(
        new_indices.size(), indices.size());

    for (int i = 0; i < projector.rows(); i++)
    {
      for (int j = 0; j < projector.cols(); j++)
      {
        // action of E24
        if (u4::E24_13_12(gArr[indices[j]], gArr[new_indices[i]]))
          projector(i, j) = gArr[indices[j]].e24_13_12();
        else if (u4::E24_13_22(gArr[indices[j]], gArr[new_indices[i]]))
          projector(i, j) = gArr[indices[j]].e24_13_22();
        else if (u4::E24_23_12(gArr[indices[j]], gArr[new_indices[i]]))
          projector(i, j) = gArr[indices[j]].e24_23_12();
        else if (u4::E24_23_22(gArr[indices[j]], gArr[new_indices[i]]))
          projector(i, j) = gArr[indices[j]].e24_23_22();
        else if (u4::E24_33_12(gArr[indices[j]], gArr[new_indices[i]]))
          projector(i, j) = gArr[indices[j]].e24_33_12();
        else if (u4::E24_33_22(gArr[indices[j]], gArr[new_indices[i]]))
          projector(i, j) = gArr[indices[j]].e24_33_22();

        // action of E13
        else if (u4::E13_12(gArr[indices[j]], gArr[new_indices[i]]))
          projector(i, j) = gArr[indices[j]].e13_12();
        else if (u4::E13_22(gArr[indices[j]], gArr[new_indices[i]]))
          projector(i, j) = gArr[indices[j]].e13_22();

        // action of E34
        else if (u4::E34_13(gArr[indices[j]], gArr[new_indices[i]]))
          projector(i, j) = gArr[indices[j]].e34_13();
        else if (u4::E34_23(gArr[indices[j]], gArr[new_indices[i]]))
          projector(i, j) = gArr[indices[j]].e34_23();
        else if (u4::E34_33(gArr[indices[j]], gArr[new_indices[i]]))
          projector(i, j) = gArr[indices[j]].e34_33();

        // action of E12
        else if (u4::E12(gArr[indices[j]], gArr[new_indices[i]]))
          projector(i, j) = gArr[indices[j]].e12();

        else
          projector(i, j) = 0.0;
      }
    }

    return projector;
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  lowerMS(int zeta_max, int Sx2, int MSx2, std::vector<u4::gelfand>& gArr,
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& transformation,
          std::vector<int>& canoBasis, std::vector<int>& canoBasis_new)
  {
    // normalize coefficient
    double n = std::sqrt(4.0 / ((Sx2 - MSx2) * (Sx2 + MSx2 + 2.0)));
    // input MSx2 is for the new state

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> transformation_new(
        canoBasis_new.size(), zeta_max);
    transformation_new.setZero();

    // basis states are obtained by acting with two generators E31 and E42
    for (int i = 0; i < canoBasis_new.size(); i++)
    {
      for (int j = 0; j < canoBasis.size(); j++)
      {
        if (u4::E13_12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                n * transformation(j, z) * gArr[canoBasis[j]].e31_12();
        else if (u4::E13_22(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                n * transformation(j, z) * gArr[canoBasis[j]].e31_22();

        else if (u4::E24_13_12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                n * transformation(j, z) * gArr[canoBasis[j]].e42_13_12();
        else if (u4::E24_13_22(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                n * transformation(j, z) * gArr[canoBasis[j]].e42_13_22();
        else if (u4::E24_23_12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                n * transformation(j, z) * gArr[canoBasis[j]].e42_23_12();
        else if (u4::E24_23_22(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                n * transformation(j, z) * gArr[canoBasis[j]].e42_23_22();
        else if (u4::E24_33_12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                n * transformation(j, z) * gArr[canoBasis[j]].e42_33_12();
        else if (u4::E24_33_22(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                n * transformation(j, z) * gArr[canoBasis[j]].e42_33_22();
      }

      for (int z = 0; z < zeta_max; z++)
        if (std::fabs(transformation_new(i, z)) < 1e-12)
          transformation_new(i, z) = 0.0;
    }
    return transformation_new;
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  lowerMT(int zeta_max, int Tx2, int MTx2, std::vector<u4::gelfand>& gArr,
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& transformation,
          std::vector<int>& canoBasis, std::vector<int>& canoBasis_new)
  {
    // normalize coefficient
    double n =  std::sqrt(4.0 / ((Tx2 - MTx2) * (Tx2 + MTx2 + 2.0)));
    // input MSx2 is for the new state

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> transformation_new(
        canoBasis_new.size(), zeta_max);
    transformation_new.setZero();

    // basis states are obtained by acting with two generators E21 and E43
    for (int i = 0; i < canoBasis_new.size(); i++)
    {
      for (int j = 0; j < canoBasis.size(); j++)
      {
        if (u4::E12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                n * transformation(j, z) * gArr[canoBasis[j]].e21();

        else if (u4::E34_13(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                n * transformation(j, z) * gArr[canoBasis[j]].e43_13();
        else if (u4::E34_23(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                n * transformation(j, z) * gArr[canoBasis[j]].e43_23();
        else if (u4::E34_33(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                n * transformation(j, z) * gArr[canoBasis[j]].e43_33();
      }

      for (int z = 0; z < zeta_max; z++)
        if (std::fabs(transformation_new(i, z)) < 1e-12)
          transformation_new(i, z) = 0.0;
    }
    return transformation_new;
  }

  // get the transformation for MS <= S and MT <= T
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  phys_to_cano(u4::physicalvector& physvec, int MSx2, int MTx2,
               std::vector<int>& canoBasis,
               std::vector<u4::gelfand>& gArr)
  {
    canoBasis = u4::expand_to_gelfand(physvec.SS, physvec.TT, gArr);

    // build the projection matrix
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> STproj =
        u4::STprojection(physvec, gArr, canoBasis);
    // and solve its null space
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> transformation =
        null_space(STproj);

    // ensure the nullspace is solved correctly
    assert(physvec.zeta == transformation.cols());

    // lower MS 
    for (int msx2 = physvec.SS - 2; msx2 >= MSx2; msx2 -= 2)
    {
      //store new expansion
      std::vector<int> canoBasis_new = u4::expand_to_gelfand(msx2, physvec.TT, gArr);

      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> transformation_new =
          u4::lowerMS(physvec.zeta, physvec.SS, msx2, gArr, transformation,
          canoBasis, canoBasis_new);

      transformation = transformation_new;
      canoBasis = canoBasis_new;
    }

    // lower MT 
    for (int mtx2 = physvec.TT - 2; mtx2 >= MTx2; mtx2 -= 2)
    {
      //store new expansion
      std::vector<int> canoBasis_new = u4::expand_to_gelfand(MSx2, mtx2, gArr);

      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> transformation_new =
          u4::lowerMT(physvec.zeta, physvec.TT, mtx2, gArr, transformation,
          canoBasis, canoBasis_new);

      transformation = transformation_new;
      canoBasis = canoBasis_new;
    }

    return transformation;
  }

} // namespace u4

#endif

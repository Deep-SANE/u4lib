#ifndef U4LIB_U4SU2SU2_TO_U4U3U2U1_H
#define U4LIB_U4SU2SU2_TO_U4U3U2U1_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#ifdef USE_SLEPC
#include <Eigen/Sparse>
#endif

#include "U4basis.h"

#include <nullspace_solver_qr.h>
#ifdef USE_SLEPC
#include <nullspace_solver_slepc.h>
#endif
 
namespace u4
{
  // construction of the canonical basis: the canonical basis vectors will be
  // stored in an array

  inline std::vector<int> 
  expand_to_gelfand(int MSx2, int MTx2, const std::vector<u4::gelfand>& gArr)
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
  inline auto STprojection(u4::physicalvector& physvec, 
    const std::vector<u4::gelfand>& gArr, std::vector<int>& indices)
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

    int row_num = new_indices.size(), col_num = indices.size();

#ifdef USE_SLEPC
    std::vector<Eigen::Triplet<double>> ija;
#else
    Eigen::MatrixXd projector(row_num, col_num);
#endif

    for (int i = 0; i < row_num; i++)
    {
      for (int j = 0; j < col_num; j++)
      {
        // action of E24
        if (u4::E24_13_12(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef USE_SLEPC
          ija.emplace_back(i, j, gArr[indices[j]].e24_13_12());
#else
          projector(i, j) = gArr[indices[j]].e24_13_12();
#endif
        else if (u4::E24_13_22(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef USE_SLEPC
          ija.emplace_back(i, j, gArr[indices[j]].e24_13_22());
#else
          projector(i, j) = gArr[indices[j]].e24_13_22();
#endif
        else if (u4::E24_23_12(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef USE_SLEPC
          ija.emplace_back(i, j, gArr[indices[j]].e24_23_12());
#else
          projector(i, j) = gArr[indices[j]].e24_23_12();
#endif
        else if (u4::E24_23_22(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef USE_SLEPC
          ija.emplace_back(i, j, gArr[indices[j]].e24_23_22());
#else
          projector(i, j) = gArr[indices[j]].e24_23_22();
#endif
        else if (u4::E24_33_12(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef USE_SLEPC
          ija.emplace_back(i, j, gArr[indices[j]].e24_33_12());
#else
          projector(i, j) = gArr[indices[j]].e24_33_12();
#endif
        else if (u4::E24_33_22(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef USE_SLEPC
          ija.emplace_back(i, j, gArr[indices[j]].e24_33_22());
#else
          projector(i, j) = gArr[indices[j]].e24_33_22();
#endif

        // action of E13
        else if (u4::E13_12(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef USE_SLEPC
          ija.emplace_back(i, j, gArr[indices[j]].e13_12());
#else
          projector(i, j) = gArr[indices[j]].e13_12();
#endif
        else if (u4::E13_22(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef USE_SLEPC
          ija.emplace_back(i, j, gArr[indices[j]].e13_22());
#else
          projector(i, j) = gArr[indices[j]].e13_22();
#endif

        // action of E34
        else if (u4::E34_13(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef USE_SLEPC
          ija.emplace_back(i, j, gArr[indices[j]].e34_13());
#else
          projector(i, j) = gArr[indices[j]].e34_13();
#endif
        else if (u4::E34_23(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef USE_SLEPC
          ija.emplace_back(i, j, gArr[indices[j]].e34_23());
#else
          projector(i, j) = gArr[indices[j]].e34_23();
#endif
        else if (u4::E34_33(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef USE_SLEPC
          ija.emplace_back(i, j, gArr[indices[j]].e34_33());
#else
          projector(i, j) = gArr[indices[j]].e34_33();
#endif

        // action of E12
        else if (u4::E12(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef USE_SLEPC
          ija.emplace_back(i, j, gArr[indices[j]].e12());
#else
          projector(i, j) = gArr[indices[j]].e12();
#endif

        else
#ifdef USE_SLEPC
          ;
#else
          projector(i, j) = 0.0;
#endif
      }
    }

#ifdef USE_SLEPC
    Eigen::SparseMatrix<double> projector(row_num, col_num);
    projector.setFromTriplets(ija.begin(), ija.end());
    projector.makeCompressed();
#endif

    return projector;
  }

  inline Eigen::MatrixXd lowerMS(int zeta_max, int Sx2, int MSx2, 
    const std::vector<u4::gelfand>& gArr,
    const Eigen::MatrixXd& transformation,
    const std::vector<int>& canoBasis, 
    const std::vector<int>& canoBasis_new)
  {
    // normalize coefficient
    double n = std::sqrt(4.0 / ((Sx2 - MSx2) * (Sx2 + MSx2 + 2.0)));
    // input MSx2 is for the new state

    Eigen::MatrixXd transformation_new(canoBasis_new.size(), zeta_max);
    transformation_new.setZero();

    // basis states are obtained by acting with two generators E31 and E42
    for (int i = 0; i < canoBasis_new.size(); i++)
    {
      for (int j = 0; j < canoBasis.size(); j++)
      {
        if (u4::E13_12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                transformation(j, z) * gArr[canoBasis[j]].e31_12();
        else if (u4::E13_22(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                transformation(j, z) * gArr[canoBasis[j]].e31_22();

        else if (u4::E24_13_12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                transformation(j, z) * gArr[canoBasis[j]].e42_13_12();
        else if (u4::E24_13_22(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                transformation(j, z) * gArr[canoBasis[j]].e42_13_22();
        else if (u4::E24_23_12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                transformation(j, z) * gArr[canoBasis[j]].e42_23_12();
        else if (u4::E24_23_22(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                transformation(j, z) * gArr[canoBasis[j]].e42_23_22();
        else if (u4::E24_33_12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                transformation(j, z) * gArr[canoBasis[j]].e42_33_12();
        else if (u4::E24_33_22(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                transformation(j, z) * gArr[canoBasis[j]].e42_33_22();
      }

      for (int z = 0; z < zeta_max; z++)
      {
        transformation_new(i, z) *= n;

        if (std::fabs(transformation_new(i, z)) < 1e-12)
          transformation_new(i, z) = 0.0;
      }
    }
    return transformation_new;
  }

  inline Eigen::MatrixXd lowerMT(int zeta_max, int Tx2, int MTx2, 
    const std::vector<u4::gelfand>& gArr,
    const Eigen::MatrixXd& transformation,
    const std::vector<int>& canoBasis, 
    const std::vector<int>& canoBasis_new)
  {
    // normalize coefficient
    double n =  std::sqrt(4.0 / ((Tx2 - MTx2) * (Tx2 + MTx2 + 2.0)));
    // input MSx2 is for the new state

    Eigen::MatrixXd transformation_new(canoBasis_new.size(), zeta_max);
    transformation_new.setZero();

    // basis states are obtained by acting with two generators E21 and E43
    for (int i = 0; i < canoBasis_new.size(); i++)
    {
      for (int j = 0; j < canoBasis.size(); j++)
      {
        if (u4::E12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                transformation(j, z) * gArr[canoBasis[j]].e21();

        else if (u4::E34_13(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                transformation(j, z) * gArr[canoBasis[j]].e43_13();
        else if (u4::E34_23(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                transformation(j, z) * gArr[canoBasis[j]].e43_23();
        else if (u4::E34_33(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          for (int z = 0; z < zeta_max; z++)
            transformation_new(i, z) +=
                transformation(j, z) * gArr[canoBasis[j]].e43_33();
      }

      for (int z = 0; z < zeta_max; z++)
      {
        transformation_new(i, z) *= n;

        if (std::fabs(transformation_new(i, z)) < 1e-12)
          transformation_new(i, z) = 0.0;
      }
    }
    return transformation_new;
  }

  // get the transformation for MS <= S and MT <= T
  inline Eigen::MatrixXd phys_to_cano(u4::physicalvector& physvec, int MSx2, int MTx2, 
    std::vector<int>& canoBasis, const std::vector<u4::gelfand>& gArr)
  {
    canoBasis = u4::expand_to_gelfand(physvec.SS, physvec.TT, gArr);

    // build the projection matrix
    auto STproj = u4::STprojection(physvec, gArr, canoBasis);
    //std::cout << STproj << std::endl;
    // and solve its null space
#ifdef USE_SLEPC
    Eigen::MatrixXd transformation =
        qcx_utils::null_space_slepc(STproj, physvec.zeta);
#else
    Eigen::MatrixXd transformation = qcx_utils::null_space(STproj);
#endif

    // ensure the nullspace is solved correctly
    assert(physvec.zeta == transformation.cols());

    // lower MS 
    for (int msx2 = physvec.SS - 2; msx2 >= MSx2; msx2 -= 2)
    {
      //store new expansion
      std::vector<int> canoBasis_new = u4::expand_to_gelfand(msx2, physvec.TT, gArr);

      Eigen::MatrixXd transformation_new = u4::lowerMS(physvec.zeta, physvec.SS,  
                          msx2, gArr, transformation, canoBasis, canoBasis_new);

      transformation = transformation_new;
      canoBasis = canoBasis_new;
    }

    // lower MT 
    for (int mtx2 = physvec.TT - 2; mtx2 >= MTx2; mtx2 -= 2)
    {
      //store new expansion
      std::vector<int> canoBasis_new = u4::expand_to_gelfand(MSx2, mtx2, gArr);

      Eigen::MatrixXd transformation_new = u4::lowerMT(physvec.zeta, physvec.TT,
                          mtx2, gArr, transformation, canoBasis, canoBasis_new);

      transformation = transformation_new;
      canoBasis = canoBasis_new;
    }

    return transformation;
  }

} // namespace u4

#endif

#ifndef U4LIB_U4SU2SU2_TO_U4U3U2U1_H
#define U4LIB_U4SU2SU2_TO_U4U3U2U1_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>   // Eigen::Triplet — single-pass STprojection build (both return paths)

#include "U4basis.h"

#include <nullspace_solver_qr.h>
#if defined(NULLSPACE_SOLVER_SLEPC)
#include <nullspace_solver_slepc.h>
#elif defined(NULLSPACE_SOLVER_SNS)
#include <nullspace_solver_sns.h>
#endif
 
namespace u4
{
  // construction of the canonical basis
  inline std::vector<int> 
  expand_to_gelfand(int MMS, int MMT, const std::vector<u4::gelfand>& gArr)
  {
    std::vector<int> indices;

    for (int i = 0; i < gArr.size(); i++)
      if (MMS == (gArr[i].e11() + gArr[i].e22() - gArr[i].e33() - gArr[i].e44()) &&
          MMT == (gArr[i].e11() - gArr[i].e22() + gArr[i].e33() - gArr[i].e44()))
        indices.push_back(i);

    return indices;
  }

  // finding the transformation by solving a system of linear equations 
  // resulted from acting S+ = E13 + E24 and T+ = E12 + E34
  inline auto STprojection(const std::vector<u4::gelfand>& gArr,
                           const std::vector<int>& indices)
  {
    /*
    std::vector<int> new_indices;

    for (int i = 0; i < gArr.size(); i++)
    {
      const auto& g_new = gArr[i];

      for (int j = 0; j < indices.size(); j++)
      {
        const auto& g = gArr[indices[j]];

        // action of E24
        if (u4::E24_13_12(g, g_new))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E24_13_22(g, g_new))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E24_23_12(g, g_new))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E24_23_22(g, g_new))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E24_33_12(g, g_new))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E24_33_22(g, g_new))
        {
          new_indices.push_back(i);
          break;
        }

        // action of E13
        else if (u4::E13_12(g, g_new))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E13_22(g, g_new))
        {
          new_indices.push_back(i);
          break;
        }

        // action of E34
        else if (u4::E34_13(g, g_new))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E34_23(g, g_new))
        {
          new_indices.push_back(i);
          break;
        }
        else if (u4::E34_33(g, g_new))
        {
          new_indices.push_back(i);
          break;
        }

        // action of E12
        else if (u4::E12(g, g_new))
        {
          new_indices.push_back(i);
          break;
        }
      }
    }

    int row_num = new_indices.size(), col_num = indices.size();

#ifdef NULLSPACE_SPARSE
    std::vector<Eigen::Triplet<double>> ija;
#else
    Eigen::MatrixXd projector(row_num, col_num);
#endif

    for (int i = 0; i < row_num; i++)
    {
      const auto& g_new = gArr[new_indices[i]];

      for (int j = 0; j < col_num; j++)
      {
        const auto& g = gArr[indices[j]];

        // action of E24
        if (u4::E24_13_12(g, g_new))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, g.e24_13_12());
#else
          projector(i, j) = g.e24_13_12();
#endif
        else if (u4::E24_13_22(g, g_new))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, g.e24_13_22());
#else
          projector(i, j) = g.e24_13_22();
#endif
        else if (u4::E24_23_12(g, g_new))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, g.e24_23_12());
#else
          projector(i, j) = g.e24_23_12();
#endif
        else if (u4::E24_23_22(g, g_new))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, g.e24_23_22());
#else
          projector(i, j) = g.e24_23_22();
#endif
        else if (u4::E24_33_12(g, g_new))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, g.e24_33_12());
#else
          projector(i, j) = g.e24_33_12();
#endif
        else if (u4::E24_33_22(g, g_new))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, g.e24_33_22());
#else
          projector(i, j) = g.e24_33_22();
#endif

        // action of E13
        else if (u4::E13_12(g, g_new))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, g.e13_12());
#else
          projector(i, j) = g.e13_12();
#endif
        else if (u4::E13_22(g, g_new))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, g.e13_22());
#else
          projector(i, j) = g.e13_22();
#endif

        // action of E34
        else if (u4::E34_13(g, g_new))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, g.e34_13());
#else
          projector(i, j) = g.e34_13();
#endif
        else if (u4::E34_23(g, g_new))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, g.e34_23());
#else
          projector(i, j) = g.e34_23();
#endif
        else if (u4::E34_33(g, g_new))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, g.e34_33());
#else
          projector(i, j) = g.e34_33();
#endif

        // action of E12
        else if (u4::E12(g, g_new))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, g.e12());
#else
          projector(i, j) = g.e12();
#endif

        else
#ifdef NULLSPACE_SPARSE
          ;
#else
          projector(i, j) = 0.0;
#endif
      }
    }

#ifdef NULLSPACE_SPARSE
    Eigen::SparseMatrix<double> projector(row_num, col_num);
    projector.setFromTriplets(ija.begin(), ija.end());
    projector.makeCompressed();
#endif
    */

    // Single pass: evaluate the raising-operator predicate chain once per
    // (candidate row i, column j); on the first match emit a triplet and mark
    // the row used. Row indices are assigned lazily so only rows with >=1 entry
    // get a slot -- reproducing new_indices' order without a separate
    // identification pass or any re-evaluation of the predicates.
    std::vector<Eigen::Triplet<double>> ija;
    int row_num = 0, col_num = indices.size();

    for (int i = 0; i < gArr.size(); i++)
    {
      const auto& g_new = gArr[i];
      bool row_has_entry = false;

      for (int j = 0; j < col_num; j++)
      {
        const auto& g = gArr[indices[j]];
        double val;

        // action of E24
        if      (u4::E24_13_12(g, g_new)) val = g.e24_13_12();
        else if (u4::E24_13_22(g, g_new)) val = g.e24_13_22();
        else if (u4::E24_23_12(g, g_new)) val = g.e24_23_12();
        else if (u4::E24_23_22(g, g_new)) val = g.e24_23_22();
        else if (u4::E24_33_12(g, g_new)) val = g.e24_33_12();
        else if (u4::E24_33_22(g, g_new)) val = g.e24_33_22();
        // action of E13
        else if (u4::E13_12(g, g_new))    val = g.e13_12();
        else if (u4::E13_22(g, g_new))    val = g.e13_22();
        // action of E34
        else if (u4::E34_13(g, g_new))    val = g.e34_13();
        else if (u4::E34_23(g, g_new))    val = g.e34_23();
        else if (u4::E34_33(g, g_new))    val = g.e34_33();
        // action of E12
        else if (u4::E12(g, g_new))       val = g.e12();
        else continue;                    // no entry at (i, j)

        ija.emplace_back(row_num, j, val);
        row_has_entry = true;
      }

      if (row_has_entry) row_num++;
    }

#ifdef NULLSPACE_SPARSE
    Eigen::SparseMatrix<double> projector(row_num, col_num);
    projector.setFromTriplets(ija.begin(), ija.end());
    projector.makeCompressed();
#else
    Eigen::MatrixXd projector = Eigen::MatrixXd::Zero(row_num, col_num);
    for (const auto& t : ija) projector(t.row(), t.col()) = t.value();
#endif

    return projector;
  }

  // action of the lowering operator S- = E31 + E42
  inline auto lowerMS(
    const u4::physicalvector& p, 
    const int& MMS, // MMS is for the new state
    const int& MMT, // MMT is for the new state, unchanged
    const std::vector<u4::gelfand>& gArr,
    const std::vector<int>& canoBasis,
    const Eigen::MatrixXd& transformation)
  {
    // normalize coefficient
    double n = std::sqrt(4.0 / ((p.SS() - MMS) * (p.SS() + MMS + 2.0)));

    // new expansion
    std::vector<int> canoBasis_new = u4::expand_to_gelfand(MMS, MMT, gArr);

    Eigen::MatrixXd transformation_new(canoBasis_new.size(), p.zeta());
    transformation_new.setZero();

    // basis states are obtained by acting with two generators E31 and E42
    for (int i = 0; i < canoBasis_new.size(); i++)
    {
      const auto& g_new = gArr[canoBasis_new[i]];

      for (int j = 0; j < canoBasis.size(); j++)
      {
        const auto& g = gArr[canoBasis[j]];

        if (u4::E13_12(g_new, g))
          transformation_new.row(i) += transformation.row(j) * g.e31_12();
        else if (u4::E13_22(g_new, g))
          transformation_new.row(i) +=
                transformation.row(j) * g.e31_22();

        else if (u4::E24_13_12(g_new, g))
          transformation_new.row(i) += transformation.row(j) * g.e42_13_12();
        else if (u4::E24_13_22(g_new, g))
          transformation_new.row(i) += transformation.row(j) * g.e42_13_22();
        else if (u4::E24_23_12(g_new, g))
          transformation_new.row(i) += transformation.row(j) * g.e42_23_12();
        else if (u4::E24_23_22(g_new, g))
          transformation_new.row(i) += transformation.row(j) * g.e42_23_22();
        else if (u4::E24_33_12(g_new, g))
          transformation_new.row(i) += transformation.row(j) * g.e42_33_12();
        else if (u4::E24_33_22(g_new, g))
          transformation_new.row(i) += transformation.row(j) * g.e42_33_22();
      }

      transformation_new.row(i) *= n;

      for (auto& x : transformation_new.row(i))
        if (std::fabs(x) < 1e-12) x = 0.0;
    }
    return std::make_pair(canoBasis_new, transformation_new);
  }

  // action of the raising operator S+ = E13 + E24
  inline auto raiseMS(
    const u4::physicalvector& p, 
    const int& MMS, // MMS is for the new state
    const int& MMT, // MMT is for the new state, unchanged
    const std::vector<u4::gelfand>& gArr,
    const std::vector<int>& canoBasis,
    const Eigen::MatrixXd& transformation)
  {
    // normalize coefficient
    double n = std::sqrt(4.0 / ((p.SS() + MMS) * (p.SS() - MMS + 2.0)));

    // new expansion
    std::vector<int> canoBasis_new = u4::expand_to_gelfand(MMS, MMT, gArr);

    Eigen::MatrixXd transformation_new(canoBasis_new.size(), p.zeta());
    transformation_new.setZero();

    // basis states are obtained by acting with two generators E31 and E42
    for (int i = 0; i < canoBasis_new.size(); i++)
    {
      const auto& g_new = gArr[canoBasis_new[i]];

      for (int j = 0; j < canoBasis.size(); j++)
      {
        const auto& g = gArr[canoBasis[j]];

        if (u4::E13_12(g, g_new))
          transformation_new.row(i) += transformation.row(j) * g.e13_12();
        else if (u4::E13_22(g, g_new))
          transformation_new.row(i) += transformation.row(j) * g.e13_22();

        else if (u4::E24_13_12(g, g_new))
          transformation_new.row(i) += transformation.row(j) * g.e24_13_12();
        else if (u4::E24_13_22(g, g_new))
          transformation_new.row(i) += transformation.row(j) * g.e24_13_22();
        else if (u4::E24_23_12(g, g_new))
          transformation_new.row(i) += transformation.row(j) * g.e24_23_12();
        else if (u4::E24_23_22(g, g_new))
          transformation_new.row(i) += transformation.row(j) * g.e24_23_22();
        else if (u4::E24_33_12(g, g_new))
          transformation_new.row(i) += transformation.row(j) * g.e24_33_12();
        else if (u4::E24_33_22(g, g_new))
          transformation_new.row(i) += transformation.row(j) * g.e24_33_22();
      }

      transformation_new.row(i) *= n;

      for (auto& x : transformation_new.row(i))
        if (std::fabs(x) < 1e-12) x = 0.0;
    }
    return std::make_pair(canoBasis_new, transformation_new);
  }

  // action of the lowering operator T- = E21 + E43
  inline auto lowerMT(
    const u4::physicalvector& p, 
    const int& MMS, // MMS is for the new state, unchanged
    const int& MMT, // MMT is for the new state
    const std::vector<u4::gelfand>& gArr,
    const std::vector<int>& canoBasis,
    const Eigen::MatrixXd& transformation)
  {
    // normalize coefficient
    double n =  std::sqrt(4.0 / ((p.TT() - MMT) * (p.TT() + MMT + 2.0)));

    // new expansion
    std::vector<int> canoBasis_new = u4::expand_to_gelfand(MMS, MMT, gArr);

    Eigen::MatrixXd transformation_new(canoBasis_new.size(), p.zeta());
    transformation_new.setZero();

    // basis states are obtained by acting with two generators E21 and E43
    for (int i = 0; i < canoBasis_new.size(); i++)
    {
      const auto& g_new = gArr[canoBasis_new[i]];

      for (int j = 0; j < canoBasis.size(); j++)
      {
        const auto& g = gArr[canoBasis[j]];

        if (u4::E12(g_new, g))
          transformation_new.row(i) += transformation.row(j) * g.e21();

        else if (u4::E34_13(g_new, g))
          transformation_new.row(i) += transformation.row(j) * g.e43_13();
        else if (u4::E34_23(g_new, g))
          transformation_new.row(i) += transformation.row(j) * g.e43_23();
        else if (u4::E34_33(g_new, g))
          transformation_new.row(i) += transformation.row(j) * g.e43_33();
      }

      transformation_new.row(i) *= n;

      for (auto& x : transformation_new.row(i))
        if (std::fabs(x) < 1e-12) x = 0.0;
    }
    return std::make_pair(canoBasis_new, transformation_new);
  }

  // action of the raising operator T+ = E12 + E34
  inline auto raiseMT(
    const u4::physicalvector& p, 
    const int& MMS, // MMS is for the new state, unchanged
    const int& MMT, // MMT is for the new state
    const std::vector<u4::gelfand>& gArr,
    const std::vector<int>& canoBasis,
    const Eigen::MatrixXd& transformation)
  {
    // normalize coefficient
    double n =  std::sqrt(4.0 / ((p.TT() + MMT) * (p.TT() - MMT + 2.0)));

    // new expansion
    std::vector<int> canoBasis_new = u4::expand_to_gelfand(MMS, MMT, gArr);

    Eigen::MatrixXd transformation_new(canoBasis_new.size(), p.zeta());
    transformation_new.setZero();

    // basis states are obtained by acting with two generators E21 and E43
    for (int i = 0; i < canoBasis_new.size(); i++)
    {
      const auto& g_new = gArr[canoBasis_new[i]];

      for (int j = 0; j < canoBasis.size(); j++)
      {
        const auto& g = gArr[canoBasis[j]];

        if (u4::E12(g, g_new))
          transformation_new.row(i) += transformation.row(j) * g.e12();

        else if (u4::E34_13(g, g_new))
          transformation_new.row(i) += transformation.row(j) * g.e34_13();
        else if (u4::E34_23(g, g_new))
          transformation_new.row(i) += transformation.row(j) * g.e34_23();
        else if (u4::E34_33(g, g_new))
          transformation_new.row(i) += transformation.row(j) * g.e34_33();
      }

      transformation_new.row(i) *= n;

      for (auto& x : transformation_new.row(i))
        if (std::fabs(x) < 1e-12) x = 0.0;
    }
    return std::make_pair(canoBasis_new, transformation_new);
  }

  // get the transformation for MS <= S and MT <= T
  inline std::pair<std::vector<int>, Eigen::MatrixXd> 
    phys_to_cano(const u4::physicalvector& p, int MMS, int MMT, 
                 const std::vector<u4::gelfand>& gArr)
  {
    if (p.P1() >= p.P3())
    {
      std::vector<int> canoBasis = u4::expand_to_gelfand(p.SS(), p.TT(), gArr);

      // build the projection matrix
      auto STproj = u4::STprojection(gArr, canoBasis);
      //std::cout << STproj << std::endl;
      // and solve its null space
#if defined(NULLSPACE_SOLVER_SNS)
      Eigen::MatrixXd transformation = qcx_utils::null_space_sns(STproj, p.zeta());
#elif defined(NULLSPACE_SOLVER_SLEPC)
      Eigen::MatrixXd transformation = qcx_utils::null_space_slepc(STproj, p.zeta());
#else
      Eigen::MatrixXd transformation = qcx_utils::null_space(STproj);
#endif

      // ensure the nullspace is solved correctly
      assert(p.zeta() == transformation.cols());

      // lower MS 
      for (int mms = p.SS() - 2; mms >= MMS; mms -= 2)
      {
        auto [canoBasis_new, transformation_new] = 
          u4::lowerMS(p, mms, p.TT(), gArr, canoBasis, transformation);

        transformation = transformation_new;
        canoBasis = canoBasis_new;
      }

      // lower MT 
      for (int mmt = p.TT() - 2; mmt >= MMT; mmt -= 2)
      {
        auto [canoBasis_new, transformation_new] = 
          u4::lowerMT(p, MMS, mmt, gArr, canoBasis, transformation);

        transformation = transformation_new;
        canoBasis = canoBasis_new;
      }

      return std::make_pair(canoBasis, transformation);
    } 
    else 
    {
      // If P1 < P3, we construct the conjugate state and 
      // then conjugate to get the desired state

      u4::irrep irr_conj = p.irr().conjugate();
      u4::physicalvector p_conj(irr_conj, p.SS(), p.TT());
      auto gArr_conj = u4::gelfandBasis(irr_conj);
      double p_phase = 
        ((2 * (p.P1() + p.P2() + p.P3()) - p.SS() - p.TT() - MMS - MMT) % 4 == 0) 
        ? 1.0 : -1.0;

      auto [canoBasis_conj, transformation_conj] = 
        u4::phys_to_cano(p_conj, -MMS, -MMT, gArr_conj);

      auto canoBasis = u4::expand_to_gelfand(MMS, MMT, gArr);
      Eigen::MatrixXd transformation(canoBasis.size(), p.zeta());

      for (size_t i = 0; i < canoBasis.size(); i++)
      {
        const auto& g = gArr[canoBasis[i]];

        for (size_t j = 0; j < canoBasis_conj.size(); j++)
        {
          const auto& g_conj = gArr_conj[canoBasis_conj[j]];

          if (g.conjugate() == g_conj)
          {
            auto g_phase = g_conj.conjugate_phase();
            transformation.row(i) = p_phase * g_phase 
                                  * transformation_conj.row(j);
            break;
          }
        }
      }

      return std::make_pair(canoBasis, transformation);
    }    
  }

  // get the transformation for MS <= S and MT <= T
  inline auto phys_to_cano_allM(const u4::physicalvector& p,
                                const std::vector<u4::gelfand>& gArr)
  {
    if (p.P1() >= p.P3())
    {
      std::vector<std::vector<std::vector<int>>> 
        canoBasis(p.SS() + 1, std::vector<std::vector<int>>(p.TT() + 1));

      std::vector<std::vector<Eigen::MatrixXd>> 
        transformation(p.SS() + 1, std::vector<Eigen::MatrixXd>(p.TT() + 1));  

      for (int MMS = p.SS(); MMS >= -p.SS(); MMS -= 2)
      {
        int indexS = (p.SS() - MMS) / 2;

        if (MMS == p.SS())
        {
          std::tie(canoBasis[0][0], transformation[0][0]) = 
                                    u4::phys_to_cano(p, p.SS(), p.TT(), gArr);
        }
              
        else // if (MMS < p.SS)
        {
          std::tie(canoBasis[indexS][0], transformation[indexS][0]) =
            u4::lowerMS(p, MMS, p.TT(), gArr,
              canoBasis[indexS - 1][0], transformation[indexS - 1][0]);
        }

        for (int MMT = p.TT() - 2; MMT >= -p.TT(); MMT -= 2)
        {
          int indexT = (p.TT() - MMT) / 2;

          std::tie(canoBasis[indexS][indexT],transformation[indexS][indexT]) =
            u4::lowerMT(p, MMS, MMT, gArr,
              canoBasis[indexS][indexT - 1], transformation[indexS][indexT - 1]);
        }
      }

      return std::make_pair(canoBasis, transformation);
    } 
    else 
    {
      std::vector<std::vector<std::vector<int>>> 
        canoBasis(p.SS() + 1, std::vector<std::vector<int>>(p.TT() + 1));

      std::vector<std::vector<Eigen::MatrixXd>> 
        transformation(p.SS() + 1, std::vector<Eigen::MatrixXd>(p.TT() + 1));  

      for (int MMS = -p.SS(); MMS <= p.SS(); MMS += 2)
      {
        int indexS = (p.SS() - MMS) / 2, indexT = p.TT();

        if (MMS == -p.SS())
        {
          std::tie(canoBasis[indexS][indexT], transformation[indexS][indexT]) = 
                                    u4::phys_to_cano(p, MMS, -p.TT(), gArr);
        }
              
        else // if (MMS < p.SS)
        {
          std::tie(canoBasis[indexS][indexT], transformation[indexS][indexT]) =
            u4::raiseMS(p, MMS, -p.TT(), gArr,
              canoBasis[indexS + 1][indexT], transformation[indexS + 1][indexT]);
        }

        for (int MMT = -p.TT() + 2; MMT <= p.TT(); MMT += 2)
        {
          indexT = (p.TT() - MMT) / 2;

          std::tie(canoBasis[indexS][indexT],transformation[indexS][indexT]) =
            u4::raiseMT(p, MMS, MMT, gArr,
              canoBasis[indexS][indexT + 1], transformation[indexS][indexT + 1]);
        }
      }

      return std::make_pair(canoBasis, transformation);
    }    
  }

} // namespace u4

#endif

/*
  // finding the transformation by solving a system of linear equations 
  // resulted from acting S+ = E13 + E24 and T+ = E12 + E34
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

#ifdef NULLSPACE_SPARSE
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
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr[indices[j]].e24_13_12());
#else
          projector(i, j) = gArr[indices[j]].e24_13_12();
#endif
        else if (u4::E24_13_22(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr[indices[j]].e24_13_22());
#else
          projector(i, j) = gArr[indices[j]].e24_13_22();
#endif
        else if (u4::E24_23_12(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr[indices[j]].e24_23_12());
#else
          projector(i, j) = gArr[indices[j]].e24_23_12();
#endif
        else if (u4::E24_23_22(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr[indices[j]].e24_23_22());
#else
          projector(i, j) = gArr[indices[j]].e24_23_22();
#endif
        else if (u4::E24_33_12(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr[indices[j]].e24_33_12());
#else
          projector(i, j) = gArr[indices[j]].e24_33_12();
#endif
        else if (u4::E24_33_22(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr[indices[j]].e24_33_22());
#else
          projector(i, j) = gArr[indices[j]].e24_33_22();
#endif

        // action of E13
        else if (u4::E13_12(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr[indices[j]].e13_12());
#else
          projector(i, j) = gArr[indices[j]].e13_12();
#endif
        else if (u4::E13_22(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr[indices[j]].e13_22());
#else
          projector(i, j) = gArr[indices[j]].e13_22();
#endif

        // action of E34
        else if (u4::E34_13(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr[indices[j]].e34_13());
#else
          projector(i, j) = gArr[indices[j]].e34_13();
#endif
        else if (u4::E34_23(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr[indices[j]].e34_23());
#else
          projector(i, j) = gArr[indices[j]].e34_23();
#endif
        else if (u4::E34_33(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr[indices[j]].e34_33());
#else
          projector(i, j) = gArr[indices[j]].e34_33();
#endif

        // action of E12
        else if (u4::E12(gArr[indices[j]], gArr[new_indices[i]]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr[indices[j]].e12());
#else
          projector(i, j) = gArr[indices[j]].e12();
#endif

        else
#ifdef NULLSPACE_SPARSE
          ;
#else
          projector(i, j) = 0.0;
#endif
      }
    }

#ifdef NULLSPACE_SPARSE
    Eigen::SparseMatrix<double> projector(row_num, col_num);
    projector.setFromTriplets(ija.begin(), ija.end());
    projector.makeCompressed();
#endif

    return projector;
  }

  // action of the lowering operator S- = E31 + E42
  inline auto lowerMS(
    const u4::physicalvector& p, 
    const int& MMS, // MMS is for the new state
    const int& MMT, // MMT is for the old state, unchanged
    const std::vector<u4::gelfand>& gArr,
    const std::vector<int>& canoBasis,
    const Eigen::MatrixXd& transformation)
  {
    // normalize coefficient
    double n = std::sqrt(4.0 / ((p.SS - MMS) * (p.SS + MMS + 2.0)));

    // new expansion
    std::vector<int> canoBasis_new = u4::expand_to_gelfand(MMS, MMT, gArr);

    Eigen::MatrixXd transformation_new(canoBasis_new.size(), p.zeta);
    transformation_new.setZero();

    // basis states are obtained by acting with two generators E31 and E42
    for (int i = 0; i < canoBasis_new.size(); i++)
    {
      for (int j = 0; j < canoBasis.size(); j++)
      {
        if (u4::E13_12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          transformation_new.row(i) +=
                transformation.row(j) * gArr[canoBasis[j]].e31_12();
        else if (u4::E13_22(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          transformation_new.row(i) +=
                transformation.row(j) * gArr[canoBasis[j]].e31_22();

        else if (u4::E24_13_12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          transformation_new.row(i) +=
                transformation.row(j) * gArr[canoBasis[j]].e42_13_12();
        else if (u4::E24_13_22(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          transformation_new.row(i) +=
                transformation.row(j) * gArr[canoBasis[j]].e42_13_22();
        else if (u4::E24_23_12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          transformation_new.row(i) +=
                transformation.row(j) * gArr[canoBasis[j]].e42_23_12();
        else if (u4::E24_23_22(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          transformation_new.row(i) +=
                transformation.row(j) * gArr[canoBasis[j]].e42_23_22();
        else if (u4::E24_33_12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          transformation_new.row(i) +=
                transformation.row(j) * gArr[canoBasis[j]].e42_33_12();
        else if (u4::E24_33_22(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          transformation_new.row(i) +=
                transformation.row(j) * gArr[canoBasis[j]].e42_33_22();
      }

      transformation_new.row(i) *= n;

      for (auto& x : transformation_new.row(i))
        if (std::fabs(x) < 1e-12) x = 0.0;
    }
    return transformation_new;
  }

  // action of the lowering operator T = E21 + E43
  inline Eigen::MatrixXd lowerMT(
    const u4::physicalvector& p, int MMT, // MMT is for the new state
    const std::vector<u4::gelfand>& gArr,
    const std::vector<int>& canoBasis,
    const Eigen::MatrixXd& transformation)
  {
    // normalize coefficient
    double n =  std::sqrt(4.0 / ((p.TT - MMT) * (p.TT + MMT + 2.0)));

    Eigen::MatrixXd transformation_new(canoBasis_new.size(), zeta_max);
    transformation_new.setZero();

    // basis states are obtained by acting with two generators E21 and E43
    for (int i = 0; i < canoBasis_new.size(); i++)
    {
      for (int j = 0; j < canoBasis.size(); j++)
      {
        if (u4::E12(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          transformation_new.row(i) +=
                transformation.row(j) * gArr[canoBasis[j]].e21();

        else if (u4::E34_13(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          transformation_new.row(i) +=
                transformation.row(j) * gArr[canoBasis[j]].e43_13();
        else if (u4::E34_23(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          transformation_new.row(i) +=
                transformation.row(j) * gArr[canoBasis[j]].e43_23();
        else if (u4::E34_33(gArr[canoBasis_new[i]], gArr[canoBasis[j]]))
          transformation_new.row(i) +=
                transformation.row(j) * gArr[canoBasis[j]].e43_33();
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
*/

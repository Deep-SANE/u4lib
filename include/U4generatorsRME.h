#ifndef U4LIB_GENERATORSRME_H
#define U4LIB_GENERATORSRME_H

#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "U4basis.h"
#include "U4SU2SU2_to_U4U3U2U1.h"

namespace u4
{

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  RME_S(u4::physicalvector& bra, u4::physicalvector& ket,
        std::vector<int>& bra_zeta, std::vector<int>& ket_zeta)
  {

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> RME(bra.zeta,
                                                              ket.zeta);

    // running through the multiplicity labels
    for (int i = 1; i <= bra.zeta; i++)
      bra_zeta.push_back(i);
    for (int j = 1; j <= ket.zeta; j++)
      ket_zeta.push_back(j);
    for (int i = 0; i < RME.rows(); i++)
    {
      for (int j = 0; j < RME.cols(); j++)
      {
        if (bra.SS != ket.SS || bra.TT != ket.TT)
        {
          RME(i, j) = 0.0;
        }
        else
        {
          // Note that my convention for reduced matrix elements follows a
          // standard convention, which involves an extra factor of
          // sqrt((2S+1)(2T+1))
          RME(i, j) =
              std::sqrt(bra.SS / 2.0 * (bra.SS / 2.0 + 1.0) * (bra.SS + 1.0) *
                        (bra.TT + 1.0)); // sqrt(S*(S+1)*(2S+1)*(2T+1))
        }
      }
    }

    return RME;
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  RME_T(u4::physicalvector& bra, u4::physicalvector& ket,
        std::vector<int>& bra_zeta, std::vector<int>& ket_zeta)
  {

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> RME(bra.zeta,
                                                              ket.zeta);

    // running through the multiplicity labels
    for (int i = 1; i <= bra.zeta; i++)
      bra_zeta.push_back(i);
    for (int j = 1; j <= ket.zeta; j++)
      ket_zeta.push_back(j);
    for (int i = 0; i < RME.rows(); i++)
    {
      for (int j = 0; j < RME.cols(); j++)
      {
        if (bra.TT != ket.TT || bra.TT != ket.TT)
        {
          RME(i, j) = 0.0;
        }
        else
        {
          // Note that my convention for reduced matrix elements follows a
          // standard convention, which involves an extra factor of sqrt(2T+1)
          RME(i, j) =
              std::sqrt(bra.TT / 2.0 * (bra.TT / 2.0 + 1.0) * (bra.TT + 1.0) *
                        (bra.SS + 1.0)); // sqrt(T*(T+1)*(2T+1)*(2S+1))
        }
      }
    }

    return RME;
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  RME_U(u4::physicalvector& bra, u4::physicalvector& ket,
        std::vector<int>& bra_zeta, std::vector<int>& ket_zeta)
  {

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> RME(bra.zeta,
                                                              ket.zeta);
    RME.setZero();

    // getting the tensor components of U
    int ss = bra.SS - ket.SS;
    int tt = bra.TT - ket.TT;

    // running through the multiplicity labels
    for (int i = 1; i <= bra.zeta; i++)
      bra_zeta.push_back(i);

    for (int j = 1; j <= ket.zeta; j++)
      ket_zeta.push_back(j);

    // compute non-zero RME's
    if (ss == 2 && tt == 2)
    {
      // Implementation here is based on Eq. (72) in F. Pan et al, Eur. Phys. J.
      // Plus 138, 662 (2023) Note that my convention for reduced matrix
      // elements follows a more standard convention, which involves an extra
      // factor of sqrt((2S'+1)(2T'+1))

      // Expanding in canonical basis
      std::vector<u4::gelfand> bra_gArr = u4::gelfandBasis(bra.toprow);
      std::vector<int> bra_indices;
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> bra_coeffs =
        u4::phys_to_cano(bra, bra.SS, bra.TT, bra_indices, bra_gArr);

      std::vector<u4::gelfand> ket_gArr = u4::gelfandBasis(ket.toprow);
      std::vector<int> ket_indices;
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ket_coeffs =
        u4::phys_to_cano(ket, ket.SS, ket.TT, ket_indices, ket_gArr);

      double c = std::sqrt((bra.SS + 1.0) * (bra.TT + 1.0));

      // running through the gelfand states
      for (int k = 0; k < ket_indices.size(); k++)
      {
        for (int b = 0; b < bra_indices.size(); b++)
        {
          if (u4::E14_13_12(ket_gArr[ket_indices[k]], bra_gArr[bra_indices[b]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e14_13_12();
          }
          else if (u4::E14_13_22(ket_gArr[ket_indices[k]], bra_gArr[bra_indices[b]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e14_13_22();
          }
          else if (u4::E14_23_12(ket_gArr[ket_indices[k]], bra_gArr[bra_indices[b]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e14_23_12();
          }
          else if (u4::E14_23_22(ket_gArr[ket_indices[k]], bra_gArr[bra_indices[b]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e14_23_22();
          }
          else if (u4::E14_33_12(ket_gArr[ket_indices[k]], bra_gArr[bra_indices[b]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e14_33_12();
          }
          else if (u4::E14_33_22(ket_gArr[ket_indices[k]], bra_gArr[bra_indices[b]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e14_33_22();
          }
        }
      }
    }

    else if (ss == 2 && tt == 0)
    {
      // Implementation here is based on Eq. (73) in F. Pan et al, Eur. Phys. J.
      // Plus 138, 662 (2023) Note that my convention for reduced matrix
      // elements follows a more standard convention, which involves an extra
      // factor of sqrt((2S'+1)(2T'+1))

      // Expanding in canonical basis
      std::vector<u4::gelfand> bra_gArr = u4::gelfandBasis(bra.toprow);
      std::vector<int> bra_indices;
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> bra_coeffs =
        u4::phys_to_cano(bra, bra.SS, bra.TT, bra_indices, bra_gArr);

      std::vector<u4::gelfand> ket_gArr = u4::gelfandBasis(ket.toprow);
      std::vector<int> ket_indices;
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ket_coeffs =
        u4::phys_to_cano(ket, ket.SS, ket.TT, ket_indices, ket_gArr);

      double c = std::sqrt((bra.SS + 1.0) * (bra.TT + 1.0) *
                           (ket.TT / 2.0 + 1.0) / ket.TT);

      // running through the gelfand states
      for (int k = 0; k < ket_indices.size(); k++)
      {
        for (int b = 0; b < bra_indices.size(); b++)
        {
          // E24
          if (u4::E24_13_12(ket_gArr[ket_indices[k]], bra_gArr[bra_indices[b]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e24_13_12();
          }
          else if (u4::E24_13_22(ket_gArr[ket_indices[k]], bra_gArr[bra_indices[b]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e24_13_22();
          }
          else if (u4::E24_23_12(ket_gArr[ket_indices[k]], bra_gArr[bra_indices[b]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e24_23_12();
          }
          else if (u4::E24_23_22(ket_gArr[ket_indices[k]], bra_gArr[bra_indices[b]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e24_23_22();
          }
          else if (u4::E24_33_12(ket_gArr[ket_indices[k]], bra_gArr[bra_indices[b]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e24_33_12();
          }
          else if (u4::E24_33_22(ket_gArr[ket_indices[k]], bra_gArr[bra_indices[b]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e24_33_22();
          }

          //-E13
          else if (u4::E13_12(ket_gArr[ket_indices[k]], bra_gArr[bra_indices[b]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) -= c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e13_12();
          }
          else if (u4::E13_22(ket_gArr[ket_indices[k]], bra_gArr[bra_indices[b]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) -= c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e13_22();
          }
        }
      }
    }

    else if (ss == 0 && tt == 0)
    {

      // Implementation here is based on Eq. (74) in F. Pan et al, Eur. Phys. J.
      // Plus 138, 662 (2023) Note that my convention for reduced matrix
      // elements follows a more standard convention, which involves an extra
      // factor of sqrt((2S'+1)(2T'+1))

      // Expanding in canonical basis
      std::vector<u4::gelfand> ket_gArr = u4::gelfandBasis(ket.toprow);
      std::vector<int> ket_indices;
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ket_coeffs =
        u4::phys_to_cano(ket, ket.SS, ket.TT, ket_indices, ket_gArr);

      double c =
          std::sqrt((bra.SS + 1.0) * (bra.TT + 1.0) * (ket.SS / 2.0 + 1.0) *
                    (ket.TT / 2.0 + 1.0) / ket.SS / ket.TT) / 2.0;

      // running through the gelfand states
      for (int k = 0; k < ket_indices.size(); k++)
      {
        // running through the multiplicity labels
        for (int i = 0; i < ket.zeta; i++)
          for (int j = 0; j < ket.zeta; j++)
            RME(i, j) += c * ket_coeffs(k, i) * ket_coeffs(k, j) *
                         (ket_gArr[ket_indices[k]].e11() - 
                          ket_gArr[ket_indices[k]].e22() -
                          ket_gArr[ket_indices[k]].e33() + 
                          ket_gArr[ket_indices[k]].e44());
      }
    }

    else if (ss == -2 && tt == 2)
    {
      // Implementation here is based on Eq. (75) in F. Pan et al, Eur. Phys. J.
      // Plus 138, 662 (2023) Note that my convention for reduced matrix
      // elements follows a more standard convention, which involves an extra
      // factor of sqrt((2S'+1)(2T'+1))

      // Expanding in canonical basis
      std::vector<u4::gelfand> bra_gArr = u4::gelfandBasis(bra.toprow);
      std::vector<int> bra_indices;
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> bra_coeffs =
        u4::phys_to_cano(bra, bra.SS, bra.TT, bra_indices, bra_gArr);

      std::vector<u4::gelfand> ket_gArr = u4::gelfandBasis(ket.toprow);
      std::vector<int> ket_indices;
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ket_coeffs =
        u4::phys_to_cano(ket, ket.SS, ket.TT, ket_indices, ket_gArr);

      double c = -std::sqrt((bra.SS + 1.0) * (bra.TT + 1.0) * (ket.SS + 1.0) /
                            (ket.SS - 1.0));

      // running through the gelfand states
      for (int k = 0; k < ket_indices.size(); k++)
      {
        for (int b = 0; b < bra_indices.size(); b++)
        {
          // E32
          if (u4::E23_12(bra_gArr[bra_indices[b]],ket_gArr[ket_indices[k]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e32_12();
          }
          else if (u4::E23_22(bra_gArr[bra_indices[b]],ket_gArr[ket_indices[k]]))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
              for (int j = 0; j < ket.zeta; j++)
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_gArr[ket_indices[k]].e32_22();
          }
        }
      }
    }

    else if ((ss == 2 && tt == -2) || (ss == 0 && tt == 2))
    {

      u4::physicalvector new_bra(bra.toprow, bra.TT, bra.SS),
          new_ket(ket.toprow, ket.TT, ket.SS);
      std::vector<int> new_bra_zeta, new_ket_zeta;

      RME = RME_U(new_bra, new_ket, new_bra_zeta, new_ket_zeta);
      bra_zeta = new_bra_zeta;
      ket_zeta = new_ket_zeta;
    }

    else if ((ss == 0 && tt == -2) || (ss == -2 && tt == 0) ||
             (ss == -2 && tt == -2))
    {

      std::vector<int> new_bra_zeta, new_ket_zeta;

      RME = RME_U(ket, bra, new_ket_zeta, new_bra_zeta);
      int phase = (((ss + tt) / 2) % 2 == 0) ? 1 : -1;

      // running through the multiplicity labels
      for (int i = 0; i < RME.rows(); i++)
        for (int j = 0; j < RME.cols(); j++)
          RME(i, j) *= phase;
      RME.transpose();
      bra_zeta = new_bra_zeta;
      ket_zeta = new_ket_zeta;
    }

    return RME;
  }

} // namespace u4

#endif
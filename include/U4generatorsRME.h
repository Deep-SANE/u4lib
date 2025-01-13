#ifndef U4GENERATORSME_H
#define U4GENERATORSME_H

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

    int ss = bra.SS - ket.SS;
    int tt = bra.TT - ket.TT;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> RME(bra.zeta,
                                                              ket.zeta);

    // running through the multiplicity labels
    for (int i = 1; i <= bra.zeta; i++)
      bra_zeta.push_back(i);
    for (int j = 1; j <= ket.zeta; j++)
      ket_zeta.push_back(j);
    for (int i = 0; i < RME.rows(); i++)
      for (int j = 0; j < RME.cols(); j++)
        RME(i, j) = 0.0;

    // compute non-zero RME's
    if (ss == 2 && tt == 2)
    {

      // Implementation here is based on Eq. (72) in F. Pan et al, Eur. Phys. J.
      // Plus 138, 662 (2023) Note that my convention for reduced matrix
      // elements follows a more standard convention, which involves an extra
      // factor of sqrt((2S'+1)(2T'+1))

      // Expanding in canonical basis
      std::vector<u4::gelfand> bra_canobasis = u4::canonicalbasis(bra);
      std::vector<u4::gelfand> ket_canobasis = u4::canonicalbasis(ket);

      // Building the projection matrix
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> bra_STproj =
          STprojection(bra, bra_canobasis);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ket_STproj =
          STprojection(ket, ket_canobasis);
      // and solving its null space
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> bra_coeffs =
          phys_to_cano(bra, bra_STproj);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ket_coeffs =
          phys_to_cano(ket, ket_STproj);

      double c = std::sqrt((bra.SS + 1.0) * (bra.TT + 1.0));

      // running through the gelfand states
      for (int k = 0; k < ket_canobasis.size(); k++)
      {
        for (int b = 0; b < bra_canobasis.size(); b++)
        {

          if (bra_canobasis[b].n13 == (ket_canobasis[k].n13 + 1) &&
              bra_canobasis[b].n23 == ket_canobasis[k].n23 &&
              bra_canobasis[b].n33 == ket_canobasis[k].n33 &&
              bra_canobasis[b].n12 == (ket_canobasis[k].n12 + 1) &&
              bra_canobasis[b].n22 == ket_canobasis[k].n22 &&
              bra_canobasis[b].n11 == (ket_canobasis[k].n11 + 1))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e14_13_12();
              }
            }
          }
          else if (bra_canobasis[b].n13 == (ket_canobasis[k].n13 + 1) &&
                   bra_canobasis[b].n23 == ket_canobasis[k].n23 &&
                   bra_canobasis[b].n33 == ket_canobasis[k].n33 &&
                   bra_canobasis[b].n12 == ket_canobasis[k].n12 &&
                   bra_canobasis[b].n22 == (ket_canobasis[k].n22 + 1) &&
                   bra_canobasis[b].n11 == (ket_canobasis[k].n11 + 1))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e14_13_22();
              }
            }
          }
          else if (bra_canobasis[b].n13 == ket_canobasis[k].n13 &&
                   bra_canobasis[b].n23 == (ket_canobasis[k].n23 + 1) &&
                   bra_canobasis[b].n33 == ket_canobasis[k].n33 &&
                   bra_canobasis[b].n12 == (ket_canobasis[k].n12 + 1) &&
                   bra_canobasis[b].n22 == ket_canobasis[k].n22 &&
                   bra_canobasis[b].n11 == (ket_canobasis[k].n11 + 1))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e14_23_12();
              }
            }
          }
          else if (bra_canobasis[b].n13 == ket_canobasis[k].n13 &&
                   bra_canobasis[b].n23 == (ket_canobasis[k].n23 + 1) &&
                   bra_canobasis[b].n33 == ket_canobasis[k].n33 &&
                   bra_canobasis[b].n12 == ket_canobasis[k].n12 &&
                   bra_canobasis[b].n22 == (ket_canobasis[k].n22 + 1) &&
                   bra_canobasis[b].n11 == (ket_canobasis[k].n11 + 1))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e14_23_22();
              }
            }
          }
          else if (bra_canobasis[b].n13 == ket_canobasis[k].n13 &&
                   bra_canobasis[b].n23 == ket_canobasis[k].n23 &&
                   bra_canobasis[b].n33 == (ket_canobasis[k].n33 + 1) &&
                   bra_canobasis[b].n12 == (ket_canobasis[k].n12 + 1) &&
                   bra_canobasis[b].n22 == ket_canobasis[k].n22 &&
                   bra_canobasis[b].n11 == (ket_canobasis[k].n11 + 1))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e14_33_12();
              }
            }
          }
          else if (bra_canobasis[b].n13 == ket_canobasis[k].n13 &&
                   bra_canobasis[b].n23 == ket_canobasis[k].n23 &&
                   bra_canobasis[b].n33 == (ket_canobasis[k].n33 + 1) &&
                   bra_canobasis[b].n12 == ket_canobasis[k].n12 &&
                   bra_canobasis[b].n22 == (ket_canobasis[k].n22 + 1) &&
                   bra_canobasis[b].n11 == (ket_canobasis[k].n11 + 1))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e14_33_22();
              }
            }
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
      std::vector<u4::gelfand> bra_canobasis = u4::canonicalbasis(bra);
      std::vector<u4::gelfand> ket_canobasis = u4::canonicalbasis(ket);

      // Building the projection matrix
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> bra_STproj =
          STprojection(bra, bra_canobasis);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ket_STproj =
          STprojection(ket, ket_canobasis);
      // and solving its null space
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> bra_coeffs =
          phys_to_cano(bra, bra_STproj);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ket_coeffs =
          phys_to_cano(ket, ket_STproj);

      double c = std::sqrt((bra.SS + 1.0) * (bra.TT + 1.0) *
                           (ket.TT / 2.0 + 1.0) / ket.TT);

      // running through the gelfand states
      for (int k = 0; k < ket_canobasis.size(); k++)
      {
        for (int b = 0; b < bra_canobasis.size(); b++)
        {

          // E24
          if (bra_canobasis[b].n13 == (ket_canobasis[k].n13 + 1) &&
              bra_canobasis[b].n23 == ket_canobasis[k].n23 &&
              bra_canobasis[b].n33 == ket_canobasis[k].n33 &&
              bra_canobasis[b].n12 == (ket_canobasis[k].n12 + 1) &&
              bra_canobasis[b].n22 == ket_canobasis[k].n22 &&
              bra_canobasis[b].n11 == ket_canobasis[k].n11)
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e24_13_12();
              }
            }
          }
          else if (bra_canobasis[b].n13 == (ket_canobasis[k].n13 + 1) &&
                   bra_canobasis[b].n23 == ket_canobasis[k].n23 &&
                   bra_canobasis[b].n33 == ket_canobasis[k].n33 &&
                   bra_canobasis[b].n12 == ket_canobasis[k].n12 &&
                   bra_canobasis[b].n22 == (ket_canobasis[k].n22 + 1) &&
                   bra_canobasis[b].n11 == ket_canobasis[k].n11)
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e24_13_22();
              }
            }
          }
          else if (bra_canobasis[b].n13 == ket_canobasis[k].n13 &&
                   bra_canobasis[b].n23 == (ket_canobasis[k].n23 + 1) &&
                   bra_canobasis[b].n33 == ket_canobasis[k].n33 &&
                   bra_canobasis[b].n12 == (ket_canobasis[k].n12 + 1) &&
                   bra_canobasis[b].n22 == ket_canobasis[k].n22 &&
                   bra_canobasis[b].n11 == ket_canobasis[k].n11)
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e24_23_12();
              }
            }
          }
          else if (bra_canobasis[b].n13 == ket_canobasis[k].n13 &&
                   bra_canobasis[b].n23 == (ket_canobasis[k].n23 + 1) &&
                   bra_canobasis[b].n33 == ket_canobasis[k].n33 &&
                   bra_canobasis[b].n12 == ket_canobasis[k].n12 &&
                   bra_canobasis[b].n22 == (ket_canobasis[k].n22 + 1) &&
                   bra_canobasis[b].n11 == ket_canobasis[k].n11)
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e24_23_22();
              }
            }
          }
          else if (bra_canobasis[b].n13 == ket_canobasis[k].n13 &&
                   bra_canobasis[b].n23 == ket_canobasis[k].n23 &&
                   bra_canobasis[b].n33 == (ket_canobasis[k].n33 + 1) &&
                   bra_canobasis[b].n12 == (ket_canobasis[k].n12 + 1) &&
                   bra_canobasis[b].n22 == ket_canobasis[k].n22 &&
                   bra_canobasis[b].n11 == ket_canobasis[k].n11)
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e24_33_12();
              }
            }
          }
          else if (bra_canobasis[b].n13 == ket_canobasis[k].n13 &&
                   bra_canobasis[b].n23 == ket_canobasis[k].n23 &&
                   bra_canobasis[b].n33 == (ket_canobasis[k].n33 + 1) &&
                   bra_canobasis[b].n12 == ket_canobasis[k].n12 &&
                   bra_canobasis[b].n22 == (ket_canobasis[k].n22 + 1) &&
                   bra_canobasis[b].n11 == ket_canobasis[k].n11)
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e24_33_22();
              }
            }
          }

          //-E13
          else if (bra_canobasis[b].n13 == ket_canobasis[k].n13 &&
                   bra_canobasis[b].n23 == ket_canobasis[k].n23 &&
                   bra_canobasis[b].n33 == ket_canobasis[k].n33 &&
                   bra_canobasis[b].n12 == (ket_canobasis[k].n12 + 1) &&
                   bra_canobasis[b].n22 == ket_canobasis[k].n22 &&
                   bra_canobasis[b].n11 == (ket_canobasis[k].n11 + 1))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) -= c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e13_12();
              }
            }
          }
          else if (bra_canobasis[b].n13 == ket_canobasis[k].n13 &&
                   bra_canobasis[b].n23 == ket_canobasis[k].n23 &&
                   bra_canobasis[b].n33 == ket_canobasis[k].n33 &&
                   bra_canobasis[b].n12 == ket_canobasis[k].n12 &&
                   bra_canobasis[b].n22 == (ket_canobasis[k].n22 + 1) &&
                   bra_canobasis[b].n11 == (ket_canobasis[k].n11 + 1))
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) -= c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e13_22();
              }
            }
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
      std::vector<u4::gelfand> ket_canobasis = u4::canonicalbasis(ket);

      // Building the projection matrix
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ket_STproj =
          STprojection(ket, ket_canobasis);
      // and solving its null space
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ket_coeffs =
          phys_to_cano(ket, ket_STproj);

      double c =
          std::sqrt((bra.SS + 1.0) * (bra.TT + 1.0) * (ket.SS / 2.0 + 1.0) *
                    (ket.TT / 2.0 + 1.0) / ket.SS / ket.TT) /
          2.0;

      // running through the gelfand states
      for (int k = 0; k < ket_canobasis.size(); k++)
      {
        // running through the multiplicity labels
        for (int i = 0; i < ket.zeta; i++)
        {
          for (int j = 0; j < ket.zeta; j++)
          {
            RME(i, j) += c * ket_coeffs(k, i) * ket_coeffs(k, j) *
                         (ket_canobasis[k].e11() - ket_canobasis[k].e22() -
                          ket_canobasis[k].e33() + ket_canobasis[k].e44());
          }
        }
      }
    }

    else if (ss == -2 && tt == 2)
    {

      // Implementation here is based on Eq. (75) in F. Pan et al, Eur. Phys. J.
      // Plus 138, 662 (2023) Note that my convention for reduced matrix
      // elements follows a more standard convention, which involves an extra
      // factor of sqrt((2S'+1)(2T'+1))

      // Expanding in canonical basis
      std::vector<u4::gelfand> bra_canobasis = u4::canonicalbasis(bra);
      std::vector<u4::gelfand> ket_canobasis = u4::canonicalbasis(ket);

      // Building the projection matrix
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> bra_STproj =
          STprojection(bra, bra_canobasis);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ket_STproj =
          STprojection(ket, ket_canobasis);
      // and solving its null space
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> bra_coeffs =
          phys_to_cano(bra, bra_STproj);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ket_coeffs =
          phys_to_cano(ket, ket_STproj);

      double c = -std::sqrt((bra.SS + 1.0) * (bra.TT + 1.0) * (ket.SS + 1.0) /
                            (ket.SS - 1.0));

      // running through the gelfand states
      for (int k = 0; k < ket_canobasis.size(); k++)
      {
        for (int b = 0; b < bra_canobasis.size(); b++)
        {
          // E32
          if (bra_canobasis[b].n13 == ket_canobasis[k].n13 &&
              bra_canobasis[b].n23 == ket_canobasis[k].n23 &&
              bra_canobasis[b].n33 == ket_canobasis[k].n33 &&
              bra_canobasis[b].n12 == (ket_canobasis[k].n12 - 1) &&
              bra_canobasis[b].n22 == ket_canobasis[k].n22 &&
              bra_canobasis[b].n11 == ket_canobasis[k].n11)
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e32_12();
              }
            }
          }
          else if (bra_canobasis[b].n13 == ket_canobasis[k].n13 &&
                   bra_canobasis[b].n23 == ket_canobasis[k].n23 &&
                   bra_canobasis[b].n33 == ket_canobasis[k].n33 &&
                   bra_canobasis[b].n12 == ket_canobasis[k].n12 &&
                   bra_canobasis[b].n22 == (ket_canobasis[k].n22 - 1) &&
                   bra_canobasis[b].n11 == ket_canobasis[k].n11)
          {
            // running through the multiplicity labels
            for (int i = 0; i < bra.zeta; i++)
            {
              for (int j = 0; j < ket.zeta; j++)
              {
                RME(i, j) += c * bra_coeffs(b, i) * ket_coeffs(k, j) *
                             ket_canobasis[k].e32_22();
              }
            }
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
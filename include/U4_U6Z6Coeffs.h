// protection against multiple header file inclusion
#ifndef U4_U6Z6COEFFS_H
#define U4_U6Z6COEFFS_H

#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "decomposition.h"
#include "U4basis.h"
#include "U4U3U2U1xU4U3U2U1.h"

namespace u4
{

  // input by users: u4a, u4b, u4abc, u4c, u4ab, u4bc
  std::vector<double> U6(u4::irrep& u4a, u4::irrep& u4b, u4::irrep& u4abc,
                         u4::irrep& u4c, u4::irrep& u4ab,
                         std::vector<int>& etaab, std::vector<int>& etaab_c,
                         u4::irrep& u4bc, std::vector<int>& etabc,
                         std::vector<int>& etaa_bc)
  {
    std::vector<double> U6coeffs;

    u4::gelfand HW(
        u4abc.n14, u4abc.n24, u4abc.n34, u4abc.n44, u4abc.n14, u4abc.n24,
        u4abc.n34, u4abc.n14, u4abc.n24,
        u4abc.n14); // highest weight Gelfand of state of the final irrep
    u4::gelfand HW23(u4bc.n14, u4bc.n24, u4bc.n34, u4bc.n44, u4bc.n14, u4bc.n24,
                     u4bc.n34, u4bc.n14, u4bc.n24, u4bc.n14);

    // coupling <G1;G23|HW> etaa_bc
    std::vector<u4::gelfand> gArr1a_bc, gArr23a_bc,
        gArra_bc; // store basis states
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coeffsa_bc;
    u4::HighestWeightCoeffs(u4a, u4bc, HW, gArr1a_bc, gArr23a_bc, gArra_bc,
                            coeffsa_bc);
    int etaa_bc_max = coeffsa_bc.cols();

    // indices of <G1;HW23|HW>
    std::vector<int> index_G1HW23;
    for (int i = 0; i < gArr1a_bc.size(); i++)
      if (u4::compare(gArr23a_bc[i], HW23))
        index_G1HW23.push_back(i);

    // coupling <G1;G2|G12> etaab
    std::vector<u4::gelfand> gArr1ab, gArr2ab, gArr12ab; // store basis states
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coeffsab;

    u4::gelfand HWab(u4ab.n14, u4ab.n24, u4ab.n34, u4ab.n44, u4ab.n14, u4ab.n24,
                     u4ab.n34, u4ab.n14, u4ab.n24, u4ab.n14);
    u4::HighestWeightCoeffs(u4a, u4b, HWab, gArr1ab, gArr2ab, gArr12ab,
                            coeffsab);

    for (int w4 = u4ab.n44; w4 <= u4ab.n14; w4++)
    {
      for (int w1 = u4ab.n14; w1 >= u4ab.n44; w1--)
      {
        for (int w3 = 0;
             w3 <= u4ab.n14 + u4ab.n24 + u4ab.n34 + u4ab.n44 - w1 - w4; w3++)
        {
          if (w4 == u4ab.n44 && w3 == u4ab.n34 && w1 == u4ab.n14)
          {
            continue;
          }
          else
          {
            u4::LowerWeightCoeffs(u4a, u4b, u4ab, w4, w3, w1, gArr1ab, gArr2ab,
                                  gArr12ab, coeffsab);
          }
        }
      }
    }
    int etaab_max = coeffsab.cols();

    // coupling <G12;G3|HW> etaab_c
    std::vector<u4::gelfand> gArr12ab_c, gArr3ab_c,
        gArrab_c; // store basis states
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coeffsab_c;
    u4::HighestWeightCoeffs(u4ab, u4c, HW, gArr12ab_c, gArr3ab_c, gArrab_c,
                            coeffsab_c);
    int etaab_c_max = coeffsab_c.cols();

    // coupling <G2;G3|HW23> etabc
    std::vector<u4::gelfand> gArr2bc, gArr3bc, gArr23bc; // store basis states
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coeffsbc;
    u4::HighestWeightCoeffs(u4b, u4c, HW23, gArr2bc, gArr3bc, gArr23bc,
                            coeffsbc);
    int etabc_max = coeffsbc.cols();

    // U6 coefficients
    for (int eab = 0; eab < etaab_max; eab++)
    {
      for (int eab_c = 0; eab_c < etaab_c_max; eab_c++)
      {
        for (int ebc = 0; ebc < etabc_max; ebc++)
        {

          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matA(
              etaa_bc_max, etaa_bc_max);
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matB(
              etaa_bc_max, 1);

          for (int i = 0; i < etaa_bc_max; i++)
          {

            for (int j = 0; j < etaa_bc_max; j++)
              matA(i, j) = coeffsa_bc(index_G1HW23[i], j);

            matB(i, 0) = 0.0;
            for (int j = 0; j < coeffsab.rows(); j++)
            {
              if (u4::compare(gArr1ab[j], gArr1a_bc[index_G1HW23[i]]))
              {
                for (int k = 0; k < coeffsab_c.rows(); k++)
                {
                  if (u4::compare(gArr12ab[j], gArr12ab_c[k]))
                  {
                    for (int l = 0; l < coeffsbc.rows(); l++)
                    {
                      if (u4::compare(gArr2bc[l], gArr2ab[j]) &&
                          u4::compare(gArr3bc[l], gArr3ab_c[k]))
                      {
                        matB(i, 0) += coeffsab(j, eab) * coeffsab_c(k, eab_c) *
                                      coeffsbc(l, ebc);
                      }
                    }
                  }
                }
              }
            }
          }

          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> U =
              matA.colPivHouseholderQr().solve(matB);

          for (int ea_bc = 0; ea_bc < etaa_bc_max; ea_bc++)
          {
            etaab.push_back(eab + 1);
            etaab_c.push_back(eab_c + 1);
            etabc.push_back(ebc + 1);
            etaa_bc.push_back(ea_bc + 1);
            U6coeffs.push_back(U(ea_bc, 0));
          }
        }
      }
    }
    return U6coeffs;
  }

  // input by users: u4b, u4a, u4abc, u4c, u4ab, u4ac
  std::vector<double> Z6(u4::irrep& u4b, u4::irrep& u4a, u4::irrep& u4abc,
                         u4::irrep& u4c, u4::irrep& u4ab,
                         std::vector<int>& etaab, std::vector<int>& etaab_c,
                         u4::irrep& u4ac, std::vector<int>& etaac,
                         std::vector<int>& etaac_b)
  {
    std::vector<double> Z6coeffs;

    u4::gelfand HW(
        u4abc.n14, u4abc.n24, u4abc.n34, u4abc.n44, u4abc.n14, u4abc.n24,
        u4abc.n34, u4abc.n14, u4abc.n24,
        u4abc.n14); // highest weight Gelfand of state of the final irrep
    u4::gelfand HW13(u4ac.n14, u4ac.n24, u4ac.n34, u4ac.n44, u4ac.n14, u4ac.n24,
                     u4ac.n34, u4ac.n14, u4ac.n24, u4ac.n14);

    // coupling <G13;G2|HW> etaac_b
    std::vector<u4::gelfand> gArr13ac_b, gArr2ac_b,
        gArrac_b; // store basis states
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coeffsac_b;
    u4::HighestWeightCoeffs(u4ac, u4b, HW, gArr13ac_b, gArr2ac_b, gArrac_b,
                            coeffsac_b);
    int etaac_b_max = coeffsac_b.cols();

    // indices of <HW13;G2|HW>
    std::vector<int> index_HW13G2;
    for (int i = 0; i < gArr13ac_b.size(); i++)
      if (u4::compare(gArr13ac_b[i], HW13))
        index_HW13G2.push_back(i);

    // coupling <G1;G2|G12> etaab
    std::vector<u4::gelfand> gArr1ab, gArr2ab, gArr12ab; // store basis states
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coeffsab;

    u4::gelfand HWab(u4ab.n14, u4ab.n24, u4ab.n34, u4ab.n44, u4ab.n14, u4ab.n24,
                     u4ab.n34, u4ab.n14, u4ab.n24, u4ab.n14);
    u4::HighestWeightCoeffs(u4a, u4b, HWab, gArr1ab, gArr2ab, gArr12ab,
                            coeffsab);

    for (int w4 = u4ab.n44; w4 <= u4ab.n14; w4++)
    {
      for (int w1 = u4ab.n14; w1 >= u4ab.n44; w1--)
      {
        for (int w3 = 0;
             w3 <= u4ab.n14 + u4ab.n24 + u4ab.n34 + u4ab.n44 - w1 - w4; w3++)
        {
          if (w4 == u4ab.n44 && w3 == u4ab.n34 && w1 == u4ab.n14)
          {
            continue;
          }
          else
          {
            u4::LowerWeightCoeffs(u4a, u4b, u4ab, w4, w3, w1, gArr1ab, gArr2ab,
                                  gArr12ab, coeffsab);
          }
        }
      }
    }
    int etaab_max = coeffsab.cols();

    // coupling <G12;G3|HW> etaab_c
    std::vector<u4::gelfand> gArr12ab_c, gArr3ab_c,
        gArrab_c; // store basis states
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coeffsab_c;
    u4::HighestWeightCoeffs(u4ab, u4c, HW, gArr12ab_c, gArr3ab_c, gArrab_c,
                            coeffsab_c);
    int etaab_c_max = coeffsab_c.cols();

    // coupling <G1;G3|HW13> etaac
    std::vector<u4::gelfand> gArr1ac, gArr3ac, gArr13ac; // store basis states
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coeffsac;
    u4::HighestWeightCoeffs(u4a, u4c, HW13, gArr1ac, gArr3ac, gArr13ac,
                            coeffsac);
    int etaac_max = coeffsac.cols();

    // Z6 coefficients
    for (int eab = 0; eab < etaab_max; eab++)
    {
      for (int eab_c = 0; eab_c < etaab_c_max; eab_c++)
      {
        for (int eac = 0; eac < etaac_max; eac++)
        {

          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matA(
              etaac_b_max, etaac_b_max);
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matB(
              etaac_b_max, 1);

          for (int i = 0; i < etaac_b_max; i++)
          {

            for (int j = 0; j < etaac_b_max; j++)
              matA(i, j) = coeffsac_b(index_HW13G2[i], j);

            matB(i, 0) = 0.0;
            for (int j = 0; j < coeffsab.rows(); j++)
            {
              if (u4::compare(gArr2ab[j], gArr2ac_b[index_HW13G2[i]]))
              {
                for (int k = 0; k < coeffsab_c.rows(); k++)
                {
                  if (u4::compare(gArr12ab[j], gArr12ab_c[k]))
                  {
                    for (int l = 0; l < coeffsac.rows(); l++)
                    {
                      if (u4::compare(gArr1ac[l], gArr1ab[j]) &&
                          u4::compare(gArr3ac[l], gArr3ab_c[k]))
                      {
                        matB(i, 0) += coeffsab(j, eab) * coeffsab_c(k, eab_c) *
                                      coeffsac(l, eac);
                      }
                    }
                  }
                }
              }
            }
          }

          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Z =
              matA.colPivHouseholderQr().solve(matB);

          for (int eac_b = 0; eac_b < etaac_b_max; eac_b++)
          {
            etaab.push_back(eab + 1);
            etaab_c.push_back(eab_c + 1);
            etaac.push_back(eac + 1);
            etaac_b.push_back(eac_b + 1);
            Z6coeffs.push_back(Z(eac_b, 0));
          }
        }
      }
    }
    return Z6coeffs;
  }

} // namespace u4
#endif
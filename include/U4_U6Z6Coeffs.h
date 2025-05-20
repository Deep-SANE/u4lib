// protection against multiple header file inclusion
#ifndef U4_U6Z6COEFFS_H
#define U4_U6Z6COEFFS_H

#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "U4basis.h"
#include "U4U3xU4U3.h"

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>

#include <U3_U6Z6Coeffs.h>

namespace u4
{

  // input by users: u4a, u4b, u4abc, u4c, u4ab, u4bc, all normalized
  inline std::vector<double> 
    U6(const u4::irrep& u4A, const std::vector<u4::gelfand>& gArrA,  
       const u4::irrep& u4B, const std::vector<u4::gelfand>& gArrB,
       const u4::irrep& u4ABC, const std::vector<u4::gelfand>& gArrABC,
       const u4::irrep& u4C, const std::vector<u4::gelfand>& gArrC,
       const u4::irrep& u4AB, const std::vector<u4::gelfand>& gArrAB, 
       int etaABmax, int etaAB_Cmax,
       const u4::irrep& u4BC, const std::vector<u4::gelfand>& gArrBC, 
       int etaBCmax, int etaA_BCmax,
       u34utils::SO3_CGC& cgcs, u34utils::SO3_Racah& racahs,
       u3::WigU3U2_db& wigU3U2db)
  {
    // coupling <gA;hwBC|hwABC> rhoA_BC etaA_BC
    auto coeffsA_BC = u4::WigU4U3(u4A, u4BC, u4ABC, etaA_BCmax,     
                      u4ABC.n14() + u4ABC.n24() + u4ABC.n34(), 
                      gArrA, gArrBC, gArrABC, cgcs, wigU3U2db);

    // coupling <gAB;gC|hwABC> rhoAB_C etaAB_C
    auto coeffsAB_C = u4::WigU4U3(u4AB, u4C, u4ABC, etaAB_Cmax,     
                      u4ABC.n14() + u4ABC.n24() + u4ABC.n34(), 
                      gArrAB, gArrC, gArrABC, cgcs, wigU3U2db);

    // coupling <gB;gC|hwBC> rhoBC etaBC
    auto coeffsBC = u4::WigU4U3(u4B, u4C, u4BC, etaBCmax,     
                      u4BC.n14() + u4BC.n24() + u4BC.n34(), 
                      gArrB, gArrC, gArrBC, cgcs, wigU3U2db);

    // coupling <gA;gB|gAB> rhoAB etaAB
    auto coeffsAB = u4::WigU4U3(u4A, u4B, u4AB, etaABmax, -1, 
                      gArrA, gArrB, gArrAB, cgcs, wigU3U2db);

    Eigen::MatrixXd A(etaA_BCmax, etaA_BCmax);
    Eigen::MatrixXd B(etaA_BCmax, etaABmax * etaAB_Cmax * etaBCmax);
    B.setZero(); 

    int r = 0;

    for (int n13A = u4A.n14(); n13A >= u4A.n24(); n13A--)
    {
      for (int n23A = u4A.n24(); n23A >= u4A.n34(); n23A--)
      {
        for (int n33A = u4A.n34(); n33A >= u4A.n44(); n33A--)
        {
          u3::irrep u3A(n13A, n23A, n33A), 
                    u3BC(u4BC.n14(), u4BC.n24(), u4BC.n34()),
                    u3ABC(u4ABC.n14(), u4ABC.n24(), u4ABC.n34());
          auto [missingA_BC, accA_BC] = coeffsA_BC.access(u3A,u3BC,u3ABC);
          if (missingA_BC == false)
          {
            //std::cout << "accA_BC = " << accA_BC << std::endl;
            u3::irrep u3a = u3A, u3bc = u3BC, u3abc = u3ABC;
            u3a.normalize();
            u3bc.normalize();
            u3abc.normalize();

            std::vector<u3::gelfand> gArr3a = u3::gelfandBasis(u3a),
                                  gArr3bc = u3::gelfandBasis(u3bc),
                                  gArr3abc = u3::gelfandBasis(u3abc);  
            int rA_BC = 0;
            A.row(r) = accA_BC.row(rA_BC);

            for (int n13B = u4B.n14(); n13B >= u4B.n24(); n13B--)
            {
              for (int n23B = u4B.n24(); n23B >= u4B.n34(); n23B--)
              {
                for (int n33B = u4B.n34(); n33B >= u4B.n44(); n33B--)
                {
                  u3::irrep u3B(n13B, n23B, n33B);

                  for (int n13AB = u4AB.n14(); n13AB >= u4AB.n24(); n13AB--)
                  {
                    for (int n23AB = u4AB.n24(); n23AB >= u4AB.n34(); n23AB--)
                    {
                      for (int n33AB = u4AB.n34(); n33AB >= u4AB.n44(); n33AB--)
                      {
                        u3::irrep u3AB(n13AB, n23AB, n33AB);
                        auto [missingAB, accAB] = coeffsAB.access(u3A,u3B,u3AB);
                        if (missingAB == false)
                        {
                          //std::cout << "accAB = " << accAB << std::endl;
                          u3::irrep u3b = u3B, u3ab = u3AB;
                          u3b.normalize();
                          u3ab.normalize();

                          std::vector<u3::gelfand> gArr3b = u3::gelfandBasis(u3b),
                                                  gArr3ab = u3::gelfandBasis(u3ab);

                          for (int n13C = u4C.n14(); n13C >= u4C.n24(); n13C--)
                          {
                            for (int n23C = u4C.n24(); n23C >= u4C.n34(); n23C--)
                            {
                              for (int n33C = u4C.n34(); n33C >= u4C.n44(); n33C--)
                              {
                                u3::irrep u3C(n13C, n23C, n33C);

                                auto [missingAB_C, accAB_C] = 
                                                coeffsAB_C.access(u3AB, u3C, u3ABC);

                                auto [missingBC, accBC] = 
                                                coeffsBC.access(u3B, u3C, u3BC);            

                                if (missingAB_C == false && missingBC == false)
                                {
                                  //std::cout << u3AB << ";" << u3C << "|||" << u3ABC 
                                            //<< " -> accAB_C = " << accAB_C << std::endl;
                                  //std::cout << u3B << ";" << u3C << "|||" << u3BC 
                                            //<< " -> accBC = " << accBC << std::endl;
                                  u3::irrep u3c = u3C;
                                  u3c.normalize();
                                  std::vector<u3::gelfand> gArr3c = u3::gelfandBasis(u3c);

                                  std::vector<double> U6_coeffs = u3::U6(
                                    u3a, gArr3a, u3b, gArr3b, 
                                    u3abc, gArr3abc, u3c, gArr3c, 
                                    u3ab, gArr3ab, accAB.rows(), accAB_C.rows(), 
                                    u3bc, gArr3bc, accBC.rows(), accA_BC.rows(), 
                                    cgcs, racahs, wigU3U2db);

                                  for (int eAB = 0; eAB < etaABmax; eAB++)
                                    for (int eAB_C = 0; eAB_C < etaAB_Cmax; eAB_C++)
                                      for (int eBC = 0; eBC < etaBCmax; eBC++)
                                        for (int rAB = 0; rAB < accAB.rows(); rAB++)
                                          for (int rAB_C = 0; rAB_C < accAB_C.rows(); rAB_C++)
                                            for (int rBC = 0; rBC < accBC.rows(); rBC++)
                                              B(r, (eAB * etaAB_Cmax + eAB_C) * etaBCmax + eBC) 
                                              += accAB(rAB, eAB) * accAB_C(rAB_C, eAB_C) 
                                              * accBC(rBC, eBC) 
                                              * U6_coeffs[((rAB * accAB_C.rows() + rAB_C) *
                                                          accBC.rows() + rBC) * 
                                                          accA_BC.rows() + rA_BC];
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            r++;
          }  
        }
      }
      if (r == etaA_BCmax)
        break;
    }

    // std::cout << A << std::endl;
    // std::cout << B << std::endl;

    Eigen::MatrixXd X = A.householderQr().solve(B);

    std::vector<double> U(etaABmax * etaAB_Cmax * etaBCmax * etaA_BCmax);

    for (int eAB = 0; eAB < etaABmax; eAB++)
      for (int eAB_C = 0; eAB_C < etaAB_Cmax; eAB_C++)
        for (int eBC = 0; eBC < etaBCmax; eBC++)
          for (int eA_BC = 0; eA_BC < etaA_BCmax; eA_BC++)
          {
            int col = (eAB * etaAB_Cmax + eAB_C) * etaBCmax + eBC;
            if (std::fabs(X(eA_BC, col)) < 1e-14)
              X(eA_BC, col) = 0.0;

            U[((eAB * etaAB_Cmax + eAB_C) * etaBCmax + eBC) * etaA_BCmax +
              eA_BC] = X(eA_BC, col);
          }

    return U;
  }

  // input by users: u4a, u4b, u4abc, u4c, u4ab, u4bc, all normalized
  inline std::vector<double> 
    Z6(const u4::irrep& u4B, const std::vector<u4::gelfand>& gArrB,
       const u4::irrep& u4A, const std::vector<u4::gelfand>& gArrA,         
       const u4::irrep& u4ABC, const std::vector<u4::gelfand>& gArrABC,
       const u4::irrep& u4C, const std::vector<u4::gelfand>& gArrC,
       const u4::irrep& u4AB, const std::vector<u4::gelfand>& gArrAB, 
       int etaABmax, int etaAB_Cmax,
       const u4::irrep& u4AC, const std::vector<u4::gelfand>& gArrAC, 
       int etaACmax, int etaAC_Bmax,
       u34utils::SO3_CGC& cgcs, u34utils::SO3_Racah& racahs,
       u3::WigU3U2_db& wigU3U2db)
  {
    // coupling <hwAC;gB|hwABC> rhoAC_B etaAC_B
    auto coeffsAC_B = u4::WigU4U3(u4AC, u4B, u4ABC, etaAC_Bmax, 
                      u4ABC.n14() + u4ABC.n24() + u4ABC.n34(),
                      gArrAC, gArrB, gArrABC, cgcs, wigU3U2db);

    // coupling <gAB;gC|hwABC> rhoAB_C etaAB_C
    auto coeffsAB_C = u4::WigU4U3(u4AB, u4C, u4ABC, etaAB_Cmax, 
                      u4ABC.n14() + u4ABC.n24() + u4ABC.n34(),
                      gArrAB, gArrC, gArrABC, cgcs, wigU3U2db);

    // coupling <gA;gC|hwAC> rhoAC etaAC
    auto coeffsAC = u4::WigU4U3(u4A, u4C, u4AC, etaACmax, 
                    u4AC.n14() + u4AC.n24() + u4AC.n34(), 
                    gArrA, gArrC, gArrAC, cgcs, wigU3U2db);

    // coupling <gA;gB|gAB> rhoAB etaAB
    auto coeffsAB = u4::WigU4U3(u4A, u4B, u4AB, etaABmax, -1, 
                    gArrA, gArrB, gArrAB, cgcs, wigU3U2db);

    Eigen::MatrixXd A(etaAC_Bmax, etaAC_Bmax);
    Eigen::MatrixXd B(etaAC_Bmax, etaABmax * etaAB_Cmax * etaACmax);
    B.setZero(); 

    int r = 0;

    for (int n13B = u4B.n14(); n13B >= u4B.n24(); n13B--)
    {
      for (int n23B = u4B.n24(); n23B >= u4B.n34(); n23B--)
      {
        for (int n33B = u4B.n34(); n33B >= u4B.n44(); n33B--)
        {
          u3::irrep u3B(n13B, n23B, n33B), 
                    u3AC(u4AC.n14(), u4AC.n24(), u4AC.n34()),
                    u3ABC(u4ABC.n14(), u4ABC.n24(), u4ABC.n34());
          auto [missingAC_B, accAC_B] = coeffsAC_B.access(u3AC,u3B,u3ABC);
          if (missingAC_B == false)
          {
            //std::cout << "accAC_B = " << accAC_B << std::endl;
            u3::irrep u3b = u3B, u3ac = u3AC, u3abc = u3ABC;
            u3b.normalize();
            u3ac.normalize();
            u3abc.normalize();

            std::vector<u3::gelfand> gArr3b = u3::gelfandBasis(u3b),
                                  gArr3ac = u3::gelfandBasis(u3ac),
                                  gArr3abc = u3::gelfandBasis(u3abc);  
            int rAC_B = 0;
            A.row(r) = accAC_B.row(rAC_B);

            for (int n13A = u4A.n14(); n13A >= u4A.n24(); n13A--)
            {
              for (int n23A = u4A.n24(); n23A >= u4A.n34(); n23A--)
              {
                for (int n33A = u4A.n34(); n33A >= u4A.n44(); n33A--)
                {
                  u3::irrep u3A(n13A, n23A, n33A);

                  for (int n13AB = u4AB.n14(); n13AB >= u4AB.n24(); n13AB--)
                  {
                    for (int n23AB = u4AB.n24(); n23AB >= u4AB.n34(); n23AB--)
                    {
                      for (int n33AB = u4AB.n34(); n33AB >= u4AB.n44(); n33AB--)
                      {
                        u3::irrep u3AB(n13AB, n23AB, n33AB);
                        auto [missingAB, accAB] = coeffsAB.access(u3A,u3B,u3AB);
                        if (missingAB == false)
                        {
                          //std::cout << "accAB = " << accAB << std::endl;
                          u3::irrep u3a = u3A, u3ab = u3AB;
                          u3a.normalize();
                          u3ab.normalize();

                          std::vector<u3::gelfand> gArr3a = u3::gelfandBasis(u3a),
                                                  gArr3ab = u3::gelfandBasis(u3ab);

                          for (int n13C = u4C.n14(); n13C >= u4C.n24(); n13C--)
                          {
                            for (int n23C = u4C.n24(); n23C >= u4C.n34(); n23C--)
                            {
                              for (int n33C = u4C.n34(); n33C >= u4C.n44(); n33C--)
                              {
                                u3::irrep u3C(n13C, n23C, n33C);

                                auto [missingAB_C, accAB_C] = 
                                                coeffsAB_C.access(u3AB, u3C, u3ABC);

                                auto [missingAC, accAC] = 
                                                coeffsAC.access(u3A, u3C, u3AC);            

                                if (missingAB_C == false && missingAC == false)
                                {
                                  //std::cout << u3AB << ";" << u3C << "|||" << u3ABC 
                                            //<< " -> accAB_C = " << accAB_C << std::endl;
                                  //std::cout << u3A << ";" << u3C << "|||" << u3AC 
                                            //<< " -> accAC = " << accAC << std::endl;
                                  u3::irrep u3c = u3C;
                                  u3c.normalize();
                                  std::vector<u3::gelfand> gArr3c = u3::gelfandBasis(u3c);

                                  std::vector<double> Z6_coeffs = u3::Z6(
                                    u3b, gArr3b, u3a, gArr3a, 
                                    u3abc, gArr3abc, u3c, gArr3c, 
                                    u3ab, gArr3ab, accAB.rows(), accAB_C.rows(), 
                                    u3ac, gArr3ac, accAC.rows(), accAC_B.rows(), 
                                    cgcs, racahs, wigU3U2db);

                                  for (int eAB = 0; eAB < etaABmax; eAB++)
                                    for (int eAB_C = 0; eAB_C < etaAB_Cmax; eAB_C++)
                                      for (int eAC = 0; eAC < etaACmax; eAC++)
                                        for (int rAB = 0; rAB < accAB.rows(); rAB++)
                                          for (int rAB_C = 0; rAB_C < accAB_C.rows(); rAB_C++)
                                            for (int rAC = 0; rAC < accAC.rows(); rAC++)
                                              B(r, (eAB * etaAB_Cmax + eAB_C) * etaACmax + eAC) 
                                              += accAB(rAB, eAB) * accAB_C(rAB_C, eAB_C) 
                                              * accAC(rAC, eAC) 
                                              * Z6_coeffs[((rAB * accAB_C.rows() + rAB_C) *
                                                          accAC.rows() + rAC) * 
                                                          accAC_B.rows() + rAC_B];
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            r++;
          }  
        }
      }
      if (r == etaAC_Bmax)
        break;
    }

    // std::cout << A << std::endl;
    // std::cout << B << std::endl;

    Eigen::MatrixXd X = A.householderQr().solve(B);

    std::vector<double> Z(etaABmax * etaAB_Cmax * etaACmax * etaAC_Bmax);

    for (int eAB = 0; eAB < etaABmax; eAB++)
      for (int eAB_C = 0; eAB_C < etaAB_Cmax; eAB_C++)
        for (int eAC = 0; eAC < etaACmax; eAC++)
          for (int eAC_B = 0; eAC_B < etaAC_Bmax; eAC_B++)
          {
            int col = (eAB * etaAB_Cmax + eAB_C) * etaACmax + eAC;
            if (std::fabs(X(eAC_B, col)) < 1e-14)
              X(eAC_B, col) = 0.0;

            Z[((eAB * etaAB_Cmax + eAB_C) * etaACmax + eAC) * etaAC_Bmax +
              eAC_B] = X(eAC_B, col);
          }

    return Z;
  }
} // namespace u4
#endif


/*
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
*/
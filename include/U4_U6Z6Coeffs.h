// protection against multiple header file inclusion
#ifndef U4LIB_U4_U6Z6COEFFS_H
#define U4LIB_U4_U6Z6COEFFS_H

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
     int etaABmax, int etaAB_Cmax, const u4::irrep& u4BC,
     const std::vector<u4::gelfand>& gArrBC, int etaBCmax, int etaA_BCmax,
     qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah& racahs,
     u3::WigU3U2_db& wigU3U2db, u4::WigU4U3_db& wigU4U3db)
  {
    /*
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
    */
    Eigen::MatrixXd A(etaA_BCmax, etaA_BCmax);
    Eigen::MatrixXd B(etaA_BCmax, etaABmax * etaAB_Cmax * etaBCmax);
    B.setZero();

    int r = 0;
    bool done = false;
    for (int n13A = u4A.n14(); n13A >= u4A.n24() && !done; n13A--)
    {
      for (int n23A = u4A.n24(); n23A >= u4A.n34() && !done; n23A--)
      {
        for (int n33A = u4A.n34(); n33A >= u4A.n44() && !done; n33A--)
        {
          /* u3::irrep u3A(n13A, n23A, n33A),
                    u3BC(u4BC.n14(), u4BC.n24(), u4BC.n34()),
                    u3ABC(u4ABC.n14(), u4ABC.n24(), u4ABC.n34());
          auto [existA_BC, accA_BC] = coeffsA_BC.access(u3A,u3BC,u3ABC); */

          auto [existA_BC, accA_BC] = wigU4U3db.get(
              u4A.n14(), u4A.n24(), u4A.n34(), u4A.n44(), n13A, n23A, n33A,
              u4BC.n14(), u4BC.n24(), u4BC.n34(), u4BC.n44(), u4BC.n14(),
              u4BC.n24(), u4BC.n34(), u4ABC.n14(), u4ABC.n24(), u4ABC.n34(),
              u4ABC.n44(), u4ABC.n14(), u4ABC.n24(), u4ABC.n34());

          if (existA_BC == true)
          {
            // std::cout << "accA_BC = " << accA_BC << std::endl;
            u3::irrep u3a(n13A - n33A, n23A - n33A, 0),
                u3bc(u4BC.n14() - u4BC.n34(), u4BC.n24() - u4BC.n34(), 0),
                u3abc(u4ABC.n14() - u4ABC.n34(), u4ABC.n24() - u4ABC.n34(), 0);
            // u3a.normalize();
            // u3bc.normalize();
            // u3abc.normalize();

            std::vector<u3::gelfand> gArr3a = u3::gelfandBasis(u3a),
                                     gArr3bc = u3::gelfandBasis(u3bc),
                                     gArr3abc = u3::gelfandBasis(u3abc);

            for (int rA_BC = 0; rA_BC < accA_BC.rows(); rA_BC++)
            {
              A.row(r) = accA_BC.row(rA_BC);

              for (int n13B = u4B.n14(); n13B >= u4B.n24(); n13B--)
              {
                for (int n23B = u4B.n24(); n23B >= u4B.n34(); n23B--)
                {
                  for (int n33B = u4B.n34(); n33B >= u4B.n44(); n33B--)
                  {
                    // u3::irrep u3B(n13B, n23B, n33B);

                    for (int n13AB = u4AB.n14(); n13AB >= u4AB.n24(); n13AB--)
                    {
                      for (int n23AB = u4AB.n24(); n23AB >= u4AB.n34(); n23AB--)
                      {
                        for (int n33AB = u4AB.n34(); n33AB >= u4AB.n44();
                             n33AB--)
                        {
                          // u3::irrep u3AB(n13AB, n23AB, n33AB);
                          // auto [existAB, accAB] =
                          // coeffsAB.access(u3A,u3B,u3AB);

                          auto [existAB, accAB] = wigU4U3db.get(
                              u4A.n14(), u4A.n24(), u4A.n34(), u4A.n44(), n13A,
                              n23A, n33A, u4B.n14(), u4B.n24(), u4B.n34(),
                              u4B.n44(), n13B, n23B, n33B, u4AB.n14(),
                              u4AB.n24(), u4AB.n34(), u4AB.n44(), n13AB, n23AB,
                              n33AB);

                          if (existAB == true)
                          {
                            // std::cout << "accAB = " << accAB << std::endl;
                            u3::irrep u3b(n13B - n33B, n23B - n33B, 0),
                                u3ab(n13AB - n33AB, n23AB - n33AB, 0);
                            // u3b.normalize();
                            // u3ab.normalize();

                            std::vector<u3::gelfand> gArr3b =
                                                         u3::gelfandBasis(u3b),
                                                     gArr3ab =
                                                         u3::gelfandBasis(u3ab);

                            for (int n13C = u4C.n14(); n13C >= u4C.n24();
                                 n13C--)
                            {
                              for (int n23C = u4C.n24(); n23C >= u4C.n34();
                                   n23C--)
                              {
                                for (int n33C = u4C.n34(); n33C >= u4C.n44();
                                     n33C--)
                                {
                                  // u3::irrep u3C(n13C, n23C, n33C);

                                  // auto [existAB_C, accAB_C] =
                                  // coeffsAB_C.access(u3AB, u3C, u3ABC);

                                  auto [existAB_C, accAB_C] = wigU4U3db.get(
                                      u4AB.n14(), u4AB.n24(), u4AB.n34(),
                                      u4AB.n44(), n13AB, n23AB, n33AB,
                                      u4C.n14(), u4C.n24(), u4C.n34(),
                                      u4C.n44(), n13C, n23C, n33C, u4ABC.n14(),
                                      u4ABC.n24(), u4ABC.n34(), u4ABC.n44(),
                                      u4ABC.n14(), u4ABC.n24(), u4ABC.n34());

                                  // auto [existBC, accBC] =
                                  // coeffsBC.access(u3B, u3C, u3BC);

                                  auto [existBC, accBC] = wigU4U3db.get(
                                      u4B.n14(), u4B.n24(), u4B.n34(),
                                      u4B.n44(), n13B, n23B, n33B, u4C.n14(),
                                      u4C.n24(), u4C.n34(), u4C.n44(), n13C,
                                      n23C, n33C, u4BC.n14(), u4BC.n24(),
                                      u4BC.n34(), u4BC.n44(), u4BC.n14(),
                                      u4BC.n24(), u4BC.n34());

                                  if (existAB_C == true && existBC == true)
                                  {
                                    // std::cout << u3AB << ";" << u3C << "|||"
                                    // << u3ABC
                                    //           << " -> accAB_C = " << accAB_C
                                    //           << std::endl;
                                    // std::cout << u3B << ";" << u3C << "|||"
                                    // << u3BC
                                    //           << " -> accBC = " << accBC <<
                                    //           std::endl;
                                    u3::irrep u3c(n13C - n33C, n23C - n33C, 0);
                                    // u3c.normalize();
                                    std::vector<u3::gelfand> gArr3c =
                                        u3::gelfandBasis(u3c);

                                    std::vector<double> U6_coeffs = u3::U6(
                                        u3a, gArr3a, u3b, gArr3b, u3abc,
                                        gArr3abc, u3c, gArr3c, u3ab, gArr3ab,
                                        accAB.rows(), accAB_C.rows(), u3bc,
                                        gArr3bc, accBC.rows(), accA_BC.rows(),
                                        cgcs, racahs, wigU3U2db);

                                    for (int eAB = 0; eAB < etaABmax; eAB++)
                                      for (int eAB_C = 0; eAB_C < etaAB_Cmax;
                                           eAB_C++)
                                        for (int eBC = 0; eBC < etaBCmax; eBC++)
                                          for (int rAB = 0; rAB < accAB.rows();
                                               rAB++)
                                            for (int rAB_C = 0;
                                                 rAB_C < accAB_C.rows();
                                                 rAB_C++)
                                              for (int rBC = 0;
                                                   rBC < accBC.rows(); rBC++)
                                                B(r,
                                                  (eAB * etaAB_Cmax + eAB_C) *
                                                          etaBCmax +
                                                      eBC) +=
                                                    accAB(rAB, eAB) *
                                                    accAB_C(rAB_C, eAB_C) *
                                                    accBC(rBC, eBC) *
                                                    U6_coeffs
                                                        [((rAB *
                                                               accAB_C.rows() +
                                                           rAB_C) *
                                                              accBC.rows() +
                                                          rBC) *
                                                             accA_BC.rows() +
                                                         rA_BC];
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

              if (r == etaA_BCmax)
              {
                done = true;
                break;
              }
            }
          }
        }
      }
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
     int etaABmax, int etaAB_Cmax, const u4::irrep& u4AC,
     const std::vector<u4::gelfand>& gArrAC, int etaACmax, int etaAC_Bmax,
     qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah& racahs,
     u3::WigU3U2_db& wigU3U2db, u4::WigU4U3_db& wigU4U3db)
  {
    /*
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
    */
    Eigen::MatrixXd A(etaAC_Bmax, etaAC_Bmax);
    Eigen::MatrixXd B(etaAC_Bmax, etaABmax * etaAB_Cmax * etaACmax);
    B.setZero();

    int r = 0;
    bool done = false;
    for (int n13B = u4B.n14(); n13B >= u4B.n24() && !done; n13B--)
    {
      for (int n23B = u4B.n24(); n23B >= u4B.n34() && !done; n23B--)
      {
        for (int n33B = u4B.n34(); n33B >= u4B.n44() && !done; n33B--)
        {
          /* u3::irrep u3B(n13B, n23B, n33B),
                    u3AC(u4AC.n14(), u4AC.n24(), u4AC.n34()),
                    u3ABC(u4ABC.n14(), u4ABC.n24(), u4ABC.n34());
          auto [existAC_B, accAC_B] = coeffsAC_B.access(u3AC,u3B,u3ABC); */

          auto [existAC_B, accAC_B] = wigU4U3db.get(
              u4AC.n14(), u4AC.n24(), u4AC.n34(), u4AC.n44(), u4AC.n14(),
              u4AC.n24(), u4AC.n34(), u4B.n14(), u4B.n24(), u4B.n34(),
              u4B.n44(), n13B, n23B, n33B, u4ABC.n14(), u4ABC.n24(),
              u4ABC.n34(), u4ABC.n44(), u4ABC.n14(), u4ABC.n24(), u4ABC.n34());

          if (existAC_B == true)
          {
            // std::cout << "accAC_B = " << accAC_B << std::endl;
            u3::irrep u3b(n13B - n33B, n23B - n33B, 0),
                u3ac(u4AC.n14() - u4AC.n34(), u4AC.n24() - u4AC.n34(), 0),
                u3abc(u4ABC.n14() - u4ABC.n34(), u4ABC.n24() - u4ABC.n34(), 0);
            // u3b.normalize();
            // u3ac.normalize();
            // u3abc.normalize();

            std::vector<u3::gelfand> gArr3b = u3::gelfandBasis(u3b),
                                     gArr3ac = u3::gelfandBasis(u3ac),
                                     gArr3abc = u3::gelfandBasis(u3abc);

            for (int rAC_B = 0; rAC_B < accAC_B.rows(); rAC_B++)
            {
              A.row(r) = accAC_B.row(rAC_B);

              for (int n13A = u4A.n14(); n13A >= u4A.n24(); n13A--)
              {
                for (int n23A = u4A.n24(); n23A >= u4A.n34(); n23A--)
                {
                  for (int n33A = u4A.n34(); n33A >= u4A.n44(); n33A--)
                  {
                    // u3::irrep u3A(n13A, n23A, n33A);

                    for (int n13AB = u4AB.n14(); n13AB >= u4AB.n24(); n13AB--)
                    {
                      for (int n23AB = u4AB.n24(); n23AB >= u4AB.n34(); n23AB--)
                      {
                        for (int n33AB = u4AB.n34(); n33AB >= u4AB.n44();
                             n33AB--)
                        {
                          // u3::irrep u3AB(n13AB, n23AB, n33AB);
                          // auto [existAB, accAB] =
                          // coeffsAB.access(u3A,u3B,u3AB);

                          auto [existAB, accAB] = wigU4U3db.get(
                              u4A.n14(), u4A.n24(), u4A.n34(), u4A.n44(), n13A,
                              n23A, n33A, u4B.n14(), u4B.n24(), u4B.n34(),
                              u4B.n44(), n13B, n23B, n33B, u4AB.n14(),
                              u4AB.n24(), u4AB.n34(), u4AB.n44(), n13AB, n23AB,
                              n33AB);

                          if (existAB == true)
                          {
                            // std::cout << "accAB = " << accAB << std::endl;
                            u3::irrep u3a(n13A - n33A, n23A - n33A, 0),
                                u3ab(n13AB - n33AB, n23AB - n33AB, 0);
                            // u3a.normalize();
                            // u3ab.normalize();

                            std::vector<u3::gelfand> gArr3a =
                                                         u3::gelfandBasis(u3a),
                                                     gArr3ab =
                                                         u3::gelfandBasis(u3ab);

                            for (int n13C = u4C.n14(); n13C >= u4C.n24();
                                 n13C--)
                            {
                              for (int n23C = u4C.n24(); n23C >= u4C.n34();
                                   n23C--)
                              {
                                for (int n33C = u4C.n34(); n33C >= u4C.n44();
                                     n33C--)
                                {
                                  // u3::irrep u3C(n13C, n23C, n33C);

                                  // auto [existAB_C, accAB_C] =
                                  // coeffsAB_C.access(u3AB, u3C, u3ABC);

                                  auto [existAB_C, accAB_C] = wigU4U3db.get(
                                      u4AB.n14(), u4AB.n24(), u4AB.n34(),
                                      u4AB.n44(), n13AB, n23AB, n33AB,
                                      u4C.n14(), u4C.n24(), u4C.n34(),
                                      u4C.n44(), n13C, n23C, n33C, u4ABC.n14(),
                                      u4ABC.n24(), u4ABC.n34(), u4ABC.n44(),
                                      u4ABC.n14(), u4ABC.n24(), u4ABC.n34());

                                  // auto [existAC, accAC] =
                                  // coeffsAC.access(u3A, u3C, u3AC);

                                  auto [existAC, accAC] = wigU4U3db.get(
                                      u4A.n14(), u4A.n24(), u4A.n34(),
                                      u4A.n44(), n13A, n23A, n33A, u4C.n14(),
                                      u4C.n24(), u4C.n34(), u4C.n44(), n13C,
                                      n23C, n33C, u4AC.n14(), u4AC.n24(),
                                      u4AC.n34(), u4AC.n44(), u4AC.n14(),
                                      u4AC.n24(), u4AC.n34());

                                  if (existAB_C == true && existAC == true)
                                  {
                                    // std::cout << u3AB << ";" << u3C << "|||"
                                    // << u3ABC
                                    //<< " -> accAB_C = " << accAB_C <<
                                    //std::endl;
                                    // std::cout << u3A << ";" << u3C << "|||"
                                    // << u3AC
                                    //<< " -> accAC = " << accAC << std::endl;
                                    u3::irrep u3c(n13C - n33C, n23C - n33C, 0);
                                    // u3c.normalize();
                                    std::vector<u3::gelfand> gArr3c =
                                        u3::gelfandBasis(u3c);

                                    std::vector<double> Z6_coeffs = u3::Z6(
                                        u3b, gArr3b, u3a, gArr3a, u3abc,
                                        gArr3abc, u3c, gArr3c, u3ab, gArr3ab,
                                        accAB.rows(), accAB_C.rows(), u3ac,
                                        gArr3ac, accAC.rows(), accAC_B.rows(),
                                        cgcs, racahs, wigU3U2db);

                                    for (int eAB = 0; eAB < etaABmax; eAB++)
                                      for (int eAB_C = 0; eAB_C < etaAB_Cmax;
                                           eAB_C++)
                                        for (int eAC = 0; eAC < etaACmax; eAC++)
                                          for (int rAB = 0; rAB < accAB.rows();
                                               rAB++)
                                            for (int rAB_C = 0;
                                                 rAB_C < accAB_C.rows();
                                                 rAB_C++)
                                              for (int rAC = 0;
                                                   rAC < accAC.rows(); rAC++)
                                                B(r,
                                                  (eAB * etaAB_Cmax + eAB_C) *
                                                          etaACmax +
                                                      eAC) +=
                                                    accAB(rAB, eAB) *
                                                    accAB_C(rAB_C, eAB_C) *
                                                    accAC(rAC, eAC) *
                                                    Z6_coeffs
                                                        [((rAB *
                                                               accAB_C.rows() +
                                                           rAB_C) *
                                                              accAC.rows() +
                                                          rAC) *
                                                             accAC_B.rows() +
                                                         rAC_B];
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

              if (r == etaAC_Bmax)
              {
                done = true;
                break;
              }
            }
          }
        }
      }
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

// protection against multiple header file inclusion
#ifndef U4LIB_U4_U9COEFFS_H
#define U4LIB_U4_U9COEFFS_H

#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "U4decomposition.h"
#include "U4_U6Z6Coeffs.h"
#include "U4basis.h"

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>

namespace u4
{

  // input by users: u4a, u4b, u4c, u4d, u4ab, u4cd, u4ac, u4bd, u4abcd
  // all normalized
  inline std::vector<double>
  U9(const u4::irrep& u4A, const std::vector<u4::gelfand>& gArrA,
     const u4::irrep& u4B, const std::vector<u4::gelfand>& gArrB,
     const u4::irrep& u4C, const std::vector<u4::gelfand>& gArrC,
     const u4::irrep& u4D, const std::vector<u4::gelfand>& gArrD,
     const u4::irrep& u4AB, const std::vector<u4::gelfand>& gArrAB,
     const u4::irrep& u4CD, const std::vector<u4::gelfand>& gArrCD,
     const u4::irrep& u4AC, const std::vector<u4::gelfand>& gArrAC,
     const u4::irrep& u4BD, const std::vector<u4::gelfand>& gArrBD,
     const u4::irrep& u4ABCD, const std::vector<u4::gelfand>& gArrABCD,
     int etaABmax, int etaCDmax, int etaAB_CDmax, int etaACmax, int etaBDmax,
     int etaAC_BDmax, qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah& racahs,
     u3::WigU3U2_db& wigU3U2db, u4::WigU4U3_db& wigU4U3db)
  {
    std::vector<double> U(etaACmax * etaBDmax * etaAB_CDmax * etaABmax *
                              etaCDmax * etaAC_BDmax,
                          0.0);

    // couple u4AB and u4C
    auto [u4AB_C, etamaxsAB_C] = u4::decomposition(u4AB, u4C);

    /* an array to store the couple I-II-III irreps
    std::vector<u4::irrep> u4ABC;
    std::vector<int> etaAB_Cmax, etaAC_Bmax;    

    for (int i = 0; i < u4AB_C.size(); i++)
    {
      int etamaxAC_B = u4::Eta(u4AC, u4B, u4AB_C[i]);
      if (etamaxAC_B > 0)
      {
        u4ABC.push_back(u4AB_C[i]);
        etaAB_Cmax.push_back(etamaxsAB_C[i]);
        etaAC_Bmax.push_back(etamaxAC_B);
      }
    } */

    //for (int i = 0; i < u4ABC.size(); i++)
    for (int i = 0; i < u4AB_C.size(); i++)
    {
      int etamaxAC_B = u4::Eta(u4AC, u4B, u4AB_C[i]);
      if (etamaxAC_B == 0)
        continue;

      int etaABC_Dmax = u4::Eta(u4AB_C[i], u4D, u4ABCD);
      if (etaABC_Dmax == 0)
        continue;

      std::vector<u4::gelfand> gArrABC = u4::gelfandBasis(u4AB_C[i]);

      std::vector<double> U6_1 =
          u4::U6(u4AC, gArrAC, u4B, gArrB, u4ABCD, gArrABCD, u4D, gArrD,
                 u4AB_C[i], gArrABC, etamaxAC_B, etaABC_Dmax, u4BD, gArrBD,
                 etaBDmax, etaAC_BDmax, cgcs, racahs, wigU3U2db, wigU4U3db);

      std::vector<double> U6_2 =
          u4::U6(u4AB, gArrAB, u4C, gArrC, u4ABCD, gArrABCD, u4D, gArrD,
                 u4AB_C[i], gArrABC, etamaxsAB_C[i], etaABC_Dmax, u4CD, gArrCD,
                 etaCDmax, etaAB_CDmax, cgcs, racahs, wigU3U2db, wigU4U3db);

      std::vector<double> Z6 =
          u4::Z6(u4B, gArrB, u4A, gArrA, u4AB_C[i], gArrABC, u4C, gArrC, u4AB,
                 gArrAB, etaABmax, etamaxsAB_C[i], u4AC, gArrAC, etaACmax,
                 etamaxAC_B, cgcs, racahs, wigU3U2db, wigU4U3db);

      for (int eAC = 0; eAC < etaACmax; eAC++)
      {
        for (int eBD = 0; eBD < etaBDmax; eBD++)
        {
          for (int eAB_CD = 0; eAB_CD < etaAB_CDmax; eAB_CD++)
          {
            for (int eAB = 0; eAB < etaABmax; eAB++)
            {
              for (int eCD = 0; eCD < etaCDmax; eCD++)
              {
                for (int eAC_BD = 0; eAC_BD < etaAC_BDmax; eAC_BD++)
                {
                  for (int eAC_B = 0; eAC_B < etamaxAC_B; eAC_B++)
                  {
                    for (int eAB_C = 0; eAB_C < etamaxsAB_C[i]; eAB_C++)
                    {
                      for (int eABC_D = 0; eABC_D < etaABC_Dmax; eABC_D++)
                      {
                        U[((((eAC * etaBDmax + eBD) * etaAB_CDmax + eAB_CD) *
                        etaABmax + eAB) * etaCDmax + eCD) * etaAC_BDmax + eAC_BD] 
                        += U6_1[((eAC_B * etaABC_Dmax + eABC_D) * etaBDmax +
                                  eBD) * etaAC_BDmax + eAC_BD] 
                         * Z6[((eAB * etamaxsAB_C[i] + eAB_C) * etaACmax +
                                eAC) * etamaxAC_B + eAC_B] 
                         * U6_2[((eAB_C * etaABC_Dmax + eABC_D) * etaCDmax +
                                  eCD) * etaAB_CDmax + eAB_CD];
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

    for (int i = 0; i < U.size(); i++)
      if (std::fabs(U[i]) < 1e-14)
        U[i] = 0.0;

    return U;
  }

} // namespace u4

#endif

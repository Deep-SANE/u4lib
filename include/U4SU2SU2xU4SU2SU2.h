#ifndef U4LIB_U4SU2SU2XU4SU2SU2_H
#define U4LIB_U4SU2SU2XU4SU2SU2_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "U4basis.h"
#include "U4SU2SU2_to_U4U3U2U1.h"
#include "U4U3xU4U3.h"

#include <so3Coeffs/SO3_CGC.h>

namespace u4
{
  // generate all Wigner U(4)->SU(2)xSU(2) coefficients for multiple
  // values of ss1, tt1, ss2, tt2, ss3, tt3
  inline auto
  WigU4SU2SU2(int etamax, 
              const std::vector<u4::physicalvector>& pArrA,
              const std::vector<u4::physicalvector>& pArrB,
              const std::vector<u4::physicalvector>& pArrC,
              qcx_utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
              u4::WigU4U3_db& wigU4U3db)
  {
    // build Gelfand basis
    std::vector<u4::gelfand> gArrA = u4::gelfandBasis(pArrA[0].irr),
                             gArrB = u4::gelfandBasis(pArrB[0].irr),
                             gArrC = u4::gelfandBasis(pArrC[0].irr);

    // find transformation for physical states belong to the 1st irrep
    std::vector<std::vector<int>> canoBasisA(pArrA.size());
    std::vector<Eigen::MatrixXd> transformationA(pArrA.size());
    for (int i = 0; i < pArrA.size(); i++)
      std::tie(canoBasisA[i], transformationA[i]) =
        u4::phys_to_cano(pArrA[i], pArrA[i].SS, pArrA[i].TT, gArrA);

    // find transformation for physical states belong to the 3rd irrep
    std::vector<std::vector<int>> canoBasisC(pArrC.size());
    std::vector<Eigen::MatrixXd> transformationC(pArrC.size());
    for (int i = 0; i < pArrC.size(); i++)
      std::tie(canoBasisC[i], transformationC[i]) =
        u4::phys_to_cano(pArrC[i], pArrC[i].SS, pArrC[i].TT, gArrC);

    std::vector<std::tuple<int, int, int>> tArr;
    std::vector<Eigen::MatrixXd> coeffs;

    int i, j, k;
    for (j = 0; j < pArrB.size(); j++)
    {
      const auto& pB = pArrB[j];

      auto [canoBasisB, transformationB] = u4::phys_to_cano_allM(pB, gArrB);    
 
      for (i = 0; i < pArrA.size(); i++)
      {
        const auto& pA = pArrA[i];
        const auto& canoA = canoBasisA[i];
        const auto& transA = transformationA[i];

        for (k = 0; k < pArrC.size(); k++)
        {
          const auto& pC = pArrC[k];
          const auto& canoC = canoBasisC[k];
          const auto& transC = transformationC[k];

          if (std::fabs(pA.SS - pB.SS) <= pC.SS && (pA.SS + pB.SS) >= pC.SS && 
              std::fabs(pA.TT - pB.TT) <= pC.TT && (pA.TT + pB.TT) >= pC.TT)
          {
            tArr.emplace_back(i,j,k);

            Eigen::MatrixXd coeffs_temp(pA.zeta * pB.zeta * pC.zeta, etamax);
            coeffs_temp.setZero();

            int mmsB = pC.SS - pA.SS, mmtB = pC.TT - pA.TT;
            int indexS = (pB.SS - mmsB) / 2, indexT = (pB.TT - mmtB) / 2;
            const auto& canoB = canoBasisB[indexS][indexT];
            const auto& transB = transformationB[indexS][indexT];

            double cgcS = cgcs.compute(pA.SS, pB.SS, pC.SS, pA.SS, mmsB, pC.SS),
                   cgcT = cgcs.compute(pA.TT, pB.TT, pC.TT, pA.TT, mmtB, pC.TT);

            for (int iA = 0; iA < canoA.size(); iA++)
            {
              const auto& gA = gArrA[canoA[iA]];

              for (int iB = 0; iB < canoB.size(); iB++)
              {
                const auto& gB = gArrB[canoB[iB]];

                for (int iC = 0; iC < canoC.size(); iC++)
                {
                  const auto& gC = gArrC[canoC[iC]];

                  auto [allowU4, accU4] = wigU4U3db.get(
                    gA.n14(), gA.n24(), gA.n34(), gA.n44(), 
                    gA.n13(), gA.n23(), gA.n33(),
                    gB.n14(), gB.n24(), gB.n34(), gB.n44(),
                    gB.n13(), gB.n23(), gB.n33(),
                    gC.n14(), gC.n24(), gC.n34(), gC.n44(),
                    gC.n13(), gC.n23(), gC.n33());
                  if (allowU4 == false) continue;

                  int jj1 = gA.Lambdax2(), mm1 = gA.MLambdax2(),
                      jj2 = gB.Lambdax2(), mm2 = gB.MLambdax2(),
                      jj3 = gC.Lambdax2(), mm3 = gC.MLambdax2();

                  if ((mm1 + mm2) == mm3 && 
                       std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
                  {
                    double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

                    auto [allowU3, accU3] = wigU3U2db.get(
                      gA.n13(), gA.n23(), gA.n33(), gA.n12(), gA.n22(), 
                      gB.n13(), gB.n23(), gB.n33(), gB.n12(), gB.n22(),
                      gC.n13(), gC.n23(), gC.n33(), gC.n12(), gC.n22());
                    if (allowU3 == false) continue;

                    for (int e = 0; e < etamax; e++)   
                      for (int z1 = 0; z1 < pA.zeta; z1++)
                        for (int z2 = 0; z2 < pB.zeta; z2++)
                          for (int z3 = 0; z3 < pC.zeta; z3++)
                          {
                            double y = 0.0;
                            for (int r = 0; r < accU4.rows(); r++)
                              y += accU4(r,e) * accU3(r);
                            
                            coeffs_temp((z1 * pB.zeta + z2) * pC.zeta + z3, e)
                              += y * cgc * transA(iA, z1)
                               * transB(iB, z2) * transC(iC, z3);
                          }
                  }
                }
              }
            }

            coeffs_temp /= (cgcS * cgcT);
            for (int row = 0; row < coeffs_temp.rows(); row++)
              for (int col = 0; col < coeffs_temp.cols(); col++)
                if (std::fabs(coeffs_temp(row,col)) <= 1e-12)
                  coeffs_temp(row,col) = 0.0;

            coeffs.push_back(coeffs_temp);
          }
        }
      }
    }
    return std::make_pair(tArr, coeffs);
  }

#define U4LIB_ENABLE_WIGU4SU2SU2_CACHE
  class WigU4SU2SU2_db
  {
#ifdef U4LIB_ENABLE_WIGU4SU2SU2_CACHE
    using key_type = std::array<int, 15>;
    std::map<key_type, Eigen::MatrixXd> map_;
#endif

    qcx_utils::SO3_CGC& cgcs_;
    u3::WigU3U2_db& wigU3U2db_;
    u4::WigU4U3_db& wigU4U3db_;

    auto calculate(int etamax, const u4::physicalvector& pA,
      const u4::physicalvector& pB, const u4::physicalvector& pC)
    {
      // build Gelfand basis
      std::vector<u4::gelfand> gArrA = u4::gelfandBasis(pA.irr),
                               gArrB = u4::gelfandBasis(pB.irr),
                               gArrC = u4::gelfandBasis(pC.irr);

      // find transformation for physical states belong to the 1st irrep
      auto [canoA, transA] = u4::phys_to_cano(pA, pA.SS, pA.TT, gArrA);

      // find transformation for physical states belong to the 3rd irrep
      auto [canoC, transC] = u4::phys_to_cano(pC, pC.SS, pC.TT, gArrC);

      // find transformation for physical states belong to the 2nd irrep
      int mmsB = pC.SS - pA.SS, mmtB = pC.TT - pA.TT;
      auto [canoB, transB] = u4::phys_to_cano(pB, mmsB, mmtB, gArrB);

      double cgcS = cgcs_.compute(pA.SS, pB.SS, pC.SS, pA.SS, mmsB, pC.SS),
             cgcT = cgcs_.compute(pA.TT, pB.TT, pC.TT, pA.TT, mmtB, pC.TT);

      Eigen::MatrixXd coeffs(pA.zeta * pB.zeta * pC.zeta, etamax);
      coeffs.setZero();

      for (int iA = 0; iA < canoA.size(); iA++)
      {
        const auto& gA = gArrA[canoA[iA]];

        for (int iB = 0; iB < canoB.size(); iB++)
        {
          const auto& gB = gArrB[canoB[iB]];

          for (int iC = 0; iC < canoC.size(); iC++)
          {
            const auto& gC = gArrC[canoC[iC]];

            auto [allowU4, accU4] = wigU4U3db_.get(
              gA.n14(), gA.n24(), gA.n34(), gA.n44(), gA.n13(), gA.n23(), gA.n33(),
              gB.n14(), gB.n24(), gB.n34(), gB.n44(), gB.n13(), gB.n23(), gB.n33(),
              gC.n14(), gC.n24(), gC.n34(), gC.n44(), gC.n13(), gC.n23(), gC.n33());
            if (allowU4 == false) continue;

            int jj1 = gA.Lambdax2(), mm1 = gA.MLambdax2(),
                jj2 = gB.Lambdax2(), mm2 = gB.MLambdax2(),
                jj3 = gC.Lambdax2(), mm3 = gC.MLambdax2();

            if ((mm1 + mm2) == mm3 && 
                 std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
            {
              double cgc = cgcs_.compute(jj1, jj2, jj3, mm1, mm2, mm3);

              auto [allowU3, accU3] = wigU3U2db_.get(
                gA.n13(), gA.n23(), gA.n33(), gA.n12(), gA.n22(), 
                gB.n13(), gB.n23(), gB.n33(), gB.n12(), gB.n22(),
                gC.n13(), gC.n23(), gC.n33(), gC.n12(), gC.n22());
              if (allowU3 == false) continue;

              for (int e = 0; e < etamax; e++)   
                for (int z1 = 0; z1 < pA.zeta; z1++)
                  for (int z2 = 0; z2 < pB.zeta; z2++)
                    for (int z3 = 0; z3 < pC.zeta; z3++)
                    {
                      double y = 0.0;
                      for (int r = 0; r < accU4.rows(); r++)
                        y += accU4(r,e) * accU3(r);
                            
                      coeffs((z1 * pB.zeta + z2) * pC.zeta + z3, e) += y * cgc 
                            * transA(iA, z1) * transB(iB, z2) * transC(iC, z3);
                    }
            }
          }
        }
      }

      coeffs /= (cgcS * cgcT);
      for (int row = 0; row < coeffs.rows(); row++)
        for (int col = 0; col < coeffs.cols(); col++)
          if (std::fabs(coeffs(row,col)) <= 1e-12)
            coeffs(row,col) = 0.0;

      return coeffs;
    }

  public:
    WigU4SU2SU2_db(qcx_utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
                   u4::WigU4U3_db& wigU4U3db)
        : cgcs_(cgcs), wigU3U2db_(wigU3U2db), wigU4U3db_(wigU4U3db)
    { }

#ifdef U4LIB_ENABLE_WIGU4SU2SU2_CACHE
    WigU4SU2SU2_db(const WigU4SU2SU2_db&) = delete;
    WigU4SU2SU2_db& operator=(const WigU4SU2SU2_db&) = delete;
#endif

    auto get(int P1a, int P2a, int P3a, int SSa, int TTa, 
             int P1b, int P2b, int P3b, int SSb, int TTb,
             int P1c, int P2c, int P3c, int SSc, int TTc)
    {
      // normalized irreps:
      u4::irrep u4a(P1a, P2a, P3a), u4b(P1b, P2b, P3b), u4c(P1c, P2c, P3c);
      int etamax = Eta(u4a, u4b, u4c);
      u4::physicalvector pA(u4a,SSa,TTa), pB(u4b,SSb,TTb), pC(u4c,SSc,TTc);

      bool flag = false;
      Eigen::MatrixXd m;

      if (etamax > 0 && pA.zeta > 0 && pB.zeta > 0 && pC.zeta > 0 &&
          std::fabs(SSa - SSb) <= SSc && (SSa + SSb) >= SSc &&
          std::fabs(TTa - TTb) <= TTc && (TTa + TTb) >= TTc)
      {
#ifdef U4LIB_ENABLE_WIGU4SU2SU2_CACHE
        key_type key = {P1a, P2a, P3a, SSa, TTa,
                        P1b, P2b, P3b, SSb, TTb,
                        P1c, P2c, P3c, SSc, TTc};

        auto iter = map_.find(key);
        if (iter == map_.end())
        {
          auto p = map_.try_emplace(key, calculate(etamax, pA, pB, pC));
          assert(p.second);
          iter = p.first;
        }

        m = iter->second;
#else
        m = calculate(etamax, pA, pB, pC);
#endif
        flag = true;
        
      }

      return std::make_pair(flag, m);
    }
  };

} // namespace u4

#endif

/*
  // generate all Wigner U(4)->SU(2)xSU(2) coefficients for multiple
  // values of ss1, tt1, ss2, tt2, ss3, tt3
  inline auto
  WigU4SU2SU2(int etamax, 
              const std::vector<u4::physicalvector>& pArrA,
              const std::vector<u4::physicalvector>& pArrB,
              const std::vector<u4::physicalvector>& pArrC,
              qcx_utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
              u4::WigU4U3_db& wigU4U3db)
  {
    // build Gelfand basis
    std::vector<u4::gelfand> gArrA = u4::gelfandBasis(pArrA[0].irr),
                             gArrB = u4::gelfandBasis(pArrB[0].irr),
                             gArrC = u4::gelfandBasis(pArrC[0].irr);

    // find transformation for physical states belong to the 1st irrep
    std::vector<std::vector<int>> canoBasisA(pArrA.size());
    std::vector<Eigen::MatrixXd> transformationA(pArrA.size());
    for (int i = 0; i < pArrA.size(); i++)
      std::tie(canoBasisA[i], transformationA[i]) =
        u4::phys_to_cano(pArrA[i], pArrA[i].SS, pArrA[i].TT, gArrA);

    // find transformation for physical states belong to the 3rd irrep
    std::vector<std::vector<int>> canoBasisC(pArrC.size());
    std::vector<Eigen::MatrixXd> transformationC(pArrC.size());
    for (int i = 0; i < pArrC.size(); i++)
      std::tie(canoBasisC[i], transformationC[i]) =
        u4::phys_to_cano(pArrC[i], pArrC[i].SS, pArrC[i].TT, gArrC);

    std::vector<std::tuple<int, int, int>> tArr;
    std::vector<Eigen::MatrixXd> coeffs;

    int i, j, k;
    for (j = 0; j < pArrB.size(); j++)
    {
      std::vector<std::vector<std::vector<int>>> canoBasisB(pArrB[j].SS + 1, 
        std::vector<std::vector<int>>(pArrB[j].TT + 1));
      std::vector<std::vector<Eigen::MatrixXd>> transformationB(pArrB[j].SS + 1, 
        std::vector<Eigen::MatrixXd>(pArrB[j].TT + 1));      

      for (int mSx2 = pArrB[j].SS; mSx2 >= -pArrB[j].SS; mSx2 -= 2)
      {
        int indexS = (pArrB[j].SS - mSx2)/2;

        if (mSx2 == pArrB[j].SS)
        {
          std::tie(canoBasisB[0][0], transformationB[0][0]) = 
            u4::phys_to_cano(pArrB[j], pArrB[j].SS, pArrB[j].TT, gArrB);
        }
              
        else if (mSx2 < pArrB[j].SS)
        {
          canoBasisB[indexS][0] =
            u4::expand_to_gelfand(mSx2, pArrB[j].TT, gArrB);

          transformationB[indexS][0] =
            u4::lowerMS(pArrB[j].zeta, pArrB[j].SS, mSx2, gArrB,
            transformationB[indexS-1][0], canoBasisB[indexS-1][0], 
            canoBasisB[indexS][0]);
        }

        for (int mTx2 = pArrB[j].TT - 2; mTx2 >= -pArrB[j].TT; mTx2 -= 2)
        {
          int indexT = (pArrB[j].TT - mTx2)/2;

          canoBasisB[indexS][indexT] =
            u4::expand_to_gelfand(mSx2, mTx2, gArrB);

          transformationB[indexS][indexT] =
            u4::lowerMT(pArrB[j].zeta, pArrB[j].TT, mTx2, gArrB,
            transformationB[indexS][indexT-1], canoBasisB[indexS][indexT-1], 
            canoBasisB[indexS][indexT]);
        }
      }  

      for (i = 0; i < pArrA.size(); i++)
      {
        for (k = 0; k < pArrC.size(); k++)
        {
          if (std::fabs(pArrA[i].SS - pArrB[j].SS) <= pArrC[k].SS 
              && (pArrA[i].SS + pArrB[j].SS) >= pArrC[k].SS && 
              std::fabs(pArrA[i].TT - pArrB[j].TT) <= pArrC[k].TT 
              && (pArrA[i].TT + pArrB[j].TT) >= pArrC[k].TT)
          {
            tArr.emplace_back(i,j,k);

            Eigen::MatrixXd coeffs_temp(pArrA[i].zeta * pArrB[j].zeta 
                                            * pArrC[k].zeta, etamax);
            coeffs_temp.setZero();

            int MSx2 = pArrC[k].SS - pArrA[i].SS,
                MTx2 = pArrC[k].TT - pArrA[i].TT;

            double cgcS = cgcs.compute(pArrA[i].SS, pArrB[j].SS, pArrC[k].SS,
                                       pArrA[i].SS, MSx2, pArrC[k].SS),
                   cgcT = cgcs.compute(pArrA[i].TT, pArrB[j].TT, pArrC[k].TT,
                                       pArrA[i].TT, MTx2, pArrC[k].TT);
            
            int indexS = (pArrB[j].SS - MSx2)/2,
                indexT = (pArrB[j].TT - MTx2)/2;

            for (int iA = 0; iA < canoBasisA[i].size(); iA++)
            {
              int gA = canoBasisA[i][iA];
              for (int iB = 0; iB < canoBasisB[indexS][indexT].size(); iB++)
              {
                int gB = canoBasisB[indexS][indexT][iB];
                for (int iC = 0; iC < canoBasisC[k].size(); iC++)
                {
                  int gC = canoBasisC[k][iC];

                  //u3::irrep u3A(gArrA[gA].n13(), gArrA[gA].n23(), gArrA[gA].n33()),
                  //          u3B(gArrB[gB].n13(), gArrB[gB].n23(), gArrB[gB].n33()),
                  //          u3C(gArrC[gC].n13(), gArrC[gC].n23(), gArrC[gC].n33());
                  //auto [exist, accU4] = canoCoeffs.access(u3A,u3B,u3C);

                  auto [exist, accU4] = wigU4U3db.get(
                    pArrA[0].irr.n14(), pArrA[0].irr.n24(), 
                    pArrA[0].irr.n34(), pArrA[0].irr.n44(),
                    gArrA[gA].n13(), gArrA[gA].n23(), gArrA[gA].n33(),
                    pArrB[0].irr.n14(), pArrB[0].irr.n24(), 
                    pArrB[0].irr.n34(), pArrB[0].irr.n44(),
                    gArrB[gB].n13(), gArrB[gB].n23(), gArrB[gB].n33(),
                    pArrC[0].irr.n14(), pArrC[0].irr.n24(), 
                    pArrC[0].irr.n34(), pArrC[0].irr.n44(),
                    gArrC[gC].n13(), gArrC[gC].n23(), gArrC[gC].n33());

                  int jj1 = gArrA[gA].Lambdax2(), mm1 = gArrA[gA].MLambdax2(),
                      jj2 = gArrB[gB].Lambdax2(), mm2 = gArrB[gB].MLambdax2(),
                      jj3 = gArrC[gC].Lambdax2(), mm3 = gArrC[gC].MLambdax2();

                  if (exist == true && (mm1 + mm2) == mm3 && 
                    std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
                  {
                    double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

                    auto [allowU3, accU3] = wigU3U2db.get(
                      gArrA[gA].n13(), gArrA[gA].n23(), gArrA[gA].n33(), 
                      gArrA[gA].n12(), gArrA[gA].n22(), 
                      gArrB[gB].n13(), gArrB[gB].n23(), gArrB[gB].n33(),                          
                      gArrB[gB].n12(), gArrB[gB].n22(),
                      gArrC[gC].n13(), gArrC[gC].n23(), gArrC[gC].n33(), 
                      gArrC[gC].n12(), gArrC[gC].n22());
                    if (allowU3 == false)
                      continue;

                    for (int e = 0; e < etamax; e++)   
                      for (int z1 = 0; z1 < pArrA[i].zeta; z1++)
                        for (int z2 = 0; z2 < pArrB[j].zeta; z2++)
                          for (int z3 = 0; z3 < pArrC[k].zeta; z3++)
                          {
                            double y = 0.0;
                            for (int r = 0; r < accU4.rows(); r++)
                              y += accU4(r,e) * accU3(r);
                            
                            coeffs_temp((z1 * pArrB[j].zeta + z2) * pArrC[k].zeta + z3, e)
                             += y * cgc * transformationA[i](iA, z1)
                             * transformationB[indexS][indexT](iB, z2)
                             * transformationC[k](iC, z3);
                          }
                  }
                }
              }
            }

            coeffs_temp /= (cgcS * cgcT);
            for (int row = 0; row < coeffs_temp.rows(); row++)
              for (int col = 0; col < coeffs_temp.cols(); col++)
                if (std::fabs(coeffs_temp(row,col)) <= 1e-15)
                  coeffs_temp(row,col) = 0.0;

            coeffs.push_back(coeffs_temp);
          }
        }
      }
    }
    return std::make_pair(tArr, coeffs);
  }
*/

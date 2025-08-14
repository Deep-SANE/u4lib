#ifndef U4LIB_U4SU2SU2XU4SU2SU2_H
#define U4LIB_U4SU2SU2XU4SU2SU2_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>
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
  inline std::vector<Eigen::MatrixXd>
  WigU4SU2SU2(int& etamax, std::vector<u4::physicalvector>& pArrA,
              std::vector<u4::physicalvector>& pArrB,
              std::vector<u4::physicalvector>& pArrC,
              std::vector<std::tuple<int, int, int>>& tArr,
              qcx_utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db,
              u4::WigU4U3_db& wigU4U3db)
  {
    // build Gelfand basis
    std::vector<u4::gelfand> gArrA = u4::gelfandBasis(pArrA[0].toprow),
                             gArrB = u4::gelfandBasis(pArrB[0].toprow),
                             gArrC = u4::gelfandBasis(pArrC[0].toprow);

    // find transformation for physical states belong to the 1st irrep
    std::vector<std::vector<int>> canoBasisA(pArrA.size());
    std::vector<Eigen::MatrixXd> transformationA(pArrA.size());
    for (int i = 0; i < pArrA.size(); i++)
      transformationA[i] =
        u4::phys_to_cano(pArrA[i], pArrA[i].SS, pArrA[i].TT, canoBasisA[i], gArrA);

    // find transformation for physical states belong to the 3rd irrep
    std::vector<int> w3mins(pArrC.size());
    std::vector<std::vector<int>> canoBasisC(pArrC.size());
    std::vector<Eigen::MatrixXd> transformationC(pArrC.size());
    for (int i = 0; i < pArrC.size(); i++)
    {
      transformationC[i] =
        u4::phys_to_cano(pArrC[i], pArrC[i].SS, pArrC[i].TT, canoBasisC[i], gArrC);

      w3mins[i] = gArrC[canoBasisC[i].back()].n13() 
                + gArrC[canoBasisC[i].back()].n23()
                + gArrC[canoBasisC[i].back()].n33();
    }

    // find all the reduced canonical coupling coefficients
    //int w3min = -1;//*std::min_element(w3mins.begin(), w3mins.end());
    //auto canoCoeffs =
    //    u4::WigU4U3(pArrA[0].toprow, pArrB[0].toprow, pArrC[0].toprow, 
    //                etamax, w3min, gArrA, gArrB, gArrC, cgcs, wigU3U2db);
    // note that u4C and all gelfand states are not normalized here

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
          transformationB[0][0] = 
            u4::phys_to_cano(pArrB[j], pArrB[j].SS, 
            pArrB[j].TT, canoBasisB[0][0], gArrB);
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
                    pArrA[0].toprow.n14(), pArrA[0].toprow.n24(), 
                    pArrA[0].toprow.n34(), pArrA[0].toprow.n44(),
                    gArrA[gA].n13(), gArrA[gA].n23(), gArrA[gA].n33(),
                    pArrB[0].toprow.n14(), pArrB[0].toprow.n24(), 
                    pArrB[0].toprow.n34(), pArrB[0].toprow.n44(),
                    gArrB[gB].n13(), gArrB[gB].n23(), gArrB[gB].n33(),
                    pArrC[0].toprow.n14(), pArrC[0].toprow.n24(), 
                    pArrC[0].toprow.n34(), pArrC[0].toprow.n44(),
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
    return coeffs;
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

    auto calculate(const irrep& u4a, int SSa, int TTa, 
                   const irrep& u4b, int SSb, int TTb, 
                   const irrep& u4c, int SSc, int TTc,
                   int etamax)
    {
      std::vector<u4::physicalvector> 
        pArrA(1, u4::physicalvector(u4a, SSa, TTa)),
        pArrB(1, u4::physicalvector(u4b, SSb, TTb)), 
        pArrC(1, u4::physicalvector(u4c, SSc, TTc)); 

      std::vector<std::tuple<int, int, int>> tArr;

      /* return one coefficient matrix
        row = (za * max(zb) + zb) * max(zc) + zc
        column = eta
      */
      return WigU4SU2SU2(etamax, pArrA, pArrB, pArrC, tArr, 
        cgcs_, wigU3U2db_, wigU4U3db_)[0];
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
      irrep u4a(P1a, P2a, P3a);
      irrep u4b(P1b, P2b, P3b);
      irrep u4c(P1c, P2c, P3c);
      int etamax = Eta(u4a, u4b, u4c);

      bool flag = false;
      Eigen::MatrixXd m;

      if (etamax > 0 && 
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
          auto p = map_.try_emplace(
            key, calculate(u4a, SSa, TTa, u4b, SSb, TTb, u4c, SSc, TTc, etamax));
          assert(p.second);
          iter = p.first;
        }

        m = iter->second;
#else
        m = calculate(u4a, SSa, TTa, u4b, SSb, TTb, u4c, SSc, TTc, etamax);
#endif
        flag = true;
        
      }

      return std::make_pair(flag, m);
    }
  };

} // namespace u4

#endif

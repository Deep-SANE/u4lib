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
  inline std::vector<Eigen::MatrixXd>  WigU4SU2SU2(int& etamax, 
    std::vector<u4::physicalvector>& pArrA,
    std::vector<u4::physicalvector>& pArrB,
    std::vector<u4::physicalvector>& pArrC,
    std::vector<std::tuple<int, int, int>>& tArr,
    u34utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db)
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
    int w3min = -1;//*std::min_element(w3mins.begin(), w3mins.end());
    auto canoCoeffs =
        u4::WigU4U3(pArrA[0].toprow, pArrB[0].toprow, pArrC[0].toprow, 
                    etamax, w3min, gArrA, gArrB, gArrC, cgcs, wigU3U2db);
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

                  u3::irrep u3A(gArrA[gA].n13(), gArrA[gA].n23(), gArrA[gA].n33()),
                            u3B(gArrB[gB].n13(), gArrB[gB].n23(), gArrB[gB].n33()),
                            u3C(gArrC[gC].n13(), gArrC[gC].n23(), gArrC[gC].n33());
                  int rhomax = u3::Rho(u3A, u3B, u3C);

                  int jj1 = gArrA[gA].Lambdax2(), mm1 = gArrA[gA].MLambdax2(),
                      jj2 = gArrB[gB].Lambdax2(), mm2 = gArrB[gB].MLambdax2(),
                      jj3 = gArrC[gC].Lambdax2(), mm3 = gArrC[gC].MLambdax2();

                  if (rhomax > 0 && (mm1 + mm2) == mm3 && 
                    std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
                  {
                    double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

                    auto [missing, accU4] = canoCoeffs.access(u3A,u3B,u3C);
                    assert(missing == false);

                    int x = u3A.n33() + u3B.n33();
                    u3C.n13(u3C.n13() - x);
                    u3C.n23(u3C.n23() - x);
                    u3C.n33(u3C.n33() - x);
                    u3A.normalize();
                    u3B.normalize();

                    std::vector<u3::gelfand> gArr3a = u3::gelfandBasis(u3A),
                                             gArr3b = u3::gelfandBasis(u3B),
                                             gArr3c = u3::gelfandBasis(u3C);

                    auto wigU3U2 = wigU3U2db.get(u3A, u3B, u3C, rhomax,
                                              gArr3a, gArr3b, gArr3c, cgcs);
                    // after this, u3c, gArr3c and wigU3U2 are normalized

                    auto accU3 = wigU3U2.access(gArrA[gA].n12() - gArrA[gA].n33(), 
                                                gArrA[gA].n22() - gArrA[gA].n33(), 
                                                gArrB[gB].n12() - gArrB[gB].n33(), 
                                                gArrB[gB].n22() - gArrB[gB].n33(), 
                                                gArrC[gC].n12() - gArrC[gC].n33(), 
                                                gArrC[gC].n22() - gArrC[gC].n33());

                    for (int e = 0; e < etamax; e++)   
                      for (int z1 = 0; z1 < pArrA[i].zeta; z1++)
                        for (int z2 = 0; z2 < pArrB[j].zeta; z2++)
                          for (int z3 = 0; z3 < pArrC[k].zeta; z3++)
                          {
                            double y = 0.0;
                            for (int r = 0; r < rhomax; r++)
                              y += accU4(r,e) * wigU3U2.get(accU3, r);
                            
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

} // namespace u4

#endif
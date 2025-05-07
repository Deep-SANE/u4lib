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
    int w3min = *std::min_element(w3mins.begin(), w3mins.end());
    auto canoCoeffs =
        u4::WigU4U3(pArrA[0].toprow, pArrB[0].toprow, pArrC[0].toprow, 
                    etamax, w3min, gArrA, gArrB, gArrC, cgcs, wigU3U2db);
    // note that u4C and all gelfand states are already normalized here

    std::vector<Eigen::MatrixXd> coeffs;

    int i, j, k;
    for (j = 0; j < pArrB.size(); j++)
    {
      std::vector<std::vector<int>> canoBasisB;
      std::vector<Eigen::MatrixXd> transformationB;      

      for (int mSx2 = pArrB[j].SS; mSx2 >= -pArrB[j].SS; mSx2 -= 2)
      {
        if (mSx2 == pArrB[j].SS)
        {
          std::vector<int> canoBasisB_new;
          Eigen::MatrixXd transformationB_new = 
            u4::phys_to_cano(pArrB[j], pArrB[j].SS, 
            pArrB[j].TT, canoBasisB_new, gArrB);
          
          canoBasisB.push_back(canoBasisB_new);
          transformationB.push_back(transformationB_new);
        }
              
        else if (mSx2 < pArrB[j].SS)
        {
          // fix mT, track the previous mS state
          int indexS = ((pArrB[j].SS - mSx2)/2 - 1) * (pArrB[j].TT + 1);

          std::vector<int> canoBasisB_new =
            u4::expand_to_gelfand(mSx2, pArrB[j].TT, gArrB);

          Eigen::MatrixXd transformationB_new =
            u4::lowerMS(pArrB[j].zeta, pArrB[j].SS, mSx2, gArrB,
            transformationB[indexS], canoBasisB[indexS], canoBasisB_new);

          canoBasisB.push_back(canoBasisB_new);
          transformationB.push_back(transformationB_new);
        }

        for (int mTx2 = pArrB[j].TT - 2; mTx2 >= -pArrB[j].TT; mTx2 -= 2)
        {
          int indexT = (pArrB[j].SS - mSx2)/2 * (pArrB[j].TT + 1)
                     + (pArrB[j].TT - mTx2)/2 - 1;

          std::vector<int> canoBasisB_new =
            u4::expand_to_gelfand(mSx2, mTx2, gArrB);

          Eigen::MatrixXd transformationB_new =
            u4::lowerMT(pArrB[j].zeta, pArrB[j].TT, mTx2, gArrB,
            transformationB[indexT], canoBasisB[indexT], canoBasisB_new);

          canoBasisB.push_back(canoBasisB_new);
          transformationB.push_back(transformationB_new);
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
            int index = (pArrB[j].SS - MSx2)/2 * (pArrB[j].TT + 1)
                      + (pArrB[j].TT - MTx2)/2;
            for (int iA = 0; iA < canoBasisA[i].size(); iA++)
            {
              int gA = canoBasisA[i][iA];
              for (int iB = 0; iB < canoBasisB[index].size(); iB++)
              {
                int gB = canoBasisB[index][iB];
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
                             * transformationB[index](iB, z2)
                             * transformationC[k](iC, z3);
                          }
                  }
                }
              }
            }

            coeffs_temp /= (cgcS * cgcT);
            coeffs.push_back(coeffs_temp);
          }
        }
      }
    }
    return coeffs;
  }

} // namespace u4

#endif

/*
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  WigU4SU2SU2(u4::irrep& u4A, u4::irrep& u4B, u4::irrep& u4C,
              std::vector<u4::physicalvector>& physvecA,
              std::vector<u4::physicalvector>& physvecB,
              std::vector<u4::physicalvector>& physvecC)
  {

    // find all the canonical coupling coefficients
    std::vector<u4::gelfand> gArrA;
    std::vector<u4::gelfand> gArrB;
    std::vector<u4::gelfand> gArrC;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
        canoCoupleCoeffs; // each column corresponds to one eta

    u4::gelfand HWc(u4C.n14, u4C.n24, u4C.n34, u4C.n44, u4C.n14, u4C.n24,
                    u4C.n34, u4C.n14, u4C.n24, u4C.n14);
    u4::HighestWeightCoeffs(u4A, u4B, HWc, gArrA, gArrB, gArrC,
                            canoCoupleCoeffs);

    for (int w4 = u4C.n44; w4 <= u4C.n14; w4++)
    {
      for (int w1 = u4C.n14; w1 >= u4C.n44; w1--)
      {
        for (int w3 = 0; w3 <= u4C.n14 + u4C.n24 + u4C.n34 + u4C.n44 - w1 - w4;
             w3++)
        {
          if (w4 == u4C.n44 && w3 == u4C.n34 && w1 == u4C.n14)
          {
            continue;
          }
          else
          {
            u4::LowerWeightCoeffs(u4A, u4B, u4C, w4, w3, w1, gArrA, gArrB,
                                  gArrC, canoCoupleCoeffs);
          }
        }
      }
    }

    // find expansion for the first irrep
    std::vector<u4::physicalvector> pbA = u4::physicalbasis(u4A);
    std::vector<int> zetaA;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
        allCanoCoeffsA; // each column corresponds to one zeta
    std::vector<std::vector<u4::gelfand>> allCanoBasisA;
    for (int i = 0; i < pbA.size(); i++)
    {

      // Construct the canonical basis associated with the physical basis state
      // that we are looking at
      std::vector<u4::gelfand> canobasis = u4::canonicalbasis(pbA[i]);

      // Building the projection matrix
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> STproj =
          u4::STprojection(pbA[i], canobasis);
      // and solving its null space
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> nullspace =
          u4::phys_to_cano(
              pbA[i],
              STproj); // after solving the nullspace, the dimension of the
                       // nullspace is the inner multiplicity of (S,T)

      zetaA.push_back(pbA[i].zeta); // this multiplicity was computed after
                                    // solving the nullspace
      allCanoCoeffsA.push_back(nullspace);
      allCanoBasisA.push_back(canobasis);
    }

    // find expansion for the third irrep
    std::vector<u4::physicalvector> pbC = u4::physicalbasis(u4C);
    std::vector<int> zetaC;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
        allCanoCoeffsC; // each column corresponds to one kappa
    std::vector<std::vector<u4::gelfand>> allCanoBasisC;
    for (int i = 0; i < pbC.size(); i++)
    {

      // Construct the canonical basis associated with the physical basis state
      // that we are looking at
      std::vector<u4::gelfand> canobasis = u4::canonicalbasis(pbC[i]);

      // Building the projection matrix
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> STproj =
          u4::STprojection(pbC[i], canobasis);
      // and solving its null space
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> nullspace =
          u4::phys_to_cano(
              pbC[i],
              STproj); // after solving the nullspace, the dimension of the
                       // nullspace is the inner multiplicity of (S,T)

      zetaC.push_back(pbC[i].zeta); // this multiplicity was computed after
                                    // solving the nullspace
      allCanoCoeffsC.push_back(nullspace);
      allCanoBasisC.push_back(canobasis);
    }

    // compute Wigner coefficients: < [ni4]1,zeta1,S1,T1 ; [ni4]2,zeta2,S2,T2 ||
    // [ni4]3,zeta3,S3,T3 >
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coeffs;
    int etamax = canoCoupleCoeffs.cols();

    // find expansion for the second irrep
    std::vector<u4::physicalvector> pbB = u4::physicalbasis(u4B);
    for (int i = 0; i < pbB.size(); i++)
    {

      std::vector<
          std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>>
          allCanoCoeffsB(
              pbB[i].SS + 1,
              std::vector<
                  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>(
                  pbB[i].TT + 1)); // each column corresponds to one zeta -- the
                                   // inner multiplicity
      std::vector<std::vector<std::vector<u4::gelfand>>> allCanoBasisB(
          pbB[i].SS + 1,
          std::vector<std::vector<u4::gelfand>>(
              pbB[i].TT +
              1)); // to store all canonical basis states associated with all
                   // physical basis states (S,MS;T,MT) with a given (S,T)

      std::vector<std::vector<double>> alpha_beta_coefficients(
          pbB[i].SS + 1,
          std::vector<double>(pbB[i].TT +
                              1)); // to store sqrt( (2S-alpha)! (2T-beta)! /
                                   // (2S)! / alpha! / (2T)! / beta!)

      // Construct the canonical basis associated with the heighest weight
      // physical basis state (S=MS,T=MT)
      allCanoBasisB[0][0] = u4::canonicalbasis(pbB[i]);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> STproj =
          u4::STprojection(pbB[i], allCanoBasisB[0][0]);
      allCanoCoeffsB[0][0] = u4::phys_to_cano(pbB[i], STproj);

      // now going lower from the HW state, remember that alpha = S-MS, beta =
      // T-MT

      // Construct the canonical basis associated with the physical basis states
      // (S,MS;T=MT)
      for (int alpha = 1; alpha <= pbB[i].SS; alpha++)
      {

        // extract the expansion of the previous state
        std::vector<u4::gelfand> cano_previous = allCanoBasisB[alpha - 1][0];
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> canocoeffs_prev =
            allCanoCoeffsB[alpha - 1][0];

        // define a new array to store new expansion, in which all canonical
        // basis states are obtained by acting with S- = E42+E31
        std::vector<u4::gelfand> cano_now;

        for (int j = 0; j < cano_previous.size(); j++)
        {
          // action of E42
          u4::gelfand g1 = cano_previous[j];
          g1.n13--;
          g1.n12--;
          if (g1.check())
            u4::newstate(g1, cano_now);

          u4::gelfand g2 = cano_previous[j];
          g2.n13--;
          g2.n22--;
          if (g2.check())
            u4::newstate(g2, cano_now);

          u4::gelfand g3 = cano_previous[j];
          g3.n23--;
          g3.n12--;
          if (g3.check())
            u4::newstate(g3, cano_now);

          u4::gelfand g4 = cano_previous[j];
          g4.n23--;
          g4.n22--;
          if (g4.check())
            u4::newstate(g4, cano_now);

          u4::gelfand g5 = cano_previous[j];
          g5.n33--;
          g5.n12--;
          if (g5.check())
            u4::newstate(g5, cano_now);

          u4::gelfand g6 = cano_previous[j];
          g6.n33--;
          g6.n22--;
          if (g6.check())
            u4::newstate(g6, cano_now);

          // action of E31
          u4::gelfand g7 = cano_previous[j];
          g7.n12--;
          g7.n11--;
          if (g7.check())
            u4::newstate(g7, cano_now);

          u4::gelfand g8 = cano_previous[j];
          g8.n22--;
          g8.n11--;
          if (g8.check())
            u4::newstate(g8, cano_now);
        }

        // calculate expansion coefficients for the new state
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> canocoeffs_now(
            cano_now.size(), pbB[i].zeta);
        for (int j = 0; j < canocoeffs_now.rows(); j++)
          for (int k = 0; k < canocoeffs_now.cols(); k++)
            canocoeffs_now(j, k) = 0.0;

        for (int j = 0; j < cano_now.size(); j++)
        {
          for (int m = 0; m < cano_previous.size(); m++)
          {
            // action of E42
            if (cano_now[j].n13 == (cano_previous[m].n13 - 1) &&
                cano_now[j].n23 == cano_previous[m].n23 &&
                cano_now[j].n33 == cano_previous[m].n33 &&
                cano_now[j].n12 == (cano_previous[m].n12 - 1) &&
                cano_now[j].n22 == cano_previous[m].n22 &&
                cano_now[j].n11 == cano_previous[m].n11)
            {
              for (int k = 0; k < pbB[i].zeta; k++)
              {
                canocoeffs_now(j, k) +=
                    canocoeffs_prev(m, k) * cano_previous[m].e42_13_12();
              }
            }
            else if (cano_now[j].n13 == (cano_previous[m].n13 - 1) &&
                     cano_now[j].n23 == cano_previous[m].n23 &&
                     cano_now[j].n33 == cano_previous[m].n33 &&
                     cano_now[j].n12 == cano_previous[m].n12 &&
                     cano_now[j].n22 == (cano_previous[m].n22 - 1) &&
                     cano_now[j].n11 == cano_previous[m].n11)
            {
              for (int k = 0; k < pbB[i].zeta; k++)
              {
                canocoeffs_now(j, k) +=
                    canocoeffs_prev(m, k) * cano_previous[m].e42_13_22();
              }
            }
            else if (cano_now[j].n13 == cano_previous[m].n13 &&
                     cano_now[j].n23 == (cano_previous[m].n23 - 1) &&
                     cano_now[j].n33 == cano_previous[m].n33 &&
                     cano_now[j].n12 == (cano_previous[m].n12 - 1) &&
                     cano_now[j].n22 == cano_previous[m].n22 &&
                     cano_now[j].n11 == cano_previous[m].n11)
            {
              for (int k = 0; k < pbB[i].zeta; k++)
              {
                canocoeffs_now(j, k) +=
                    canocoeffs_prev(m, k) * cano_previous[m].e42_23_12();
              }
            }
            else if (cano_now[j].n13 == cano_previous[m].n13 &&
                     cano_now[j].n23 == (cano_previous[m].n23 - 1) &&
                     cano_now[j].n33 == cano_previous[m].n33 &&
                     cano_now[j].n12 == cano_previous[m].n12 &&
                     cano_now[j].n22 == (cano_previous[m].n22 - 1) &&
                     cano_now[j].n11 == cano_previous[m].n11)
            {
              for (int k = 0; k < pbB[i].zeta; k++)
              {
                canocoeffs_now(j, k) +=
                    canocoeffs_prev(m, k) * cano_previous[m].e42_23_22();
              }
            }
            else if (cano_now[j].n13 == cano_previous[m].n13 &&
                     cano_now[j].n23 == cano_previous[m].n23 &&
                     cano_now[j].n33 == (cano_previous[m].n33 - 1) &&
                     cano_now[j].n12 == (cano_previous[m].n12 - 1) &&
                     cano_now[j].n22 == cano_previous[m].n22 &&
                     cano_now[j].n11 == cano_previous[m].n11)
            {
              for (int k = 0; k < pbB[i].zeta; k++)
              {
                canocoeffs_now(j, k) +=
                    canocoeffs_prev(m, k) * cano_previous[m].e42_33_12();
              }
            }
            else if (cano_now[j].n13 == cano_previous[m].n13 &&
                     cano_now[j].n23 == cano_previous[m].n23 &&
                     cano_now[j].n33 == (cano_previous[m].n33 - 1) &&
                     cano_now[j].n12 == cano_previous[m].n12 &&
                     cano_now[j].n22 == (cano_previous[m].n22 - 1) &&
                     cano_now[j].n11 == cano_previous[m].n11)
            {
              for (int k = 0; k < pbB[i].zeta; k++)
              {
                canocoeffs_now(j, k) +=
                    canocoeffs_prev(m, k) * cano_previous[m].e42_33_22();
              }
            }

            // action of E31
            else if (cano_now[j].n13 == cano_previous[m].n13 &&
                     cano_now[j].n23 == cano_previous[m].n23 &&
                     cano_now[j].n33 == cano_previous[m].n33 &&
                     cano_now[j].n12 == (cano_previous[m].n12 - 1) &&
                     cano_now[j].n22 == cano_previous[m].n22 &&
                     cano_now[j].n11 == (cano_previous[m].n11 - 1))
            {
              for (int k = 0; k < pbA[i].zeta; k++)
              {
                canocoeffs_now(j, k) +=
                    canocoeffs_prev(m, k) * cano_previous[m].e31_12();
              }
            }
            else if (cano_now[j].n13 == cano_previous[m].n13 &&
                     cano_now[j].n23 == cano_previous[m].n23 &&
                     cano_now[j].n33 == cano_previous[m].n33 &&
                     cano_now[j].n12 == cano_previous[m].n12 &&
                     cano_now[j].n22 == (cano_previous[m].n22 - 1) &&
                     cano_now[j].n11 == (cano_previous[m].n11 - 1))
            {
              for (int k = 0; k < pbA[i].zeta; k++)
              {
                canocoeffs_now(j, k) +=
                    canocoeffs_prev(m, k) * cano_previous[m].e31_22();
              }
            }
          }
        }

        allCanoBasisB[alpha][0] = cano_now;
        allCanoCoeffsB[alpha][0] = canocoeffs_now;
      }

      // Construct the canonical basis associated with the physical basis states
      // (S,MS;T>MT)
      for (int alpha = 0; alpha <= pbB[i].SS; alpha++)
      {
        for (int beta = 1; beta <= pbB[i].TT; beta++)
        {

          // extract the expansion of the previous state
          std::vector<u4::gelfand> cano_previous =
              allCanoBasisB[alpha][beta - 1];
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
              canocoeffs_prev = allCanoCoeffsB[alpha][beta - 1];

          // define a new array to store new expansion, in which all canonical
          // basis states are obtained by acting with T- = E43+E21
          std::vector<u4::gelfand> cano_now;

          for (int j = 0; j < cano_previous.size(); j++)
          {
            // determine the action of E43
            u4::gelfand g1 = cano_previous[j];
            g1.n13--;
            if (g1.check())
              u4::newstate(g1, cano_now);

            u4::gelfand g2 = cano_previous[j];
            g2.n23--;
            if (g2.check())
              u4::newstate(g2, cano_now);

            u4::gelfand g3 = cano_previous[j];
            g3.n33--;
            if (g3.check())
              u4::newstate(g3, cano_now);

            // determine the action of E21
            u4::gelfand g4 = cano_previous[j];
            g4.n11--;
            if (g4.check())
              u4::newstate(g4, cano_now);
          }

          // calculate expansion coefficients for the new state
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> canocoeffs_now(
              cano_now.size(), pbB[i].zeta);
          for (int j = 0; j < canocoeffs_now.rows(); j++)
            for (int k = 0; k < canocoeffs_now.cols(); k++)
              canocoeffs_now(j, k) = 0.0;

          for (int j = 0; j < cano_now.size(); j++)
          {
            for (int m = 0; m < cano_previous.size(); m++)
            {
              // action of E43
              if (cano_now[j].n13 == (cano_previous[m].n13 - 1) &&
                  cano_now[j].n23 == cano_previous[m].n23 &&
                  cano_now[j].n33 == cano_previous[m].n33 &&
                  cano_now[j].n12 == cano_previous[m].n12 &&
                  cano_now[j].n22 == cano_previous[m].n22 &&
                  cano_now[j].n11 == cano_previous[m].n11)
              {
                for (int k = 0; k < pbB[i].zeta; k++)
                {
                  canocoeffs_now(j, k) +=
                      canocoeffs_prev(m, k) * cano_previous[m].e43_13();
                }
              }
              else if (cano_now[j].n13 == cano_previous[m].n13 &&
                       cano_now[j].n23 == (cano_previous[m].n23 - 1) &&
                       cano_now[j].n33 == cano_previous[m].n33 &&
                       cano_now[j].n12 == cano_previous[m].n12 &&
                       cano_now[j].n22 == cano_previous[m].n22 &&
                       cano_now[j].n11 == cano_previous[m].n11)
              {
                for (int k = 0; k < pbB[i].zeta; k++)
                {
                  canocoeffs_now(j, k) +=
                      canocoeffs_prev(m, k) * cano_previous[m].e43_23();
                }
              }
              else if (cano_now[j].n13 == cano_previous[m].n13 &&
                       cano_now[j].n23 == cano_previous[m].n23 &&
                       cano_now[j].n33 == (cano_previous[m].n33 - 1) &&
                       cano_now[j].n12 == cano_previous[m].n12 &&
                       cano_now[j].n22 == cano_previous[m].n22 &&
                       cano_now[j].n11 == cano_previous[m].n11)
              {
                for (int k = 0; k < pbB[i].zeta; k++)
                {
                  canocoeffs_now(j, k) +=
                      canocoeffs_prev(m, k) * cano_previous[m].e43_33();
                }
              }

              // action of E21
              else if (cano_now[j].n13 == cano_previous[m].n13 &&
                       cano_now[j].n23 == cano_previous[m].n23 &&
                       cano_now[j].n33 == cano_previous[m].n33 &&
                       cano_now[j].n12 == cano_previous[m].n12 &&
                       cano_now[j].n22 == cano_previous[m].n22 &&
                       cano_now[j].n11 == (cano_previous[m].n11 - 1))
              {
                for (int k = 0; k < pbB[i].zeta; k++)
                {
                  canocoeffs_now(j, k) +=
                      canocoeffs_prev(m, k) * cano_previous[m].e21();
                }
              }
            }
          }

          allCanoBasisB[alpha][beta] = cano_now;
          allCanoCoeffsB[alpha][beta] = canocoeffs_now;
        }
      }

      // compute alpha_beta_coefficients
      for (int alpha = 0; alpha <= pbB[i].SS; alpha++)
      {
        for (int beta = 0; beta <= pbB[i].TT; beta++)
        {
          double SSalphaFac = 1.0, TTbetaFac = 1.0, SSFac = 1.0, alphaFac = 1.0,
                 TTFac = 1.0, betaFac = 1.0;

          for (int j = 1; j <= pbB[i].SS; j++)
          {
            SSFac *= j;
            if (j == alpha)
              alphaFac = SSFac;
            if (j == (pbB[i].SS - alpha))
              SSalphaFac = SSFac;
          }

          for (int j = 1; j <= pbB[i].TT; j++)
          {
            TTFac *= j;
            if (j == beta)
              betaFac = TTFac;
            if (j == (pbB[i].TT - beta))
              TTbetaFac = TTFac;
          }

          alpha_beta_coefficients[alpha][beta] = std::sqrt(
              SSalphaFac * TTbetaFac / SSFac / alphaFac / TTFac / betaFac);
        }
      }

      // last step is to compute Wigner coefficients < [ni4]1,zeta1,S1,T1 ;
      // [ni4]2,zeta2,S2,T2 || [ni4]3,zeta3,S3,T3 >
      for (int z_B = 1; z_B <= pbB[i].zeta; z_B++)
      {
        for (int j = 0; j < pbA.size(); j++)
        {
          for (int z_A = 1; z_A <= pbA[j].zeta; z_A++)
          {
            for (int k = 0; k < pbC.size(); k++)
            {
              for (int z_C = 1; z_C <= pbC[k].zeta; z_C++)
              {

                double so3cgcS =
                    CGC_SO3(pbA[j].SS, pbB[i].SS, pbC[k].SS, pbA[j].SS,
                            pbC[k].SS - pbA[j].SS, pbC[k].SS);
                double so3cgcT =
                    CGC_SO3(pbA[j].TT, pbB[i].TT, pbC[k].TT, pbA[j].TT,
                            pbC[k].TT - pbA[j].TT, pbC[k].TT);

                if (so3cgcS != 0.0 && so3cgcT != 0)
                {
                  int alpha = (pbA[j].SS + pbB[i].SS - pbC[k].SS) / 2,
                      beta = (pbA[j].TT + pbB[i].TT - pbC[k].TT) / 2;

                  u4::physicalvector pb1 = pbA[j], pb2 = pbB[i], pb3 = pbC[k];
                  pb1.zeta = z_A;
                  physvecA.push_back(pb1);
                  pb2.zeta = z_B;
                  physvecB.push_back(pb2);
                  pb3.zeta = z_C;
                  physvecC.push_back(pb3);

                  std::vector<u4::gelfand> cb1 = allCanoBasisA[j],
                                           cb2 = allCanoBasisB[alpha][beta],
                                           cb3 = allCanoBasisC[k];
                  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
                      canocoeffs1 = allCanoCoeffsA[j],
                      canocoeffs2 = allCanoCoeffsB[alpha][beta],
                      canocoeffs3 = allCanoCoeffsC[k];

                  int old_row = coeffs.rows();

                  coeffs.conservativeResize(old_row + 1, etamax);

                  for (int eta = 1; eta <= etamax; eta++)
                  {
                    coeffs(old_row, eta - 1) = 0.0;

                    for (int m = 0; m < cb1.size(); m++)
                    {
                      for (int n = 0; n < cb2.size(); n++)
                      {
                        for (int l = 0; l < cb3.size(); l++)
                        {
                          double canocoupling = 0.0;
                          for (int p = 0; p < canoCoupleCoeffs.rows(); p++)
                          {
                            if (u4::compare(cb1[m], gArrA[p]) &&
                                u4::compare(cb2[n], gArrB[p]) &&
                                u4::compare(cb3[l], gArrC[p]))
                            {
                              canocoupling = canoCoupleCoeffs(p, eta - 1);
                            }
                          }
                          coeffs(old_row, eta - 1) += canocoeffs1(m, z_A - 1) *
                                                      canocoeffs2(n, z_B - 1) *
                                                      canocoeffs3(l, z_C - 1) *
                                                      canocoupling;
                        }
                      }
                    }

                    coeffs(old_row, eta - 1) *=
                        (1 / so3cgcS) * (1 / so3cgcT) *
                        alpha_beta_coefficients[alpha][beta];
                  }
                }
                else
                {
                  continue;
                }
              }
            }
          }
        }
      }
    }

    return coeffs;
  }
*/
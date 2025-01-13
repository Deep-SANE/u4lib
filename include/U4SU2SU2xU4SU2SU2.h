// protection against multiple header file inclusion
#ifndef U4SU2SU2XU4SU2SU2_H
#define U4SU2SU2XU4SU2SU2_H

#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

#include "U4basis.h"
#include "U4SU2SU2_to_U4U3U2U1.h"
#include "U4U3U2U1xU4U3U2U1.h"

extern "C"
{
#include <wigxjpf.h>
}

namespace u4
{

  // usual SO(3) Clebsh-Gordan coefficients <JJ1 MM1; JJ2 MM2 | JJ3 MM3>
  double CGC_SO3(int jj1, int jj2, int jj3, int mm1, int mm2, int mm3)
  {
    int phase = ((-jj1 + jj2 - mm3) % 4 == 0) ? 1 : -1;

    wig_table_init(2 * 100, 9);
    wig_temp_init(2 * 100);

    double val3j = wig3jj(jj1, jj2, jj3, mm1, mm2, -mm3);

    wig_temp_free();
    wig_table_free();

    return phase * std::sqrt(jj3 + 1) * val3j;
  }

  // generate all Wigner U(4)->SU(2)xSU(2) coefficients
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

} // namespace u4

#endif
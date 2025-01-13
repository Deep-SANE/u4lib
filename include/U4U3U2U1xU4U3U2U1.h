// protection against multiple header file inclusion
#ifndef U4U3U2U1XU4U3U2U1_H
#define U4U3U2U1XU4U3U2U1_H

#include <assert.h>
#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "decomposition.h"
#include "U4basis.h"
#include "utils/inverseSolver.h"
#include "utils/nullspaceSolver.h"

namespace u4
{

  // expansion of a coupled gelfand basis state in terms of those uncoupled ones
  // based on the weight selection rule; hence, this expansion is the same for
  // those other coupled states that have the same weight with the state being
  // expanded
  void expansion(u4::irrep& u4a, u4::irrep& u4b, u4::gelfand& u4c,
                 std::vector<u4::gelfand>& gArr1,
                 std::vector<u4::gelfand>& gArr2)
  {
    for (int n13a = u4a.n14; n13a >= u4a.n24; n13a--)
    {
      for (int n23a = u4a.n24; n23a >= u4a.n34; n23a--)
      {
        for (int n33a = u4a.n34; n33a >= u4a.n44; n33a--)
        {
          for (int n12a = n13a; n12a >= n23a; n12a--)
          {
            for (int n22a = n23a; n22a >= n33a; n22a--)
            {
              for (int n11a = n12a; n11a >= n22a; n11a--)
              {

                for (int n13b = u4b.n14; n13b >= u4b.n24; n13b--)
                {
                  for (int n23b = u4b.n24; n23b >= u4b.n34; n23b--)
                  {
                    for (int n33b = u4b.n34; n33b >= u4b.n44; n33b--)
                    {
                      for (int n12b = n13b; n12b >= n23b; n12b--)
                      {
                        for (int n22b = n23b; n22b >= n33b; n22b--)
                        {
                          for (int n11b = n12b; n11b >= n22b; n11b--)
                          {
                            if ((n13a + n23a + n33a - n12a - n22a + n13b +
                                 n23b + n33b - n12b - n22b) == u4c.e33() &&
                                (n12a + n22a - n11a + n12b + n22b - n11b) ==
                                    u4c.e22() &&
                                (n11a + n11b) == u4c.e11())
                            {
                              u4::gelfand g1(u4a.n14, u4a.n24, u4a.n34, u4a.n44,
                                             n13a, n23a, n33a, n12a, n22a,
                                             n11a);
                              u4::gelfand g2(u4b.n14, u4b.n24, u4b.n34, u4b.n44,
                                             n13b, n23b, n33b, n12b, n22b,
                                             n11b);

                              gArr1.push_back(g1);
                              gArr2.push_back(g2);
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
              }
            }
          }
        }
      }
    }

    return;
  }

  void newstates(u4::gelfand& temp_gelfand, std::vector<u4::gelfand>& gArr1,
                 std::vector<u4::gelfand>& gArr2,
                 std::vector<u4::gelfand>& newgArr1,
                 std::vector<u4::gelfand>& newgArr2, int index)
  {
    if (newgArr1.size() == 0)
    {
      newgArr1.push_back(temp_gelfand);
      newgArr2.push_back(gArr2[index]);
    }
    else
    {
      int flag = 0;
      for (int j = 0; j < newgArr1.size(); j++)
      {
        if (u4::compare(temp_gelfand, newgArr1[j]) &&
            u4::compare(gArr2[index], newgArr2[j]))
        {
          flag = 1;
          break;
        }
      }
      if (flag == 0)
      {
        newgArr1.push_back(temp_gelfand);
        newgArr2.push_back(gArr2[index]);
      }
    }
    return;
  }

  // build the matrix representating a system of linear equations resulting from
  // the action two generators E34, E23 and E12 that annihiliate the highest
  // weight state
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  E34E23E12(std::vector<u4::gelfand>& gArr1, std::vector<u4::gelfand>& gArr2)
  {

    int col_num = gArr1.size();

    std::vector<u4::gelfand> newgArr1;
    std::vector<u4::gelfand> newgArr2;

    for (int i = 0; i < col_num; i++)
    {

      // E34, E23 and E12 act on the first state
      if ((gArr1[i].n13 + 1) <= gArr1[i].n14)
      {
        u4::gelfand temp_gelfand = gArr1[i];
        temp_gelfand.n13++;
        newstates(temp_gelfand, gArr1, gArr2, newgArr1, newgArr2, i);
      }
      if ((gArr1[i].n23 + 1) <= gArr1[i].n24)
      {
        u4::gelfand temp_gelfand = gArr1[i];
        temp_gelfand.n23++;
        newstates(temp_gelfand, gArr1, gArr2, newgArr1, newgArr2, i);
      }
      if ((gArr1[i].n33 + 1) <= gArr1[i].n34)
      {
        u4::gelfand temp_gelfand = gArr1[i];
        temp_gelfand.n33++;
        newstates(temp_gelfand, gArr1, gArr2, newgArr1, newgArr2, i);
      }
      if ((gArr1[i].n12 + 1) <= gArr1[i].n13)
      {
        u4::gelfand temp_gelfand = gArr1[i];
        temp_gelfand.n12++;
        newstates(temp_gelfand, gArr1, gArr2, newgArr1, newgArr2, i);
      }
      if ((gArr1[i].n22 + 1) <= gArr1[i].n23)
      {
        u4::gelfand temp_gelfand = gArr1[i];
        temp_gelfand.n22++;
        newstates(temp_gelfand, gArr1, gArr2, newgArr1, newgArr2, i);
      }
      if ((gArr1[i].n11 + 1) <= gArr1[i].n12)
      {
        u4::gelfand temp_gelfand = gArr1[i];
        temp_gelfand.n11++;
        newstates(temp_gelfand, gArr1, gArr2, newgArr1, newgArr2, i);
      }

      // E34, E23 and E12 act on the second state
      if ((gArr2[i].n13 + 1) <= gArr2[i].n14)
      {
        u4::gelfand temp_gelfand = gArr2[i];
        temp_gelfand.n13++;
        newstates(temp_gelfand, gArr2, gArr1, newgArr2, newgArr1, i);
      }
      if ((gArr2[i].n23 + 1) <= gArr2[i].n24)
      {
        u4::gelfand temp_gelfand = gArr2[i];
        temp_gelfand.n23++;
        newstates(temp_gelfand, gArr2, gArr1, newgArr2, newgArr1, i);
      }
      if ((gArr2[i].n33 + 1) <= gArr2[i].n34)
      {
        u4::gelfand temp_gelfand = gArr2[i];
        temp_gelfand.n33++;
        newstates(temp_gelfand, gArr2, gArr1, newgArr2, newgArr1, i);
      }
      if ((gArr2[i].n12 + 1) <= gArr2[i].n13)
      {
        u4::gelfand temp_gelfand = gArr2[i];
        temp_gelfand.n12++;
        newstates(temp_gelfand, gArr2, gArr1, newgArr2, newgArr1, i);
      }
      if ((gArr2[i].n22 + 1) <= gArr2[i].n23)
      {
        u4::gelfand temp_gelfand = gArr2[i];
        temp_gelfand.n22++;
        newstates(temp_gelfand, gArr2, gArr1, newgArr2, newgArr1, i);
      }
      if ((gArr2[i].n11 + 1) <= gArr2[i].n12)
      {
        u4::gelfand temp_gelfand = gArr2[i];
        temp_gelfand.n11++;
        newstates(temp_gelfand, gArr2, gArr1, newgArr2, newgArr1, i);
      }
    }

    int row_num = newgArr1.size();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat(row_num, col_num);

    for (int i = 0; i < row_num; i++)
    {
      for (int j = 0; j < col_num; j++)
      {

        // action of E34
        if (newgArr1[i].n13 == (gArr1[j].n13 + 1) &&
            newgArr1[i].n23 == gArr1[j].n23 &&
            newgArr1[i].n33 == gArr1[j].n33 &&
            newgArr1[i].n12 == gArr1[j].n12 &&
            newgArr1[i].n22 == gArr1[j].n22 &&
            newgArr1[i].n11 == gArr1[j].n11 &&
            u4::compare(newgArr2[i], gArr2[j]))
        {
          mat(i, j) = gArr1[j].e34_13();
        }
        else if (newgArr1[i].n13 == gArr1[j].n13 &&
                 newgArr1[i].n23 == (gArr1[j].n23 + 1) &&
                 newgArr1[i].n33 == gArr1[j].n33 &&
                 newgArr1[i].n12 == gArr1[j].n12 &&
                 newgArr1[i].n22 == gArr1[j].n22 &&
                 newgArr1[i].n11 == gArr1[j].n11 &&
                 u4::compare(newgArr2[i], gArr2[j]))
        {
          mat(i, j) = gArr1[j].e34_23();
        }
        else if (newgArr1[i].n13 == gArr1[j].n13 &&
                 newgArr1[i].n23 == gArr1[j].n23 &&
                 newgArr1[i].n33 == (gArr1[j].n33 + 1) &&
                 newgArr1[i].n12 == gArr1[j].n12 &&
                 newgArr1[i].n22 == gArr1[j].n22 &&
                 newgArr1[i].n11 == gArr1[j].n11 &&
                 u4::compare(newgArr2[i], gArr2[j]))
        {
          mat(i, j) = gArr1[j].e34_33();
        }
        else if (u4::compare(newgArr1[i], gArr1[j]) &&
                 newgArr2[i].n13 == (gArr2[j].n13 + 1) &&
                 newgArr2[i].n23 == gArr2[j].n23 &&
                 newgArr2[i].n33 == gArr2[j].n33 &&
                 newgArr2[i].n12 == gArr2[j].n12 &&
                 newgArr2[i].n22 == gArr2[j].n22 &&
                 newgArr2[i].n11 == gArr2[j].n11)
        {
          mat(i, j) = gArr2[j].e34_13();
        }
        else if (u4::compare(newgArr1[i], gArr1[j]) &&
                 newgArr2[i].n13 == gArr2[j].n13 &&
                 newgArr2[i].n23 == (gArr2[j].n23 + 1) &&
                 newgArr2[i].n33 == gArr2[j].n33 &&
                 newgArr2[i].n12 == gArr2[j].n12 &&
                 newgArr2[i].n22 == gArr2[j].n22 &&
                 newgArr2[i].n11 == gArr2[j].n11)
        {
          mat(i, j) = gArr2[j].e34_23();
        }
        else if (u4::compare(newgArr1[i], gArr1[j]) &&
                 newgArr2[i].n13 == gArr2[j].n13 &&
                 newgArr2[i].n23 == gArr2[j].n23 &&
                 newgArr2[i].n33 == (gArr2[j].n33 + 1) &&
                 newgArr2[i].n12 == gArr2[j].n12 &&
                 newgArr2[i].n22 == gArr2[j].n22 &&
                 newgArr2[i].n11 == gArr2[j].n11)
        {
          mat(i, j) = gArr2[j].e34_33();
        }

        // action of E23
        else if (newgArr1[i].n13 == gArr1[j].n13 &&
                 newgArr1[i].n23 == gArr1[j].n23 &&
                 newgArr1[i].n33 == gArr1[j].n33 &&
                 newgArr1[i].n12 == (gArr1[j].n12 + 1) &&
                 newgArr1[i].n22 == gArr1[j].n22 &&
                 newgArr1[i].n11 == gArr1[j].n11 &&
                 u4::compare(newgArr2[i], gArr2[j]))
        {
          mat(i, j) = gArr1[j].e23_12();
        }
        else if (newgArr1[i].n13 == gArr1[j].n13 &&
                 newgArr1[i].n23 == gArr1[j].n23 &&
                 newgArr1[i].n33 == gArr1[j].n33 &&
                 newgArr1[i].n12 == gArr1[j].n12 &&
                 newgArr1[i].n22 == (gArr1[j].n22 + 1) &&
                 newgArr1[i].n11 == gArr1[j].n11 &&
                 u4::compare(newgArr2[i], gArr2[j]))
        {
          mat(i, j) = gArr1[j].e23_22();
        }
        else if (u4::compare(newgArr1[i], gArr1[j]) &&
                 newgArr2[i].n13 == gArr2[j].n13 &&
                 newgArr2[i].n23 == gArr2[j].n23 &&
                 newgArr2[i].n33 == gArr2[j].n33 &&
                 newgArr2[i].n12 == (gArr2[j].n12 + 1) &&
                 newgArr2[i].n22 == gArr2[j].n22 &&
                 newgArr2[i].n11 == gArr2[j].n11)
        {
          mat(i, j) = gArr2[j].e23_12();
        }
        else if (u4::compare(newgArr1[i], gArr1[j]) &&
                 newgArr2[i].n13 == gArr2[j].n13 &&
                 newgArr2[i].n23 == gArr2[j].n23 &&
                 newgArr2[i].n33 == gArr2[j].n33 &&
                 newgArr2[i].n12 == gArr2[j].n12 &&
                 newgArr2[i].n22 == (gArr2[j].n22 + 1) &&
                 newgArr2[i].n11 == gArr2[j].n11)
        {
          mat(i, j) = gArr2[j].e23_22();
        }

        // action of E12
        else if (newgArr1[i].n13 == gArr1[j].n13 &&
                 newgArr1[i].n23 == gArr1[j].n23 &&
                 newgArr1[i].n33 == gArr1[j].n33 &&
                 newgArr1[i].n12 == gArr1[j].n12 &&
                 newgArr1[i].n22 == gArr1[j].n22 &&
                 newgArr1[i].n11 == (gArr1[j].n11 + 1) &&
                 u4::compare(newgArr2[i], gArr2[j]))
        {
          mat(i, j) = gArr1[j].e12();
        }
        else if (u4::compare(newgArr1[i], gArr1[j]) &&
                 newgArr2[i].n13 == gArr2[j].n13 &&
                 newgArr2[i].n23 == gArr2[j].n23 &&
                 newgArr2[i].n33 == gArr2[j].n33 &&
                 newgArr2[i].n12 == gArr2[j].n12 &&
                 newgArr2[i].n22 == gArr2[j].n22 &&
                 newgArr2[i].n11 == (gArr2[j].n11 + 1))
        {
          mat(i, j) = gArr2[j].e12();
        }

        else
        {
          mat(i, j) = 0.0;
        }
      }
    }

    return mat;
  }

  // highest-weight coefficients < G1; G2| HW3 >
  void HighestWeightCoeffs(
      u4::irrep& u4a, u4::irrep& u4b, u4::gelfand& HWc,
      std::vector<u4::gelfand>& gArr1, std::vector<u4::gelfand>& gArr2,
      std::vector<u4::gelfand>& gArr3,
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& coeffs)
  {

    expansion(u4a, u4b, HWc, gArr1, gArr2);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat =
        E34E23E12(gArr1, gArr2);

    coeffs = null_space(mat);

    for (int i = 0; i < coeffs.rows(); i++)
      gArr3.push_back(HWc);

    return;
  }

  // build gelfand basis states from a given weight
  std::vector<u4::gelfand> SameWeight(u4::irrep& u4, int w4, int w3, int w1)
  {
    std::vector<u4::gelfand> sameweight;

    for (int n13 = u4.n14; n13 >= u4.n24; n13--)
    {
      for (int n23 = u4.n24; n23 >= u4.n34; n23--)
      {
        for (int n33 = u4.n34; n33 >= u4.n44; n33--)
        {
          for (int n12 = n13; n12 >= n23; n12--)
          {
            for (int n22 = n23; n22 >= n33; n22--)
            {
              for (int n11 = n12; n11 >= n22; n11--)
              {
                if (n11 == w1 && (n13 + n23 + n33 - n12 - n22) == w3 &&
                    (u4.n14 + u4.n24 + u4.n34 + u4.n44 - n13 - n23 - n33) == w4)
                {
                  u4::gelfand g(u4.n14, u4.n24, u4.n34, u4.n44, n13, n23, n33,
                                n12, n22, n11);
                  sameweight.push_back(g);
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

    return sameweight;
  }

  // lower-weight coefficients < G1; G2 | G3 >
  void LowerWeightCoeffs(
      u4::irrep& u4a, u4::irrep& u4b, u4::irrep& u4c, int w4, int w3, int w1,
      std::vector<u4::gelfand>& gArr1, std::vector<u4::gelfand>& gArr2,
      std::vector<u4::gelfand>& gArr3,
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& coeffs)
  {

    // build all coupled states that have the same weight
    std::vector<u4::gelfand> gArr3_temp = SameWeight(u4c, w4, w3, w1);

    if (gArr3_temp.size() != 0)
    {

      // std::cout << gArr3_temp.size() << std::endl;
      std::vector<u4::gelfand> gArr1_temp;
      std::vector<u4::gelfand> gArr2_temp;

      // the coupled states having the same weight have the same expansion of
      // uncoupled states
      expansion(u4a, u4b, gArr3_temp[0], gArr1_temp, gArr2_temp);
      // std::cout << gArr1_temp.size() << std::endl;

      coeffs.conservativeResize(
          coeffs.rows() + gArr3_temp.size() * gArr1_temp.size(), coeffs.cols());

      std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
          coeffs_temp;

      for (int outer_multi = 0; outer_multi < coeffs.cols(); outer_multi++)
      {

        // build the b-matrix
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> b(
            gArr3_temp.size(), gArr3_temp.size());

        // build the lowering matrix from the action of E21, E32, and E43
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> E21E32E43(
            gArr3_temp.size(), gArr1_temp.size());

        // set matrix elements to zero
        for (int i = 0; i < gArr3_temp.size(); i++)
        {
          for (int j = 0; j < gArr3_temp.size(); j++)
            b(i, j) = 0.0;
          for (int j = 0; j < gArr1_temp.size(); j++)
            E21E32E43(i, j) = 0.0;
        }

        for (int i = 0; i < gArr3_temp.size(); i++)
        {

          int flag = 0;

          // build with generator E21
          u4::gelfand g3_n11p1 = gArr3_temp[i];
          g3_n11p1.n11++; // this is a parent state
          if (g3_n11p1.check())
          {
            for (int k = 0; k < coeffs.rows(); k++)
            {
              if (u4::compare(g3_n11p1, gArr3[k]))
              {
                for (int j = 0; j < gArr3_temp.size(); j++)
                {
                  if (gArr3[k].n13 == gArr3_temp[j].n13 &&
                      gArr3[k].n23 == gArr3_temp[j].n23 &&
                      gArr3[k].n33 == gArr3_temp[j].n33 &&
                      gArr3[k].n12 == gArr3_temp[j].n12 &&
                      gArr3[k].n22 == gArr3_temp[j].n22 &&
                      (gArr3[k].n11 - 1) == gArr3_temp[j].n11)
                  {
                    b(i, j) = gArr3[k].e21();
                  }
                }
                for (int j = 0; j < gArr1_temp.size(); j++)
                {
                  for (int l = k; l < coeffs.rows(); l++)
                  {
                    if (u4::compare(gArr3[k], gArr3[l]) &&
                        u4::compare(gArr2_temp[j], gArr2[l]) &&
                        gArr1_temp[j].n13 == gArr1[l].n13 &&
                        gArr1_temp[j].n23 == gArr1[l].n23 &&
                        gArr1_temp[j].n33 == gArr1[l].n33 &&
                        gArr1_temp[j].n12 == gArr1[l].n12 &&
                        gArr1_temp[j].n22 == gArr1[l].n22 &&
                        gArr1_temp[j].n11 == (gArr1[l].n11 - 1))
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr1[l].e21();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr1_temp[j], gArr1[l]) &&
                             gArr2_temp[j].n13 == gArr2[l].n13 &&
                             gArr2_temp[j].n23 == gArr2[l].n23 &&
                             gArr2_temp[j].n33 == gArr2[l].n33 &&
                             gArr2_temp[j].n12 == gArr2[l].n12 &&
                             gArr2_temp[j].n22 == gArr2[l].n22 &&
                             gArr2_temp[j].n11 == (gArr2[l].n11 - 1))
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr2[l].e21();
                      flag++;
                    }
                  }
                }
                break;
              }
            }
          }

          if (flag > 0)
            continue;

          // build with generator E32

          u4::gelfand g3_n12p1 = gArr3_temp[i];
          g3_n12p1.n12++; // this is a parent state
          if (g3_n12p1.check())
          {
            for (int k = 0; k < coeffs.rows(); k++)
            {
              if (u4::compare(g3_n12p1, gArr3[k]))
              {
                for (int j = 0; j < gArr3_temp.size(); j++)
                {
                  if (gArr3[k].n13 == gArr3_temp[j].n13 &&
                      gArr3[k].n23 == gArr3_temp[j].n23 &&
                      gArr3[k].n33 == gArr3_temp[j].n33 &&
                      (gArr3[k].n12 - 1) == gArr3_temp[j].n12 &&
                      gArr3[k].n22 == gArr3_temp[j].n22 &&
                      gArr3[k].n11 == gArr3_temp[j].n11)
                  {
                    b(i, j) = gArr3[k].e32_12();
                  }
                  else if (gArr3[k].n13 == gArr3_temp[j].n13 &&
                           gArr3[k].n23 == gArr3_temp[j].n23 &&
                           gArr3[k].n33 == gArr3_temp[j].n33 &&
                           gArr3[k].n12 == gArr3_temp[j].n12 &&
                           (gArr3[k].n22 - 1) == gArr3_temp[j].n22 &&
                           gArr3[k].n11 == gArr3_temp[j].n11)
                  {
                    b(i, j) = gArr3[k].e32_22();
                  }
                }
                for (int j = 0; j < gArr1_temp.size(); j++)
                {
                  for (int l = k; l < coeffs.rows(); l++)
                  {
                    if (u4::compare(gArr3[k], gArr3[l]) &&
                        u4::compare(gArr2_temp[j], gArr2[l]) &&
                        gArr1_temp[j].n13 == gArr1[l].n13 &&
                        gArr1_temp[j].n23 == gArr1[l].n23 &&
                        gArr1_temp[j].n33 == gArr1[l].n33 &&
                        gArr1_temp[j].n12 == (gArr1[l].n12 - 1) &&
                        gArr1_temp[j].n22 == gArr1[l].n22 &&
                        gArr1_temp[j].n11 == gArr1[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr1[l].e32_12();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr2_temp[j], gArr2[l]) &&
                             gArr1_temp[j].n13 == gArr1[l].n13 &&
                             gArr1_temp[j].n23 == gArr1[l].n23 &&
                             gArr1_temp[j].n33 == gArr1[l].n33 &&
                             gArr1_temp[j].n12 == gArr1[l].n12 &&
                             gArr1_temp[j].n22 == (gArr1[l].n22 - 1) &&
                             gArr1_temp[j].n11 == gArr1[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr1[l].e32_22();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr1_temp[j], gArr1[l]) &&
                             gArr2_temp[j].n13 == gArr2[l].n13 &&
                             gArr2_temp[j].n23 == gArr2[l].n23 &&
                             gArr2_temp[j].n33 == gArr2[l].n33 &&
                             gArr2_temp[j].n12 == (gArr2[l].n12 - 1) &&
                             gArr2_temp[j].n22 == gArr2[l].n22 &&
                             gArr2_temp[j].n11 == gArr2[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr2[l].e32_12();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr1_temp[j], gArr1[l]) &&
                             gArr2_temp[j].n13 == gArr2[l].n13 &&
                             gArr2_temp[j].n23 == gArr2[l].n23 &&
                             gArr2_temp[j].n33 == gArr2[l].n33 &&
                             gArr2_temp[j].n12 == gArr2[l].n12 &&
                             gArr2_temp[j].n22 == (gArr2[l].n22 - 1) &&
                             gArr2_temp[j].n11 == gArr2[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr2[l].e32_22();
                      flag++;
                    }
                  }
                }
                break;
              }
            }
          }

          if (flag > 0)
            continue;

          u4::gelfand g3_n22p1 = gArr3_temp[i];
          g3_n22p1.n22++; // this is a parent state
          if (g3_n22p1.check())
          {
            for (int k = 0; k < coeffs.rows(); k++)
            {
              if (u4::compare(g3_n22p1, gArr3[k]))
              {
                for (int j = 0; j < gArr3_temp.size(); j++)
                {
                  if (gArr3[k].n13 == gArr3_temp[j].n13 &&
                      gArr3[k].n23 == gArr3_temp[j].n23 &&
                      gArr3[k].n33 == gArr3_temp[j].n33 &&
                      (gArr3[k].n12 - 1) == gArr3_temp[j].n12 &&
                      gArr3[k].n22 == gArr3_temp[j].n22 &&
                      gArr3[k].n11 == gArr3_temp[j].n11)
                  {
                    b(i, j) = gArr3[k].e32_12();
                  }
                  else if (gArr3[k].n13 == gArr3_temp[j].n13 &&
                           gArr3[k].n23 == gArr3_temp[j].n23 &&
                           gArr3[k].n33 == gArr3_temp[j].n33 &&
                           gArr3[k].n12 == gArr3_temp[j].n12 &&
                           (gArr3[k].n22 - 1) == gArr3_temp[j].n22 &&
                           gArr3[k].n11 == gArr3_temp[j].n11)
                  {
                    b(i, j) = gArr3[k].e32_22();
                  }
                }
                for (int j = 0; j < gArr1_temp.size(); j++)
                {
                  for (int l = k; l < coeffs.rows(); l++)
                  {
                    if (u4::compare(gArr3[k], gArr3[l]) &&
                        u4::compare(gArr2_temp[j], gArr2[l]) &&
                        gArr1_temp[j].n13 == gArr1[l].n13 &&
                        gArr1_temp[j].n23 == gArr1[l].n23 &&
                        gArr1_temp[j].n33 == gArr1[l].n33 &&
                        gArr1_temp[j].n12 == (gArr1[l].n12 - 1) &&
                        gArr1_temp[j].n22 == gArr1[l].n22 &&
                        gArr1_temp[j].n11 == gArr1[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr1[l].e32_12();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr2_temp[j], gArr2[l]) &&
                             gArr1_temp[j].n13 == gArr1[l].n13 &&
                             gArr1_temp[j].n23 == gArr1[l].n23 &&
                             gArr1_temp[j].n33 == gArr1[l].n33 &&
                             gArr1_temp[j].n12 == gArr1[l].n12 &&
                             gArr1_temp[j].n22 == (gArr1[l].n22 - 1) &&
                             gArr1_temp[j].n11 == gArr1[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr1[l].e32_22();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr1_temp[j], gArr1[l]) &&
                             gArr2_temp[j].n13 == gArr2[l].n13 &&
                             gArr2_temp[j].n23 == gArr2[l].n23 &&
                             gArr2_temp[j].n33 == gArr2[l].n33 &&
                             gArr2_temp[j].n12 == (gArr2[l].n12 - 1) &&
                             gArr2_temp[j].n22 == gArr2[l].n22 &&
                             gArr2_temp[j].n11 == gArr2[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr2[l].e32_12();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr1_temp[j], gArr1[l]) &&
                             gArr2_temp[j].n13 == gArr2[l].n13 &&
                             gArr2_temp[j].n23 == gArr2[l].n23 &&
                             gArr2_temp[j].n33 == gArr2[l].n33 &&
                             gArr2_temp[j].n12 == gArr2[l].n12 &&
                             gArr2_temp[j].n22 == (gArr2[l].n22 - 1) &&
                             gArr2_temp[j].n11 == gArr2[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr2[l].e32_22();
                      flag++;
                    }
                  }
                }
                break;
              }
            }
          }

          if (flag > 0)
            continue;

          // build with generator E43

          u4::gelfand g3_n13p1 = gArr3_temp[i];
          g3_n13p1.n13++; // this is a parent state
          if (g3_n13p1.check())
          {
            for (int k = 0; k < coeffs.rows(); k++)
            {
              if (u4::compare(g3_n13p1, gArr3[k]))
              {
                for (int j = 0; j < gArr3_temp.size(); j++)
                {
                  if ((gArr3[k].n13 - 1) == gArr3_temp[j].n13 &&
                      gArr3[k].n23 == gArr3_temp[j].n23 &&
                      gArr3[k].n33 == gArr3_temp[j].n33 &&
                      gArr3[k].n12 == gArr3_temp[j].n12 &&
                      gArr3[k].n22 == gArr3_temp[j].n22 &&
                      gArr3[k].n11 == gArr3_temp[j].n11)
                  {
                    b(i, j) = gArr3[k].e43_13();
                  }
                  else if (gArr3[k].n13 == gArr3_temp[j].n13 &&
                           (gArr3[k].n23 - 1) == gArr3_temp[j].n23 &&
                           gArr3[k].n33 == gArr3_temp[j].n33 &&
                           gArr3[k].n12 == gArr3_temp[j].n12 &&
                           gArr3[k].n22 == gArr3_temp[j].n22 &&
                           gArr3[k].n11 == gArr3_temp[j].n11)
                  {
                    b(i, j) = gArr3[k].e43_23();
                  }
                  else if (gArr3[k].n13 == gArr3_temp[j].n13 &&
                           gArr3[k].n23 == gArr3_temp[j].n23 &&
                           (gArr3[k].n33 - 1) == gArr3_temp[j].n33 &&
                           gArr3[k].n12 == gArr3_temp[j].n12 &&
                           gArr3[k].n22 == gArr3_temp[j].n22 &&
                           gArr3[k].n11 == gArr3_temp[j].n11)
                  {
                    b(i, j) = gArr3[k].e43_33();
                  }
                }
                for (int j = 0; j < gArr1_temp.size(); j++)
                {
                  for (int l = k; l < coeffs.rows(); l++)
                  {
                    if (u4::compare(gArr3[k], gArr3[l]) &&
                        u4::compare(gArr2_temp[j], gArr2[l]) &&
                        gArr1_temp[j].n13 == (gArr1[l].n13 - 1) &&
                        gArr1_temp[j].n23 == gArr1[l].n23 &&
                        gArr1_temp[j].n33 == gArr1[l].n33 &&
                        gArr1_temp[j].n12 == gArr1[l].n12 &&
                        gArr1_temp[j].n22 == gArr1[l].n22 &&
                        gArr1_temp[j].n11 == gArr1[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr1[l].e43_13();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr2_temp[j], gArr2[l]) &&
                             gArr1_temp[j].n13 == gArr1[l].n13 &&
                             gArr1_temp[j].n23 == (gArr1[l].n23 - 1) &&
                             gArr1_temp[j].n33 == gArr1[l].n33 &&
                             gArr1_temp[j].n12 == gArr1[l].n12 &&
                             gArr1_temp[j].n22 == gArr1[l].n22 &&
                             gArr1_temp[j].n11 == gArr1[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr1[l].e43_23();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr2_temp[j], gArr2[l]) &&
                             gArr1_temp[j].n13 == gArr1[l].n13 &&
                             gArr1_temp[j].n23 == gArr1[l].n23 &&
                             gArr1_temp[j].n33 == (gArr1[l].n33 - 1) &&
                             gArr1_temp[j].n12 == gArr1[l].n12 &&
                             gArr1_temp[j].n22 == gArr1[l].n22 &&
                             gArr1_temp[j].n11 == gArr1[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr1[l].e43_33();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr1_temp[j], gArr1[l]) &&
                             gArr2_temp[j].n13 == (gArr2[l].n13 - 1) &&
                             gArr2_temp[j].n23 == gArr2[l].n23 &&
                             gArr2_temp[j].n33 == gArr2[l].n33 &&
                             gArr2_temp[j].n12 == gArr2[l].n12 &&
                             gArr2_temp[j].n22 == gArr2[l].n22 &&
                             gArr2_temp[j].n11 == gArr2[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr2[l].e43_13();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr1_temp[j], gArr1[l]) &&
                             gArr2_temp[j].n13 == gArr2[l].n13 &&
                             gArr2_temp[j].n23 == (gArr2[l].n23 - 1) &&
                             gArr2_temp[j].n33 == gArr2[l].n33 &&
                             gArr2_temp[j].n12 == gArr2[l].n12 &&
                             gArr2_temp[j].n22 == gArr2[l].n22 &&
                             gArr2_temp[j].n11 == gArr2[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr2[l].e43_23();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr1_temp[j], gArr1[l]) &&
                             gArr2_temp[j].n13 == gArr2[l].n13 &&
                             gArr2_temp[j].n23 == gArr2[l].n23 &&
                             gArr2_temp[j].n33 == (gArr2[l].n33 - 1) &&
                             gArr2_temp[j].n12 == gArr2[l].n12 &&
                             gArr2_temp[j].n22 == gArr2[l].n22 &&
                             gArr2_temp[j].n11 == gArr2[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr2[l].e43_33();
                      flag++;
                    }
                  }
                }
                break;
              }
            }
          }

          if (flag > 0)
            continue;

          u4::gelfand g3_n23p1 = gArr3_temp[i];
          g3_n23p1.n23++; // this is a parent state
          if (g3_n23p1.check())
          {
            for (int k = 0; k < coeffs.rows(); k++)
            {
              if (u4::compare(g3_n23p1, gArr3[k]))
              {
                for (int j = 0; j < gArr3_temp.size(); j++)
                {
                  if ((gArr3[k].n13 - 1) == gArr3_temp[j].n13 &&
                      gArr3[k].n23 == gArr3_temp[j].n23 &&
                      gArr3[k].n33 == gArr3_temp[j].n33 &&
                      gArr3[k].n12 == gArr3_temp[j].n12 &&
                      gArr3[k].n22 == gArr3_temp[j].n22 &&
                      gArr3[k].n11 == gArr3_temp[j].n11)
                  {
                    b(i, j) = gArr3[k].e43_13();
                  }
                  else if (gArr3[k].n13 == gArr3_temp[j].n13 &&
                           (gArr3[k].n23 - 1) == gArr3_temp[j].n23 &&
                           gArr3[k].n33 == gArr3_temp[j].n33 &&
                           gArr3[k].n12 == gArr3_temp[j].n12 &&
                           gArr3[k].n22 == gArr3_temp[j].n22 &&
                           gArr3[k].n11 == gArr3_temp[j].n11)
                  {
                    b(i, j) = gArr3[k].e43_23();
                  }
                  else if (gArr3[k].n13 == gArr3_temp[j].n13 &&
                           gArr3[k].n23 == gArr3_temp[j].n23 &&
                           (gArr3[k].n33 - 1) == gArr3_temp[j].n33 &&
                           gArr3[k].n12 == gArr3_temp[j].n12 &&
                           gArr3[k].n22 == gArr3_temp[j].n22 &&
                           gArr3[k].n11 == gArr3_temp[j].n11)
                  {
                    b(i, j) = gArr3[k].e43_33();
                  }
                }
                for (int j = 0; j < gArr1_temp.size(); j++)
                {
                  for (int l = k; l < coeffs.rows(); l++)
                  {
                    if (u4::compare(gArr3[k], gArr3[l]) &&
                        u4::compare(gArr2_temp[j], gArr2[l]) &&
                        gArr1_temp[j].n13 == (gArr1[l].n13 - 1) &&
                        gArr1_temp[j].n23 == gArr1[l].n23 &&
                        gArr1_temp[j].n33 == gArr1[l].n33 &&
                        gArr1_temp[j].n12 == gArr1[l].n12 &&
                        gArr1_temp[j].n22 == gArr1[l].n22 &&
                        gArr1_temp[j].n11 == gArr1[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr1[l].e43_13();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr2_temp[j], gArr2[l]) &&
                             gArr1_temp[j].n13 == gArr1[l].n13 &&
                             gArr1_temp[j].n23 == (gArr1[l].n23 - 1) &&
                             gArr1_temp[j].n33 == gArr1[l].n33 &&
                             gArr1_temp[j].n12 == gArr1[l].n12 &&
                             gArr1_temp[j].n22 == gArr1[l].n22 &&
                             gArr1_temp[j].n11 == gArr1[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr1[l].e43_23();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr2_temp[j], gArr2[l]) &&
                             gArr1_temp[j].n13 == gArr1[l].n13 &&
                             gArr1_temp[j].n23 == gArr1[l].n23 &&
                             gArr1_temp[j].n33 == (gArr1[l].n33 - 1) &&
                             gArr1_temp[j].n12 == gArr1[l].n12 &&
                             gArr1_temp[j].n22 == gArr1[l].n22 &&
                             gArr1_temp[j].n11 == gArr1[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr1[l].e43_33();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr1_temp[j], gArr1[l]) &&
                             gArr2_temp[j].n13 == (gArr2[l].n13 - 1) &&
                             gArr2_temp[j].n23 == gArr2[l].n23 &&
                             gArr2_temp[j].n33 == gArr2[l].n33 &&
                             gArr2_temp[j].n12 == gArr2[l].n12 &&
                             gArr2_temp[j].n22 == gArr2[l].n22 &&
                             gArr2_temp[j].n11 == gArr2[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr2[l].e43_13();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr1_temp[j], gArr1[l]) &&
                             gArr2_temp[j].n13 == gArr2[l].n13 &&
                             gArr2_temp[j].n23 == (gArr2[l].n23 - 1) &&
                             gArr2_temp[j].n33 == gArr2[l].n33 &&
                             gArr2_temp[j].n12 == gArr2[l].n12 &&
                             gArr2_temp[j].n22 == gArr2[l].n22 &&
                             gArr2_temp[j].n11 == gArr2[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr2[l].e43_23();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr1_temp[j], gArr1[l]) &&
                             gArr2_temp[j].n13 == gArr2[l].n13 &&
                             gArr2_temp[j].n23 == gArr2[l].n23 &&
                             gArr2_temp[j].n33 == (gArr2[l].n33 - 1) &&
                             gArr2_temp[j].n12 == gArr2[l].n12 &&
                             gArr2_temp[j].n22 == gArr2[l].n22 &&
                             gArr2_temp[j].n11 == gArr2[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr2[l].e43_33();
                      flag++;
                    }
                  }
                }
                break;
              }
            }
          }

          if (flag > 0)
            continue;

          u4::gelfand g3_n33p1 = gArr3_temp[i];
          g3_n33p1.n33++; // this is a parent state
          if (g3_n33p1.check())
          {
            for (int k = 0; k < coeffs.rows(); k++)
            {
              if (u4::compare(g3_n33p1, gArr3[k]))
              {
                for (int j = 0; j < gArr3_temp.size(); j++)
                {
                  if ((gArr3[k].n13 - 1) == gArr3_temp[j].n13 &&
                      gArr3[k].n23 == gArr3_temp[j].n23 &&
                      gArr3[k].n33 == gArr3_temp[j].n33 &&
                      gArr3[k].n12 == gArr3_temp[j].n12 &&
                      gArr3[k].n22 == gArr3_temp[j].n22 &&
                      gArr3[k].n11 == gArr3_temp[j].n11)
                  {
                    b(i, j) = gArr3[k].e43_13();
                  }
                  else if (gArr3[k].n13 == gArr3_temp[j].n13 &&
                           (gArr3[k].n23 - 1) == gArr3_temp[j].n23 &&
                           gArr3[k].n33 == gArr3_temp[j].n33 &&
                           gArr3[k].n12 == gArr3_temp[j].n12 &&
                           gArr3[k].n22 == gArr3_temp[j].n22 &&
                           gArr3[k].n11 == gArr3_temp[j].n11)
                  {
                    b(i, j) = gArr3[k].e43_23();
                  }
                  else if (gArr3[k].n13 == gArr3_temp[j].n13 &&
                           gArr3[k].n23 == gArr3_temp[j].n23 &&
                           (gArr3[k].n33 - 1) == gArr3_temp[j].n33 &&
                           gArr3[k].n12 == gArr3_temp[j].n12 &&
                           gArr3[k].n22 == gArr3_temp[j].n22 &&
                           gArr3[k].n11 == gArr3_temp[j].n11)
                  {
                    b(i, j) = gArr3[k].e43_33();
                  }
                }
                for (int j = 0; j < gArr1_temp.size(); j++)
                {
                  for (int l = k; l < coeffs.rows(); l++)
                  {
                    if (u4::compare(gArr3[k], gArr3[l]) &&
                        u4::compare(gArr2_temp[j], gArr2[l]) &&
                        gArr1_temp[j].n13 == (gArr1[l].n13 - 1) &&
                        gArr1_temp[j].n23 == gArr1[l].n23 &&
                        gArr1_temp[j].n33 == gArr1[l].n33 &&
                        gArr1_temp[j].n12 == gArr1[l].n12 &&
                        gArr1_temp[j].n22 == gArr1[l].n22 &&
                        gArr1_temp[j].n11 == gArr1[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr1[l].e43_13();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr2_temp[j], gArr2[l]) &&
                             gArr1_temp[j].n13 == gArr1[l].n13 &&
                             gArr1_temp[j].n23 == (gArr1[l].n23 - 1) &&
                             gArr1_temp[j].n33 == gArr1[l].n33 &&
                             gArr1_temp[j].n12 == gArr1[l].n12 &&
                             gArr1_temp[j].n22 == gArr1[l].n22 &&
                             gArr1_temp[j].n11 == gArr1[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr1[l].e43_23();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr2_temp[j], gArr2[l]) &&
                             gArr1_temp[j].n13 == gArr1[l].n13 &&
                             gArr1_temp[j].n23 == gArr1[l].n23 &&
                             gArr1_temp[j].n33 == (gArr1[l].n33 - 1) &&
                             gArr1_temp[j].n12 == gArr1[l].n12 &&
                             gArr1_temp[j].n22 == gArr1[l].n22 &&
                             gArr1_temp[j].n11 == gArr1[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr1[l].e43_33();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr1_temp[j], gArr1[l]) &&
                             gArr2_temp[j].n13 == (gArr2[l].n13 - 1) &&
                             gArr2_temp[j].n23 == gArr2[l].n23 &&
                             gArr2_temp[j].n33 == gArr2[l].n33 &&
                             gArr2_temp[j].n12 == gArr2[l].n12 &&
                             gArr2_temp[j].n22 == gArr2[l].n22 &&
                             gArr2_temp[j].n11 == gArr2[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr2[l].e43_13();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr1_temp[j], gArr1[l]) &&
                             gArr2_temp[j].n13 == gArr2[l].n13 &&
                             gArr2_temp[j].n23 == (gArr2[l].n23 - 1) &&
                             gArr2_temp[j].n33 == gArr2[l].n33 &&
                             gArr2_temp[j].n12 == gArr2[l].n12 &&
                             gArr2_temp[j].n22 == gArr2[l].n22 &&
                             gArr2_temp[j].n11 == gArr2[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr2[l].e43_23();
                      flag++;
                    }
                    else if (u4::compare(gArr3[k], gArr3[l]) &&
                             u4::compare(gArr1_temp[j], gArr1[l]) &&
                             gArr2_temp[j].n13 == gArr2[l].n13 &&
                             gArr2_temp[j].n23 == gArr2[l].n23 &&
                             gArr2_temp[j].n33 == (gArr2[l].n33 - 1) &&
                             gArr2_temp[j].n12 == gArr2[l].n12 &&
                             gArr2_temp[j].n22 == gArr2[l].n22 &&
                             gArr2_temp[j].n11 == gArr2[l].n11)
                    {
                      E21E32E43(i, j) +=
                          coeffs(l, outer_multi) * gArr2[l].e43_33();
                      flag++;
                    }
                  }
                }
                break;
              }
            }
          }
        }

        coeffs_temp.push_back(inverse(b) * E21E32E43);
      }

      for (int i = 0; i < gArr3_temp.size(); i++)
      {
        for (int j = 0; j < gArr1_temp.size(); j++)
        {
          gArr1.push_back(gArr1_temp[j]);
          gArr2.push_back(gArr2_temp[j]);
          gArr3.push_back(gArr3_temp[i]);
          for (int outer_multi = 0; outer_multi < coeffs.cols(); outer_multi++)
          {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat =
                coeffs_temp[outer_multi];
            if (abs(mat(i, j)) < 1e-10)
              mat(i, j) = 0.0;
            coeffs(gArr1.size() - 1, outer_multi) = mat(i, j);
          }
        }
      }
    }
    return;
  }

} // namespace u4

#endif

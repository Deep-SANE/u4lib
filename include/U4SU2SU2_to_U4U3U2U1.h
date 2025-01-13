#ifndef U4SU2SU2_to_U4U3U2U1_H
#define U4SU2SU2_to_U4U3U2U1_H

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>

#include "U4basis.h"
#include "utils/nullspaceSolver.h"

namespace u4
{

  // construction of the physical basis: the physical basis vectors will be
  // stored in an array
  std::vector<u4::physicalvector> physicalbasis(u4::irrep& u4)
  {
    std::vector<u4::physicalvector> basis;

    int ss, tt, sum; // sum = ss + tt

    // Upper limit for S,T
    int ss_max = u4.n14 + u4.n24 - u4.n34 - u4.n44; // it is also tt_max
    int sum_max = 2 * (u4.n14 - u4.n44);

    // Lower limit for S,T
    int ss_min = ((u4.n14 + u4.n24 - u4.n34 - u4.n44) % 2 == 0) ? 0 : 1;

    for (ss = ss_max; ss >= ss_min; ss -= 2)
    {
      for (tt = ss_max; tt >= ss_min; tt -= 2)
      {
        if ((ss + tt) >= ss_min && (ss + tt) <= sum_max)
        {
          u4::physicalvector physvec(u4, ss, tt);
          if (physvec.zeta == 0)
            continue;
          else
            basis.push_back(physvec);
        }
      }
    }
    return basis;
  }

  /*------------------------------------------------------
  The canonical basis for each (S,T) pair can be expressed by four parameters
  a1,a2,a3,q
  |     n14,n24,n34,n44   \
  |n14-a1, n24-a2, n34-a3  \
  |n14-a1-q,n/2+S-n14+a1+q /
  |   n44+T+S+a1+a2+a3    /
  where
  0 <= a1 <= n14-n24
  0 <= a2 <= n24-n34
  0 <= a3 <= n34-n44
  qmin(S) <= q <= qmax(S,T)
  0 <= a1 + a2 + a3 <= n14 - n44 - T - S
  qmin(S) = max(0, n14+n34-n/2-S-a1-a3)
  qmax(S,T) = min(n14+n24-n/2-S-a1-a2, n14-n44-S-T-2a1-a2-a3,
  n14+n44-n/2+T+a2+a3, n14-n24-a1+a2)
  */

  // Definition of a min function for an array
  int min(int* x, int n)
  {
    int min = x[0];
    for (int i = 1; i < n; i++)
      if (min > x[i])
        min = x[i];
    return min;
  }

  // Definition of a max function for an array
  int max(int* x, int n)
  {
    int max = x[0];
    for (int i = 1; i < n; i++)
      if (max < x[i])
        max = x[i];
    return max;
  }

  // construction of the canonical basis: the canonical basis vectors will be
  // stored in an array
  std::vector<u4::gelfand> canonicalbasis(physicalvector& physvec)
  {
    std::vector<u4::gelfand> gArr;

    int n14 = physvec.toprow.n14, n24 = physvec.toprow.n24,
        n34 = physvec.toprow.n34, n44 = physvec.toprow.n44,
        n = physvec.toprow.n(), ss = physvec.SS, tt = physvec.TT;

    for (int i = 0; i <= (n14 - n24); i++)
    { // i is iterator for a1
      for (int j = 0; j <= (n24 - n34); j++)
      { // j is iterator for a2
        for (int k = 0; k <= (n34 - n44); k++)
        { // k is iterator for a3

          int x[2] = {0, n34 + n14 - (n + ss) / 2 - i - k};
          int qmin = max(x, 2);
          int y[4] = {n24 + n14 - (n + ss) / 2 - i - j,
                      n14 - n44 - (tt + ss) / 2 - 2 * i - j - k,
                      n14 + n44 - (n - tt) / 2 + j + k, n14 - n24 - i + j};
          int qmax = min(y, 4);

          if (qmin > qmax)
          {
            continue;
          }
          else
          {
            for (int q1 = qmin; q1 <= qmax; q1++)
            { // q1 is iterator for q
              // std::cout << i << "," << j << "," << k << "," << q1 <<
              // std::endl;
              u4::gelfand g(n14, n24, n34, n44, n14 - i, n24 - j, n34 - k,
                            n14 - i - q1, (n + ss) / 2 - n14 + i + q1,
                            n44 + (ss + tt) / 2 + i + j + k);
              assert(g.check());
              gArr.push_back(g);
            }
          }
        }
      }
    }
    return gArr;
  }

  // Finding the transformation by solving a system of linear equations resulted
  // from acting S+ = E13 + E24 and T+ = E12 + E34

  void newstate(u4::gelfand& g, std::vector<u4::gelfand>& newgArr)
  {
    if (newgArr.size() == 0)
    {
      newgArr.push_back(g);
    }
    else
    {
      int flag = 0;
      for (int j = 0; j < newgArr.size(); j++)
      {
        if (u4::compare(g, newgArr[j]))
        {
          flag = 1;
          break;
        }
      }
      if (flag == 0)
        newgArr.push_back(g);
    }
    return;
  }

  // Building the projection matrix from the physical basis
  // |[n14,n24,n34,n44],zeta,SS,TT> onto gelfand basis
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  STprojection(u4::physicalvector& physvec, std::vector<u4::gelfand>& gArr)
  {

    std::vector<u4::gelfand> newgArr;

    for (int i = 0; i < gArr.size(); i++)
    {

      // determine the action of E24
      u4::gelfand g1 = gArr[i];
      g1.n13++;
      g1.n12++;
      if (g1.check())
        newstate(g1, newgArr);
      u4::gelfand g2 = gArr[i];
      g2.n13++;
      g2.n22++;
      if (g2.check())
        newstate(g2, newgArr);
      u4::gelfand g3 = gArr[i];
      g3.n23++;
      g3.n12++;
      if (g3.check())
        newstate(g3, newgArr);
      u4::gelfand g4 = gArr[i];
      g4.n23++;
      g4.n22++;
      if (g4.check())
        newstate(g4, newgArr);
      u4::gelfand g5 = gArr[i];
      g5.n33++;
      g5.n12++;
      if (g5.check())
        newstate(g5, newgArr);
      u4::gelfand g6 = gArr[i];
      g6.n33++;
      g6.n22++;
      if (g6.check())
        newstate(g6, newgArr);

      // determine the action of E13
      u4::gelfand g7 = gArr[i];
      g7.n12++;
      g7.n11++;
      if (g7.check())
        newstate(g7, newgArr);
      u4::gelfand g8 = gArr[i];
      g8.n22++;
      g8.n11++;
      if (g8.check())
        newstate(g8, newgArr);

      // determine the action of E34
      u4::gelfand g9 = gArr[i];
      g9.n13++;
      if (g9.check())
        newstate(g9, newgArr);
      u4::gelfand g10 = gArr[i];
      g10.n23++;
      if (g10.check())
        newstate(g10, newgArr);
      u4::gelfand g11 = gArr[i];
      g11.n33++;
      if (g11.check())
        newstate(g11, newgArr);

      // determine the action of E12
      u4::gelfand g12 = gArr[i];
      g12.n11++;
      if (g12.check())
        newstate(g12, newgArr);
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> projector(
        newgArr.size(), gArr.size());

    for (int i = 0; i < projector.rows(); i++)
    {
      for (int j = 0; j < projector.cols(); j++)
      {

        // action of E24
        if (newgArr[i].n13 == (gArr[j].n13 + 1) &&
            newgArr[i].n23 == gArr[j].n23 && newgArr[i].n33 == gArr[j].n33 &&
            newgArr[i].n12 == (gArr[j].n12 + 1) &&
            newgArr[i].n22 == gArr[j].n22 && newgArr[i].n11 == gArr[j].n11)
        {
          projector(i, j) = gArr[j].e24_13_12();
        }
        else if (newgArr[i].n13 == (gArr[j].n13 + 1) &&
                 newgArr[i].n23 == gArr[j].n23 &&
                 newgArr[i].n33 == gArr[j].n33 &&
                 newgArr[i].n12 == gArr[j].n12 &&
                 newgArr[i].n22 == (gArr[j].n22 + 1) &&
                 newgArr[i].n11 == gArr[j].n11)
        {
          projector(i, j) = gArr[j].e24_13_22();
        }
        else if (newgArr[i].n13 == gArr[j].n13 &&
                 newgArr[i].n23 == (gArr[j].n23 + 1) &&
                 newgArr[i].n33 == gArr[j].n33 &&
                 newgArr[i].n12 == (gArr[j].n12 + 1) &&
                 newgArr[i].n22 == gArr[j].n22 && newgArr[i].n11 == gArr[j].n11)
        {
          projector(i, j) = gArr[j].e24_23_12();
        }
        else if (newgArr[i].n13 == gArr[j].n13 &&
                 newgArr[i].n23 == (gArr[j].n23 + 1) &&
                 newgArr[i].n33 == gArr[j].n33 &&
                 newgArr[i].n12 == gArr[j].n12 &&
                 newgArr[i].n22 == (gArr[j].n22 + 1) &&
                 newgArr[i].n11 == gArr[j].n11)
        {
          projector(i, j) = gArr[j].e24_23_22();
        }
        else if (newgArr[i].n13 == gArr[j].n13 &&
                 newgArr[i].n23 == gArr[j].n23 &&
                 newgArr[i].n33 == (gArr[j].n33 + 1) &&
                 newgArr[i].n12 == (gArr[j].n12 + 1) &&
                 newgArr[i].n22 == gArr[j].n22 && newgArr[i].n11 == gArr[j].n11)
        {
          projector(i, j) = gArr[j].e24_33_12();
        }
        else if (newgArr[i].n13 == gArr[j].n13 &&
                 newgArr[i].n23 == gArr[j].n23 &&
                 newgArr[i].n33 == (gArr[j].n33 + 1) &&
                 newgArr[i].n12 == gArr[j].n12 &&
                 newgArr[i].n22 == (gArr[j].n22 + 1) &&
                 newgArr[i].n11 == gArr[j].n11)
        {
          projector(i, j) = gArr[j].e24_33_22();
        }

        // action of E13
        else if (newgArr[i].n13 == gArr[j].n13 &&
                 newgArr[i].n23 == gArr[j].n23 &&
                 newgArr[i].n33 == gArr[j].n33 &&
                 newgArr[i].n12 == (gArr[j].n12 + 1) &&
                 newgArr[i].n22 == gArr[j].n22 &&
                 newgArr[i].n11 == (gArr[j].n11 + 1))
        {
          projector(i, j) = gArr[j].e13_12();
        }
        else if (newgArr[i].n13 == gArr[j].n13 &&
                 newgArr[i].n23 == gArr[j].n23 &&
                 newgArr[i].n33 == gArr[j].n33 &&
                 newgArr[i].n12 == gArr[j].n12 &&
                 newgArr[i].n22 == (gArr[j].n22 + 1) &&
                 newgArr[i].n11 == (gArr[j].n11 + 1))
        {
          projector(i, j) = gArr[j].e13_22();
        }

        // action of E34
        else if (newgArr[i].n13 == (gArr[j].n13 + 1) &&
                 newgArr[i].n23 == gArr[j].n23 &&
                 newgArr[i].n33 == gArr[j].n33 &&
                 newgArr[i].n12 == gArr[j].n12 &&
                 newgArr[i].n22 == gArr[j].n22 && newgArr[i].n11 == gArr[j].n11)
        {
          projector(i, j) = gArr[j].e34_13();
        }
        else if (newgArr[i].n13 == gArr[j].n13 &&
                 newgArr[i].n23 == (gArr[j].n23 + 1) &&
                 newgArr[i].n33 == gArr[j].n33 &&
                 newgArr[i].n12 == gArr[j].n12 &&
                 newgArr[i].n22 == gArr[j].n22 && newgArr[i].n11 == gArr[j].n11)
        {
          projector(i, j) = gArr[j].e34_23();
        }
        else if (newgArr[i].n13 == gArr[j].n13 &&
                 newgArr[i].n23 == gArr[j].n23 &&
                 newgArr[i].n33 == (gArr[j].n33 + 1) &&
                 newgArr[i].n12 == gArr[j].n12 &&
                 newgArr[i].n22 == gArr[j].n22 && newgArr[i].n11 == gArr[j].n11)
        {
          projector(i, j) = gArr[j].e34_33();
        }

        // action of E12
        else if (newgArr[i].n13 == gArr[j].n13 &&
                 newgArr[i].n23 == gArr[j].n23 &&
                 newgArr[i].n33 == gArr[j].n33 &&
                 newgArr[i].n12 == gArr[j].n12 &&
                 newgArr[i].n22 == gArr[j].n22 &&
                 newgArr[i].n11 == (gArr[j].n11 + 1))
        {
          projector(i, j) = gArr[j].e12();
        }

        else
        {
          projector(i, j) = 0.0;
        }
      }
    }
    return projector;
  }

  // Solving the null space of the projection matrix
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  phys_to_cano(u4::physicalvector& physvec,
               Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& STproj)
  {

    // Create a matrix to store the null space
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> nullSpace =
        null_space(STproj);

    // return the multipplicity
    physvec.zeta = nullSpace.cols();

    return nullSpace;
  }

} // namespace u4

#endif

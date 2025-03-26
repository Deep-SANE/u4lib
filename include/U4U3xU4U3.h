#ifndef U4LIB_U4U3U2XU4U3U2_H
#define U4LIB_U4U3U2XU4U3U2_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#ifdef U4LIB_TIME_LOG
#include <chrono>
#include <iostream>
#endif

#ifdef U4LIB_SAVE_MAT
#include <fstream>
#include <iomanip>
#endif

#include <Eigen/Dense>

#ifdef U4LIB_SPARSE_MAT
#include <Eigen/Sparse>
#endif

#include <U3basis.h>
#include <U3decomposition.h>

#include "U4decomposition.h"
#include "so3Coeffs/SO3_CGC.h"
#include "U4basis.h"
#include "utils/nullspaceSolver.h"

extern "C"
{
#include <wigxjpf.h>
}

namespace u4
{
  class WigU4U3_coeffs
  {
    u4::irrep u4a_, u4b_, u4c_;

    std::size_t etamax_;

    std::size_t D1_, D2_, D3_, D4_;

    struct Record
    {
      int p1, q1;
      int kmin, kmax;
      std::size_t first_row;
    };

    std::vector<Record> records_;

    std::vector<int> computed_;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        coeffs_;

    std::size_t record_index(int pa, int qa, int pb, int qb) const
    {
      assert(pa >= 0);
      assert(pa <= u3a_.lam());

      assert(qa >= 0);
      assert(qa <= u3a_.mu());

      assert(pb >= 0);
      assert(pb <= u3b_.lam());

      assert(qb >= 0);
      assert(qb <= u3b_.mu());

      std::size_t index = ((pa * D2_ + qa) * D3_ + pb) * D4_ + qb;

      assert(index < records_.size());

      return index;
    }

    const auto& record(int pa, int qa, int pb, int qb) const
    {
      return records_[record_index(pa, qa, pb, qb)];
    }

    auto& record(int pa, int qa, int pb, int qb)
    {
      return records_[record_index(pa, qa, pb, qb)];
    }

  public:
    struct Accessor
    {
      bool allowed;
      bool computed;
      std::size_t row;
    };

    WigU3U2_coeffs(const u3::irrep& u3a, const u3::irrep& u3b,
                   const u3::irrep& u3c, int rhomax)
        : u3a_(u3a), u3b_(u3b), u3c_(u3c), rhomax_(rhomax)
    {
      assert(rhomax > 0);

      D1_ = u3a.lam() + 1;
      D2_ = u3a.mu() + 1;
      D3_ = u3b.lam() + 1;
      D4_ = u3b.mu() + 1;

      records_.resize(D1_ * D2_ * D3_ * D4_);

      int temp = 2 * (u3c.n13() - 2 * u3c.n23() + u3c.n33());

      std::size_t row = 0;

      for (int pa = 0; pa <= u3a.lam(); pa++)
        for (int qa = 0; qa <= u3a.mu(); qa++)
        {
          u3::gelfand gA(u3a.n13(), u3a.n23(), u3a.n33(), u3a.n23() + pa,
                         u3a.n23() - qa, u3a.n23() - qa);

          for (int pb = 0; pb <= u3b.lam(); pb++)
            for (int qb = 0; qb <= u3b.mu(); qb++)
            {
              u3::gelfand gB(u3b.n13(), u3b.n23(), u3b.n33(), u3b.n23() + pb,
                             u3b.n23() - qb, u3b.n23() - qb);

              int Q = temp - gA.epsilon() - gB.epsilon();
              assert(Q % 3 == 0);
              int R = Q / 3;

              int p1 = (R >= 0) ? R : 0;
              int q1 = (R >= 0) ? 0 : -R;

              // epsilon-selection rule:
              int ub1 = std::min(u3c.lam() - p1, u3c.mu() - q1);

              // Lambda-selection rule:
              int absR = std::abs(R);
              int lb2 = std::abs(gA.Lambdax2() - gB.Lambdax2()) - absR;
              int ub2 = gA.Lambdax2() + gB.Lambdax2() - absR;

              // combine selection rules:
              int kmin = (std::max(0, lb2) + 1) / 2;
              int kmax = std::min(2 * ub1, ub2) / 2;

              record(pa, qa, pb, qb) = {p1, q1, kmin, kmax, row};

              if (kmax >= kmin)
                row += kmax - kmin + 1;
            }
        }

      coeffs_.resize(row, rhomax);
      coeffs_.setZero();

      computed_.resize(row, 0);
    }

    void normalize()
    {
      u3a_.normalize();
      u3b_.normalize();
      u3c_.normalize();
    }

    Accessor access(int n12a, int n22a, int n12b, int n22b, int n12c,
                    int n22c) const
    {
      int pa = n12a - u3a_.n23();
      int qa = u3a_.n23() - n22a;
      int pb = n12b - u3b_.n23();
      int qb = u3b_.n23() - n22b;

      const auto& rec = record(pa, qa, pb, qb);

      if (rec.kmax < rec.kmin)
        return {false, false, 0};

      int k1 = n12c - u3c_.n23() - rec.p1;
      int k2 = u3c_.n23() - n22c - rec.q1;

      if (k1 != k2)
        return {false, false, 0};

      if ((k1 < rec.kmin) || (k1 > rec.kmax))
        return {false, false, 0};

      assert(rec.first_row + k1 < coeffs_.rows());

      std::size_t row = rec.first_row + k1 - rec.kmin;

      return {true, computed_[row] == rhomax_, row};
    }

    double get(const Accessor& acc, int rho) const
    {
      assert(acc.allowed == true);
      assert(acc.computed == true);

      return coeffs_(acc.row, rho);
    }

    void set(const Accessor& acc, int rho, double value)
    {
      assert(acc.allowed == true);

      coeffs_(acc.row, rho) = value;

      computed_[acc.row]++;
    }

    auto get_row(const Accessor& acc) const
    {
      assert(acc.allowed == true);
      assert(acc.computed == true);

      return coeffs_.row(acc.row);
    }

    template <typename T>
    void set_row(const Accessor& acc, T&& values)
    {
      assert(acc.allowed == true);

      coeffs_.row(acc.row) = std::forward<T>(values);

      computed_[acc.row] = rhomax_;
    }

    std::size_t size() const { return coeffs_.rows(); }
  };

  // expansion of a coupled gelfand basis state (u4c) in terms of those
  // uncoupled ones based on the weight selection rule; hence, this expansion is
  // the same for those other coupled states that have the same weight with the
  // state being expanded. Here indices of the uncoupled states are stored.
  std::vector<std::tuple<int, int, int>>
  expand_to_uncoupled(const u4::gelfand& g3,
                      const std::vector<u4::gelfand>& gArr1,
                      const std::vector<u4::gelfand>& gArr2)
  {
    std::vector<std::tuple<int, int, int>> indices;
    for (int i = 0; i < gArr1.size(); i++)
    {
      for (int j = 0; j < gArr2.size(); j++)
      {
        if ((gArr1[i].e33() + gArr2[j].e33()) == g3.e33() &&
            (gArr1[i].e22() + gArr2[j].e22()) == g3.e22() &&
            (gArr1[i].e11() + gArr2[j].e11()) == g3.e11())
        {
          u3::irrep u31()
          indices.emplace_back(i, j);
        }
      }
    }
    return indices;
  }

  // build the matrix representating a system of linear equations resulting from
  // the action three generators E34, E23 and E12 that annihiliate the highest
  // weight state
  auto E34E23E12(const std::vector<u4::gelfand>& gArr1,
              const std::vector<u4::gelfand>& gArr2,
              std::vector<std::pair<int, int>>& indices)
  {
    int col_num = indices.size();

    std::vector<std::pair<int, int>> new_indices; // store the parent states

    for (int k = 0; k < indices.size(); k++)
    {
      int g1 = indices[k].first, g2 = indices[k].second;

      // parent of the first state
      u3::gelfand g1_n11p1 = gArr1[g1];
      g1_n11p1.n11(g1_n11p1.n11() + 1);
      if (g1_n11p1.check())
      {
        auto i = u3::findGelfand(g1_n11p1, gArr1);
        std::pair<int, int> p(i, g2);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      u3::gelfand g1_n12p1 = gArr1[g1];
      g1_n12p1.n12(g1_n12p1.n12() + 1);
      if (g1_n12p1.check())
      {
        auto i = u3::findGelfand(g1_n12p1, gArr1);
        std::pair<int, int> p(i, g2);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      u3::gelfand g1_n22p1 = gArr1[g1];
      g1_n22p1.n22(g1_n22p1.n22() + 1);
      if (g1_n22p1.check())
      {
        auto i = u3::findGelfand(g1_n22p1, gArr1);
        std::pair<int, int> p(i, g2);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      // parent of the second state
      u3::gelfand g2_n11p1 = gArr2[g2];
      g2_n11p1.n11(g2_n11p1.n11() + 1);
      if (g2_n11p1.check())
      {
        auto i = u3::findGelfand(g2_n11p1, gArr2);
        std::pair<int, int> p(g1, i);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      u3::gelfand g2_n12p1 = gArr2[g2];
      g2_n12p1.n12(g2_n12p1.n12() + 1);
      if (g2_n12p1.check())
      {
        auto i = u3::findGelfand(g2_n12p1, gArr2);
        std::pair<int, int> p(g1, i);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      u3::gelfand g2_n22p1 = gArr2[g2];
      g2_n22p1.n22(g2_n22p1.n22() + 1);
      if (g2_n22p1.check())
      {
        auto i = u3::findGelfand(g2_n22p1, gArr2);
        std::pair<int, int> p(g1, i);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }
    }

    // for (const auto& p : new_indices) {
    // std::cout << "(" << p.first << ", " << p.second << ")\n";
    //}

    int row_num = new_indices.size();

#ifdef U3LIB_SPARSE_MAT
    std::vector<Eigen::Triplet<double>> ija;
#else
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat(row_num, col_num);
#endif

    for (int i = 0; i < row_num; i++)
    {
      int new_g1 = new_indices[i].first, new_g2 = new_indices[i].second;
      for (int j = 0; j < col_num; j++)
      {
        int g1 = indices[j].first, g2 = indices[j].second;
        // action of E23
        if (u3::E23_12(gArr1[g1], gArr1[new_g1]) && new_g2 == g2)
#ifdef U3LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr1[g1].e23_12());
#else
          mat(i, j) = gArr1[g1].e23_12();
#endif
        else if (u3::E23_22(gArr1[g1], gArr1[new_g1]) && new_g2 == g2)
#ifdef U3LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr1[g1].e23_22());
#else
          mat(i, j) = gArr1[g1].e23_22();
#endif
        else if (new_g1 == g1 && u3::E23_12(gArr2[g2], gArr2[new_g2]))
#ifdef U3LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr2[g2].e23_12());
#else
          mat(i, j) = gArr2[g2].e23_12();
#endif
        else if (new_g1 == g1 && u3::E23_22(gArr2[g2], gArr2[new_g2]))
#ifdef U3LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr2[g2].e23_22());
#else
          mat(i, j) = gArr2[g2].e23_22();
#endif

        // action of E12
        else if (u3::E12(gArr1[g1], gArr1[new_g1]) && new_g2 == g2)
#ifdef U3LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr1[g1].e12());
#else
          mat(i, j) = gArr1[g1].e12();
#endif
        else if (new_g1 == g1 && u3::E12(gArr2[g2], gArr2[new_g2]))
#ifdef U3LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr2[g2].e12());
#else
          mat(i, j) = gArr2[g2].e12();
#endif

        else
#ifdef U3LIB_SPARSE_MAT
          ;
#else
          mat(i, j) = 0.0;
#endif
      }
    }

#ifdef U3LIB_SPARSE_MAT
    Eigen::SparseMatrix<double> mat(row_num, col_num);
    mat.setFromTriplets(ija.begin(), ija.end());

#ifdef U3LIB_SAVE_MAT
    {
      std::ofstream fmat("mat.ija");

      fmat << row_num << std::endl;
      fmat << col_num << std::endl;
      fmat << ija.size() << std::endl;

      for (const auto& elem : ija)
        fmat << elem.row() << " " << elem.col() << " " << std::setprecision(12)
             << elem.value() << std::endl;
    }
#endif

#endif

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

    for (int n13 = u4.n14(); n13 >= u4.n24(); n13--)
    {
      for (int n23 = u4.n24(); n23 >= u4.n34(); n23--)
      {
        for (int n33 = u4.n34(); n33 >= u4.n44(); n33--)
        {
          for (int n12 = n13; n12 >= n23; n12--)
          {
            for (int n22 = n23; n22 >= n33; n22--)
            {
              for (int n11 = n12; n11 >= n22; n11--)
              {
                if (n11 == w1 && (n13 + n23 + n33 - n12 - n22) == w3 &&
                    (u4.n14() + u4.n24() + u4.n34() + u4.n44() - n13 - n23 - n33) == w4)
                {
                  u4::gelfand g(u4.n14(), u4.n24(), u4.n34(), u4.n44(), n13, n23, n33,
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
              if (u4::E11(g3_n11p1, gArr3[k]))
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
                    if (u4::E11(gArr3[k], gArr3[l]) &&
                        u4::E11(gArr2_temp[j], gArr2[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr1_temp[j], gArr1[l]) &&
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
              if (u4::E11(g3_n12p1, gArr3[k]))
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
                    if (u4::E11(gArr3[k], gArr3[l]) &&
                        u4::E11(gArr2_temp[j], gArr2[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr2_temp[j], gArr2[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr1_temp[j], gArr1[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr1_temp[j], gArr1[l]) &&
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
              if (u4::E11(g3_n22p1, gArr3[k]))
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
                    if (u4::E11(gArr3[k], gArr3[l]) &&
                        u4::E11(gArr2_temp[j], gArr2[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr2_temp[j], gArr2[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr1_temp[j], gArr1[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr1_temp[j], gArr1[l]) &&
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
              if (u4::E11(g3_n13p1, gArr3[k]))
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
                    if (u4::E11(gArr3[k], gArr3[l]) &&
                        u4::E11(gArr2_temp[j], gArr2[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr2_temp[j], gArr2[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr2_temp[j], gArr2[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr1_temp[j], gArr1[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr1_temp[j], gArr1[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr1_temp[j], gArr1[l]) &&
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
              if (u4::E11(g3_n23p1, gArr3[k]))
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
                    if (u4::E11(gArr3[k], gArr3[l]) &&
                        u4::E11(gArr2_temp[j], gArr2[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr2_temp[j], gArr2[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr2_temp[j], gArr2[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr1_temp[j], gArr1[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr1_temp[j], gArr1[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr1_temp[j], gArr1[l]) &&
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
              if (u4::E11(g3_n33p1, gArr3[k]))
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
                    if (u4::E11(gArr3[k], gArr3[l]) &&
                        u4::E11(gArr2_temp[j], gArr2[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr2_temp[j], gArr2[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr2_temp[j], gArr2[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr1_temp[j], gArr1[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr1_temp[j], gArr1[l]) &&
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
                    else if (u4::E11(gArr3[k], gArr3[l]) &&
                             u4::E11(gArr1_temp[j], gArr1[l]) &&
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

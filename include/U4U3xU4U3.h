#ifndef U4LIB_U4U3XU4U3_H
#define U4LIB_U4U3XU4U3_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <tuple>
#include <unordered_map>
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
#include <U3U2xU3U2.h>

#include <so3Coeffs/SO3_CGC.h>

#include <nullspace_solver_qr.h>
#ifdef U4LIB_USE_SLEPC
#include <nullspace_solver_slepc.h>
#endif

#include "U4basis.h"
#include "U4decomposition.h"

namespace u4
{
  class WigU4U3_coeffs
  {
    u4::irrep u4a_, u4b_, u4c_;

    std::size_t etamax_;

    using key_type = std::array<int, 9>;

    struct hasher
    {
      static void hash_combine(std::size_t& seed, int value)
      {
        seed ^=
            std::hash<int>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }

      std::size_t operator()(const key_type& key) const
      {
        std::size_t seed = 0;

        hash_combine(seed, key[0]);
        hash_combine(seed, key[1]);
        hash_combine(seed, key[2]);
        hash_combine(seed, key[3]);
        hash_combine(seed, key[4]);
        hash_combine(seed, key[5]);
        hash_combine(seed, key[6]);
        hash_combine(seed, key[7]);
        hash_combine(seed, key[8]);

        return seed;
      }
    };

    std::unordered_map<key_type, std::size_t, hasher> map_;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        coeffs_;

  public:
    WigU4U3_coeffs(const u4::irrep& u4a, const u4::irrep& u4b,
                   const u4::irrep& u4c, int etamax)
        : u4a_(u4a), u4b_(u4b), u4c_(u4c), etamax_(etamax)
    {
      assert(etamax > 0);

      coeffs_.resize(0, etamax);
    }

    // std::pair<bool, decltype(coeffs_.block(0, 0, 0, 0))>
    auto push(const u3::irrep& u3a, const u3::irrep& u3b, const u3::irrep& u3c)
    {
      auto rhomax = u3::Rho(u3a, u3b, u3c);

      assert(rhomax > 0);

      key_type key = {u3a.n13(), u3a.n23(), u3a.n33(), u3b.n13(), u3b.n23(),
                      u3b.n33(), u3c.n13(), u3c.n23(), u3c.n33()};

      auto [iter, inserted] = map_.try_emplace(key);
      if (inserted)
      {
        iter->second = coeffs_.rows();
        coeffs_.conservativeResize(coeffs_.rows() + rhomax, Eigen::NoChange);
        coeffs_.block(iter->second, 0, rhomax, etamax_).setZero();
      }

      assert(iter->second + rhomax <= coeffs_.rows());

      return std::pair(inserted,
                       coeffs_.block(iter->second, 0, rhomax, etamax_));
    }

    auto access(const u3::irrep& u3a, const u3::irrep& u3b,
                const u3::irrep& u3c) const
    {
      bool exist = false;
      Eigen::MatrixXd X;

      auto rhomax = u3::Rho(u3a, u3b, u3c);
      if (rhomax > 0)
      {
        key_type key = {u3a.n13(), u3a.n23(), u3a.n33(), u3b.n13(), u3b.n23(),
                        u3b.n33(), u3c.n13(), u3c.n23(), u3c.n33()};

        auto iter = map_.find(key);
        if (iter != map_.end())
        {
          X = coeffs_.block(iter->second, 0, rhomax, etamax_);
          exist = true;
        }
      }

      return std::pair(exist, X);
    }

    const auto& matrix() const { return coeffs_; }
  };

  // expansion of a coupled gelfand basis state (u4c) in terms of those
  // uncoupled ones based on the weight selection rule; hence, this expansion is
  // the same for those other coupled states that have the same weight with the
  // state being expanded. Here indices of the uncoupled states are stored.
  inline std::vector<std::pair<int, int>>
  expand_to_uncoupled(const u4::gelfand& g3,
                      const std::vector<u4::gelfand>& gArr1,
                      const std::vector<u4::gelfand>& gArr2)
  {
    int x = gArr1[0].n14() + gArr1[0].n24() + gArr1[0].n34() + gArr2[0].n14() +
            gArr2[0].n24() + gArr2[0].n34() - g3.n14() - g3.n24() - g3.n34();

    assert(x % 4 == 0);

    x /= 4;

    std::vector<std::pair<int, int>> indices;
    for (int i = 0; i < gArr1.size(); i++)
    {
      for (int j = 0; j < gArr2.size(); j++)
      {
        if ((gArr1[i].e33() + gArr2[j].e33()) == (g3.e33() + x) &&
            (gArr1[i].e22() + gArr2[j].e22()) == (g3.e22() + x) &&
            (gArr1[i].e11() + gArr2[j].e11()) == (g3.e11() + x))
        {
          indices.emplace_back(i, j);
        }
      }
    }
    return indices;
  }

  // build the matrix representating a system of linear equations resulting from
  // the action three generators E34, E23 and E12 that annihiliate the highest
  // weight state
  inline auto E34E23E12(const std::vector<u4::gelfand>& gArr1,
                        const std::vector<u4::gelfand>& gArr2,
                        const std::vector<std::pair<int, int>>& indices)
  {
    int col_num = indices.size();

    std::vector<std::pair<int, int>> new_indices; // store the parent states

    for (int k = 0; k < indices.size(); k++)
    {
      int g1 = indices[k].first, g2 = indices[k].second;

      // parent of the first state
      u4::gelfand g1_n11p1 = gArr1[g1];
      g1_n11p1.n11(g1_n11p1.n11() + 1);
      if (g1_n11p1.check())
      {
        auto i = u4::findGelfand(g1_n11p1, gArr1);
        std::pair<int, int> p(i, g2);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      u4::gelfand g1_n12p1 = gArr1[g1];
      g1_n12p1.n12(g1_n12p1.n12() + 1);
      if (g1_n12p1.check())
      {
        auto i = u4::findGelfand(g1_n12p1, gArr1);
        std::pair<int, int> p(i, g2);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      u4::gelfand g1_n22p1 = gArr1[g1];
      g1_n22p1.n22(g1_n22p1.n22() + 1);
      if (g1_n22p1.check())
      {
        auto i = u4::findGelfand(g1_n22p1, gArr1);
        std::pair<int, int> p(i, g2);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      u4::gelfand g1_n13p1 = gArr1[g1];
      g1_n13p1.n13(g1_n13p1.n13() + 1);
      if (g1_n13p1.check())
      {
        auto i = u4::findGelfand(g1_n13p1, gArr1);
        std::pair<int, int> p(i, g2);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      u4::gelfand g1_n23p1 = gArr1[g1];
      g1_n23p1.n23(g1_n23p1.n23() + 1);
      if (g1_n23p1.check())
      {
        auto i = u4::findGelfand(g1_n23p1, gArr1);
        std::pair<int, int> p(i, g2);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      u4::gelfand g1_n33p1 = gArr1[g1];
      g1_n33p1.n33(g1_n33p1.n33() + 1);
      if (g1_n33p1.check())
      {
        auto i = u4::findGelfand(g1_n33p1, gArr1);
        std::pair<int, int> p(i, g2);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      // parent of the second state
      u4::gelfand g2_n11p1 = gArr2[g2];
      g2_n11p1.n11(g2_n11p1.n11() + 1);
      if (g2_n11p1.check())
      {
        auto i = u4::findGelfand(g2_n11p1, gArr2);
        std::pair<int, int> p(g1, i);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      u4::gelfand g2_n12p1 = gArr2[g2];
      g2_n12p1.n12(g2_n12p1.n12() + 1);
      if (g2_n12p1.check())
      {
        auto i = u4::findGelfand(g2_n12p1, gArr2);
        std::pair<int, int> p(g1, i);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      u4::gelfand g2_n22p1 = gArr2[g2];
      g2_n22p1.n22(g2_n22p1.n22() + 1);
      if (g2_n22p1.check())
      {
        auto i = u4::findGelfand(g2_n22p1, gArr2);
        std::pair<int, int> p(g1, i);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      u4::gelfand g2_n13p1 = gArr2[g2];
      g2_n13p1.n13(g2_n13p1.n13() + 1);
      if (g2_n13p1.check())
      {
        auto i = u4::findGelfand(g2_n13p1, gArr2);
        std::pair<int, int> p(g1, i);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      u4::gelfand g2_n23p1 = gArr2[g2];
      g2_n23p1.n23(g2_n23p1.n23() + 1);
      if (g2_n23p1.check())
      {
        auto i = u4::findGelfand(g2_n23p1, gArr2);
        std::pair<int, int> p(g1, i);
        auto it = std::find(new_indices.begin(), new_indices.end(), p);
        if (it == new_indices.end())
          new_indices.push_back(p);
      }

      u4::gelfand g2_n33p1 = gArr2[g2];
      g2_n33p1.n33(g2_n33p1.n33() + 1);
      if (g2_n33p1.check())
      {
        auto i = u4::findGelfand(g2_n33p1, gArr2);
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

#ifdef U4LIB_SPARSE_MAT
    std::vector<Eigen::Triplet<double>> ija;
#else
    Eigen::MatrixXd mat(row_num, col_num);
#endif

    for (int i = 0; i < row_num; i++)
    {
      int new_g1 = new_indices[i].first, new_g2 = new_indices[i].second;
      for (int j = 0; j < col_num; j++)
      {
        int g1 = indices[j].first, g2 = indices[j].second;
        // action of E34
        if (u4::E34_13(gArr1[g1], gArr1[new_g1]) && new_g2 == g2)
#ifdef U4LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr1[g1].e34_13());
#else
          mat(i, j) = gArr1[g1].e34_13();
#endif
        else if (u4::E34_23(gArr1[g1], gArr1[new_g1]) && new_g2 == g2)
#ifdef U4LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr1[g1].e34_23());
#else
          mat(i, j) = gArr1[g1].e34_23();
#endif
        else if (u4::E34_33(gArr1[g1], gArr1[new_g1]) && new_g2 == g2)
#ifdef U4LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr1[g1].e34_33());
#else
          mat(i, j) = gArr1[g1].e34_33();
#endif
        else if (new_g1 == g1 && u4::E34_13(gArr2[g2], gArr2[new_g2]))
#ifdef U4LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr2[g2].e34_13());
#else
          mat(i, j) = gArr2[g2].e34_13();
#endif
        else if (new_g1 == g1 && u4::E34_23(gArr2[g2], gArr2[new_g2]))
#ifdef U4LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr2[g2].e34_23());
#else
          mat(i, j) = gArr2[g2].e34_23();
#endif
        else if (new_g1 == g1 && u4::E34_33(gArr2[g2], gArr2[new_g2]))
#ifdef U4LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr2[g2].e34_33());
#else
          mat(i, j) = gArr2[g2].e34_33();
#endif

        // action of E23
        else if (u4::E23_12(gArr1[g1], gArr1[new_g1]) && new_g2 == g2)
#ifdef U4LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr1[g1].e23_12());
#else
          mat(i, j) = gArr1[g1].e23_12();
#endif
        else if (u4::E23_22(gArr1[g1], gArr1[new_g1]) && new_g2 == g2)
#ifdef U4LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr1[g1].e23_22());
#else
          mat(i, j) = gArr1[g1].e23_22();
#endif
        else if (new_g1 == g1 && u4::E23_12(gArr2[g2], gArr2[new_g2]))
#ifdef U4LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr2[g2].e23_12());
#else
          mat(i, j) = gArr2[g2].e23_12();
#endif
        else if (new_g1 == g1 && u4::E23_22(gArr2[g2], gArr2[new_g2]))
#ifdef U4LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr2[g2].e23_22());
#else
          mat(i, j) = gArr2[g2].e23_22();
#endif

        // action of E12
        else if (u4::E12(gArr1[g1], gArr1[new_g1]) && new_g2 == g2)
#ifdef U4LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr1[g1].e12());
#else
          mat(i, j) = gArr1[g1].e12();
#endif
        else if (new_g1 == g1 && u4::E12(gArr2[g2], gArr2[new_g2]))
#ifdef U4LIB_SPARSE_MAT
          ija.emplace_back(i, j, gArr2[g2].e12());
#else
          mat(i, j) = gArr2[g2].e12();
#endif

        else
#ifdef U4LIB_SPARSE_MAT
          ;
#else
          mat(i, j) = 0.0;
#endif
      }
    }

#ifdef U4LIB_SPARSE_MAT
    Eigen::SparseMatrix<double> mat(row_num, col_num);
    mat.setFromTriplets(ija.begin(), ija.end());

#ifdef U4LIB_SAVE_MAT
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

  // highest-weight reduced coefficients
  // / n14a n24a n34a n44a ; n14b n24b n34b n44b ||| n14c n24c n34c n44c \
  // \   n13a  n23a  n33a      n13b  n23b n33b   |||   n14c  n24c n34c /_eta,rho

  inline void HighestWeightReducedCoeffs(const std::vector<u4::gelfand>& gArr1,
                                         const std::vector<u4::gelfand>& gArr2,
                                         const std::vector<u4::gelfand>& gArr3,
                                         int etamax, u34utils::SO3_CGC& cgcs,
                                         u4::WigU4U3_coeffs& coeffs,
                                         u3::WigU3U2_db& wigU3U2db)
  {
    std::vector<std::pair<int, int>> indices =
        expand_to_uncoupled(gArr3[0], gArr1, gArr2);

    auto mat = E34E23E12(gArr1, gArr2, indices);

#ifdef U4LIB_TIME_LOG
    auto t0 = std::chrono::high_resolution_clock::now();
#endif

#ifdef U4LIB_NULLSPACE_LOG
    std::cout << "mat: " << mat.rows() << " x " << mat.cols() << std::endl;
    std::cout << "\telements: " << mat.rows() * mat.cols() << std::endl;

#ifdef U4LIB_SPARSE_MAT
    long nnz = mat.nonZeros();
#else
    long nnz = 0;
    for (auto x : mat.reshaped())
      if (x != 0.0)
        nnz++;
#endif
    std::cout << "\tnonzero elements: " << nnz << std::endl;
#endif

#ifdef U4LIB_USE_SLEPC
    Eigen::MatrixXd nullspace = u34utils::null_space_slepc(mat, etamax);
#else
    Eigen::MatrixXd nullspace = u34utils::null_space(mat);
#endif

#ifdef U4LIB_NULLSPACE_LOG
    std::cout << "nullspace: " << nullspace.rows() << " x " << nullspace.cols()
              << std::endl;
    std::cout << "\telements: " << nullspace.rows() * nullspace.cols()
              << std::endl;
    nnz = 0;
    for (auto x : nullspace.reshaped())
      if (x != 0.0)
        nnz++;
    std::cout << "\tnonzero elements: " << nnz << std::endl;
#endif

#ifdef U4LIB_TIME_LOG
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = t1 - t0;
    std::cerr << "null_space time: " << duration.count() << " [s]" << std::endl;
#endif
    /*
        for (int i = 0; i < indices.size(); i++)
        {
          std::cout << "< " << gArr1[indices[i].first] << " ; "
                    << gArr2[indices[i].second] << " | "
                    << gArr3[0] << " > " << nullspace.row(i) << std::endl;
        }
    */
    int jj3 = gArr3[0].Lambdax2(), mm3 = gArr3[0].MLambdax2();

    for (int i = 0; i < indices.size(); i++)
    {
      int i1 = indices[i].first, i2 = indices[i].second;

      u3::irrep u3a(gArr1[i1].n13(), gArr1[i1].n23(), gArr1[i1].n33()),
          u3b(gArr2[i2].n13(), gArr2[i2].n23(), gArr2[i2].n33()),
          u3c(gArr3[0].n13(), gArr3[0].n23(), gArr3[0].n33());

      auto rhomax = u3::Rho(u3a, u3b, u3c);

      if (rhomax > 0)
      {
        auto [missing, coeffs_] = coeffs.push(u3a, u3b, u3c);

        if (missing == false)
          continue;

        std::vector<Eigen::RowVectorXd> A_rows, B_rows;

        // vary U(2) and U(1) labels
        for (int j = i; j < indices.size(); j++)
        {
          int i1_ = indices[j].first, i2_ = indices[j].second;
          int jj1 = gArr1[i1_].Lambdax2(), mm1 = gArr1[i1_].MLambdax2(),
              jj2 = gArr2[i2_].Lambdax2(), mm2 = gArr2[i2_].MLambdax2();

          if (gArr1[i1_].n13() == gArr1[i1].n13() &&
              gArr1[i1_].n23() == gArr1[i1].n23() &&
              gArr1[i1_].n33() == gArr1[i1].n33() &&
              gArr2[i2_].n13() == gArr2[i2].n13() &&
              gArr2[i2_].n23() == gArr2[i2].n23() &&
              gArr2[i2_].n33() == gArr2[i2].n33() && (mm1 + mm2) == mm3 &&
              jj3 <= (jj1 + jj2) && jj3 >= std::fabs(jj1 - jj2))
          {
            auto [allowU3, accU3] = wigU3U2db.get(
                gArr1[i1_].n13(), gArr1[i1_].n23(), gArr1[i1_].n33(),
                gArr1[i1_].n12(), gArr1[i1_].n22(), gArr2[i2_].n13(),
                gArr2[i2_].n23(), gArr2[i2_].n33(), gArr2[i2_].n12(),
                gArr2[i2_].n22(), gArr3[0].n13(), gArr3[0].n23(),
                gArr3[0].n33(), gArr3[0].n12(), gArr3[0].n22());

            if (allowU3 == false)
              continue;

            double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

            Eigen::RowVectorXd A_row = accU3 * cgc;
            Eigen::RowVectorXd B_row = nullspace.row(j);

            // not consider vanishing rows in A (otherwise not invertible)
            if (A_row.norm() <= 1e-14)
              continue;

            // only add independent row
            bool shouldAdd = true;
            for (int i = 0; i < A_rows.size(); i++)
            {
              Eigen::MatrixXd M(2, A_row.size());
              M.row(0) = A_row;
              M.row(1) = A_rows[i];

              if (M.fullPivLu().rank() < 2) // dependent
              {
                shouldAdd = false;
                break; // no need to check further
              }
            }

            if (shouldAdd == true)
            {
              A_rows.push_back(A_row);
              B_rows.push_back(B_row);
            }
          }
        }

        Eigen::MatrixXd A(A_rows.size(), rhomax), B(A_rows.size(), etamax);
        for (int i = 0; i < A_rows.size(); ++i)
        {
          A.row(i) = A_rows[i];
          B.row(i) = B_rows[i];
        }

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> factorize;
        factorize.compute(A);
        coeffs_ = factorize.solve(B);
        for (int m = 0; m < coeffs_.rows(); m++)
          for (int n = 0; n < coeffs_.cols(); n++)
            if (std::fabs(coeffs_(m, n)) < 1e-14)
              coeffs_(m, n) = 0.0;
      }
    }

    return;
  }

  // lower-weight reduced coefficients
  // / n14a n24a n34a n44a ; n14b n24b n34b n44b ||| n14c n24c n34c n44c \
  // \   n13a  n23a  n33a      n13b  n23b n33b   |||   n13c  n23c n33c /_eta,rho

  // build with E43
  inline void
  E43(int parent_g3, int& row, int etamax,
      const std::vector<int>& same_weight_gArr3,
      const std::vector<std::pair<int, int>>& uncoupled_same_weight_gArr3,
      const std::vector<u4::gelfand>& gArr1,
      const std::vector<u4::gelfand>& gArr2,
      const std::vector<u4::gelfand>& gArr3, Eigen::MatrixXd& b,
      u34utils::SO3_CGC& cgcs, std::vector<Eigen::MatrixXd>& E43E42E41,
      u4::WigU4U3_coeffs& coeffs, u3::WigU3U2_db& wigU3U2db)
  {
    int jj3 = gArr3[parent_g3].Lambdax2(), mm3 = gArr3[parent_g3].MLambdax2();

    // loop over daughter states
    for (int j = 0; j < same_weight_gArr3.size(); j++)
      if (u4::E34_13(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e43_13();
      else if (u4::E34_23(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e43_23();
      else if (u4::E34_33(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e43_33();
      else
        b(row, j) = 0.0;

    for (int j = 0; j < uncoupled_same_weight_gArr3.size(); j++)
    {
      int g1 = uncoupled_same_weight_gArr3[j].first,
          g2 = uncoupled_same_weight_gArr3[j].second;

      // act on the first state
      u4::gelfand g1_n13p1 = gArr1[g1];
      g1_n13p1.n13(g1_n13p1.n13() + 1);
      if (g1_n13p1.check())
      {
        u3::irrep u3a(g1_n13p1.n13(), g1_n13p1.n23(), g1_n13p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n13p1.Lambdax2(), mm1 = g1_n13p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n13p1.n13(), g1_n13p1.n23(), g1_n13p1.n33(), g1_n13p1.n12(),
              g1_n13p1.n22(), gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(),
              gArr2[g2].n12(), gArr2[g2].n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n13p1.e43_13();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g1_n23p1 = gArr1[g1];
      g1_n23p1.n23(g1_n23p1.n23() + 1);
      if (g1_n23p1.check())
      {
        u3::irrep u3a(g1_n23p1.n13(), g1_n23p1.n23(), g1_n23p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n23p1.Lambdax2(), mm1 = g1_n23p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n23p1.n13(), g1_n23p1.n23(), g1_n23p1.n33(), g1_n23p1.n12(),
              g1_n23p1.n22(), gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(),
              gArr2[g2].n12(), gArr2[g2].n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n23p1.e43_23();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g1_n33p1 = gArr1[g1];
      g1_n33p1.n33(g1_n33p1.n33() + 1);
      if (g1_n33p1.check())
      {
        u3::irrep u3a(g1_n33p1.n13(), g1_n33p1.n23(), g1_n33p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n33p1.Lambdax2(), mm1 = g1_n33p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n33p1.n13(), g1_n33p1.n23(), g1_n33p1.n33(), g1_n33p1.n12(),
              g1_n33p1.n22(), gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(),
              gArr2[g2].n12(), gArr2[g2].n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n33p1.e43_33();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      // act on the second state
      u4::gelfand g2_n13p1 = gArr2[g2];
      g2_n13p1.n13(g2_n13p1.n13() + 1);
      if (g2_n13p1.check())
      {
        u3::irrep u3b(g2_n13p1.n13(), g2_n13p1.n23(), g2_n13p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n13p1.Lambdax2(), mm2 = g2_n13p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] =
              wigU3U2db.get(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
                            gArr1[g1].n12(), gArr1[g1].n22(), g2_n13p1.n13(),
                            g2_n13p1.n23(), g2_n13p1.n33(), g2_n13p1.n12(),
                            g2_n13p1.n22(), gArr3[parent_g3].n13(),
                            gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
                            gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n13p1.e43_13();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g2_n23p1 = gArr2[g2];
      g2_n23p1.n23(g2_n23p1.n23() + 1);
      if (g2_n23p1.check())
      {
        u3::irrep u3b(g2_n23p1.n13(), g2_n23p1.n23(), g2_n23p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n23p1.Lambdax2(), mm2 = g2_n23p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] =
              wigU3U2db.get(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
                            gArr1[g1].n12(), gArr1[g1].n22(), g2_n23p1.n13(),
                            g2_n23p1.n23(), g2_n23p1.n33(), g2_n23p1.n12(),
                            g2_n23p1.n22(), gArr3[parent_g3].n13(),
                            gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
                            gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n23p1.e43_23();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g2_n33p1 = gArr2[g2];
      g2_n33p1.n33(g2_n33p1.n33() + 1);
      if (g2_n33p1.check())
      {
        u3::irrep u3b(g2_n33p1.n13(), g2_n33p1.n23(), g2_n33p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n33p1.Lambdax2(), mm2 = g2_n33p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] =
              wigU3U2db.get(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
                            gArr1[g1].n12(), gArr1[g1].n22(), g2_n33p1.n13(),
                            g2_n33p1.n23(), g2_n33p1.n33(), g2_n33p1.n12(),
                            g2_n33p1.n22(), gArr3[parent_g3].n13(),
                            gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
                            gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n33p1.e43_33();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }
    }
    row++;
    return;
  }

  // build with E42
  inline void
  E42(int parent_g3, int& row, int etamax,
      const std::vector<int>& same_weight_gArr3,
      const std::vector<std::pair<int, int>>& uncoupled_same_weight_gArr3,
      const std::vector<u4::gelfand>& gArr1,
      const std::vector<u4::gelfand>& gArr2,
      const std::vector<u4::gelfand>& gArr3, Eigen::MatrixXd& b,
      u34utils::SO3_CGC& cgcs, std::vector<Eigen::MatrixXd>& E43E42E41,
      u4::WigU4U3_coeffs& coeffs, u3::WigU3U2_db& wigU3U2db)
  {
    int jj3 = gArr3[parent_g3].Lambdax2(), mm3 = gArr3[parent_g3].MLambdax2();

    // loop over daughter states
    for (int j = 0; j < same_weight_gArr3.size(); j++)
      if (u4::E24_13_12(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e42_13_12();
      else if (u4::E24_13_22(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e42_13_22();
      else if (u4::E24_23_12(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e42_23_12();
      else if (u4::E24_23_22(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e42_23_22();
      else if (u4::E24_33_12(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e42_33_12();
      else if (u4::E24_33_22(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e42_33_22();
      else
        b(row, j) = 0.0;

    for (int j = 0; j < uncoupled_same_weight_gArr3.size(); j++)
    {
      int g1 = uncoupled_same_weight_gArr3[j].first,
          g2 = uncoupled_same_weight_gArr3[j].second;

      // act on the first state
      u4::gelfand g1_n13n12p1 = gArr1[g1];
      g1_n13n12p1.n13(g1_n13n12p1.n13() + 1);
      g1_n13n12p1.n12(g1_n13n12p1.n12() + 1);
      if (g1_n13n12p1.check())
      {
        u3::irrep u3a(g1_n13n12p1.n13(), g1_n13n12p1.n23(), g1_n13n12p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n13n12p1.Lambdax2(), mm1 = g1_n13n12p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n13n12p1.n13(), g1_n13n12p1.n23(), g1_n13n12p1.n33(),
              g1_n13n12p1.n12(), g1_n13n12p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n13n12p1.e42_13_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g1_n13n22p1 = gArr1[g1];
      g1_n13n22p1.n13(g1_n13n22p1.n13() + 1);
      g1_n13n22p1.n22(g1_n13n22p1.n22() + 1);
      if (g1_n13n22p1.check())
      {
        u3::irrep u3a(g1_n13n22p1.n13(), g1_n13n22p1.n23(), g1_n13n22p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n13n22p1.Lambdax2(), mm1 = g1_n13n22p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n13n22p1.n13(), g1_n13n22p1.n23(), g1_n13n22p1.n33(),
              g1_n13n22p1.n12(), g1_n13n22p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n13n22p1.e42_13_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g1_n23n12p1 = gArr1[g1];
      g1_n23n12p1.n23(g1_n23n12p1.n23() + 1);
      g1_n23n12p1.n12(g1_n23n12p1.n12() + 1);
      if (g1_n23n12p1.check())
      {
        u3::irrep u3a(g1_n23n12p1.n13(), g1_n23n12p1.n23(), g1_n23n12p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n23n12p1.Lambdax2(), mm1 = g1_n23n12p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n23n12p1.n13(), g1_n23n12p1.n23(), g1_n23n12p1.n33(),
              g1_n23n12p1.n12(), g1_n23n12p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n23n12p1.e42_23_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g1_n23n22p1 = gArr1[g1];
      g1_n23n22p1.n23(g1_n23n22p1.n23() + 1);
      g1_n23n22p1.n22(g1_n23n22p1.n22() + 1);
      if (g1_n23n22p1.check())
      {
        u3::irrep u3a(g1_n23n22p1.n13(), g1_n23n22p1.n23(), g1_n23n22p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n23n22p1.Lambdax2(), mm1 = g1_n23n22p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n23n22p1.n13(), g1_n23n22p1.n23(), g1_n23n22p1.n33(),
              g1_n23n22p1.n12(), g1_n23n22p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n23n22p1.e42_23_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g1_n33n12p1 = gArr1[g1];
      g1_n33n12p1.n33(g1_n33n12p1.n33() + 1);
      g1_n33n12p1.n12(g1_n33n12p1.n12() + 1);
      if (g1_n33n12p1.check())
      {
        u3::irrep u3a(g1_n33n12p1.n13(), g1_n33n12p1.n23(), g1_n33n12p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n33n12p1.Lambdax2(), mm1 = g1_n33n12p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n33n12p1.n13(), g1_n33n12p1.n23(), g1_n33n12p1.n33(),
              g1_n33n12p1.n12(), g1_n33n12p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n33n12p1.e42_33_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g1_n33n22p1 = gArr1[g1];
      g1_n33n22p1.n33(g1_n33n22p1.n33() + 1);
      g1_n33n22p1.n22(g1_n33n22p1.n22() + 1);
      if (g1_n33n22p1.check())
      {
        u3::irrep u3a(g1_n33n22p1.n13(), g1_n33n22p1.n23(), g1_n33n22p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n33n22p1.Lambdax2(), mm1 = g1_n33n22p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n33n22p1.n13(), g1_n33n22p1.n23(), g1_n33n22p1.n33(),
              g1_n33n22p1.n12(), g1_n33n22p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n33n22p1.e42_33_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      // act on the second state
      u4::gelfand g2_n13n12p1 = gArr2[g2];
      g2_n13n12p1.n13(g2_n13n12p1.n13() + 1);
      g2_n13n12p1.n12(g2_n13n12p1.n12() + 1);
      if (g2_n13n12p1.check())
      {
        u3::irrep u3b(g2_n13n12p1.n13(), g2_n13n12p1.n23(), g2_n13n12p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n13n12p1.Lambdax2(), mm2 = g2_n13n12p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n13n12p1.n13(),
              g2_n13n12p1.n23(), g2_n13n12p1.n33(), g2_n13n12p1.n12(),
              g2_n13n12p1.n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n13n12p1.e42_13_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g2_n13n22p1 = gArr2[g2];
      g2_n13n22p1.n13(g2_n13n22p1.n13() + 1);
      g2_n13n22p1.n22(g2_n13n22p1.n22() + 1);
      if (g2_n13n22p1.check())
      {
        u3::irrep u3b(g2_n13n22p1.n13(), g2_n13n22p1.n23(), g2_n13n22p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n13n22p1.Lambdax2(), mm2 = g2_n13n22p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n13n22p1.n13(),
              g2_n13n22p1.n23(), g2_n13n22p1.n33(), g2_n13n22p1.n12(),
              g2_n13n22p1.n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n13n22p1.e42_13_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g2_n23n12p1 = gArr2[g2];
      g2_n23n12p1.n23(g2_n23n12p1.n23() + 1);
      g2_n23n12p1.n12(g2_n23n12p1.n12() + 1);
      if (g2_n23n12p1.check())
      {
        u3::irrep u3b(g2_n23n12p1.n13(), g2_n23n12p1.n23(), g2_n23n12p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n23n12p1.Lambdax2(), mm2 = g2_n23n12p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n23n12p1.n13(),
              g2_n23n12p1.n23(), g2_n23n12p1.n33(), g2_n23n12p1.n12(),
              g2_n23n12p1.n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n23n12p1.e42_23_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g2_n23n22p1 = gArr2[g2];
      g2_n23n22p1.n23(g2_n23n22p1.n23() + 1);
      g2_n23n22p1.n22(g2_n23n22p1.n22() + 1);
      if (g2_n23n22p1.check())
      {
        u3::irrep u3b(g2_n23n22p1.n13(), g2_n23n22p1.n23(), g2_n23n22p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n23n22p1.Lambdax2(), mm2 = g2_n23n22p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n23n22p1.n13(),
              g2_n23n22p1.n23(), g2_n23n22p1.n33(), g2_n23n22p1.n12(),
              g2_n23n22p1.n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n23n22p1.e42_23_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g2_n33n12p1 = gArr2[g2];
      g2_n33n12p1.n33(g2_n33n12p1.n33() + 1);
      g2_n33n12p1.n12(g2_n33n12p1.n12() + 1);
      if (g2_n33n12p1.check())
      {
        u3::irrep u3b(g2_n33n12p1.n13(), g2_n33n12p1.n23(), g2_n33n12p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n33n12p1.Lambdax2(), mm2 = g2_n33n12p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n33n12p1.n13(),
              g2_n33n12p1.n23(), g2_n33n12p1.n33(), g2_n33n12p1.n12(),
              g2_n33n12p1.n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n33n12p1.e42_33_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g2_n33n22p1 = gArr2[g2];
      g2_n33n22p1.n33(g2_n33n22p1.n33() + 1);
      g2_n33n22p1.n22(g2_n33n22p1.n22() + 1);
      if (g2_n33n22p1.check())
      {
        u3::irrep u3b(g2_n33n22p1.n13(), g2_n33n22p1.n23(), g2_n33n22p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n33n22p1.Lambdax2(), mm2 = g2_n33n22p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n33n22p1.n13(),
              g2_n33n22p1.n23(), g2_n33n22p1.n33(), g2_n33n22p1.n12(),
              g2_n33n22p1.n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n33n22p1.e42_33_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }
    }
    row++;
    return;
  }

  // build with E41
  inline void
  E41(int parent_g3, int& row, int etamax,
      const std::vector<int>& same_weight_gArr3,
      const std::vector<std::pair<int, int>>& uncoupled_same_weight_gArr3,
      const std::vector<u4::gelfand>& gArr1,
      const std::vector<u4::gelfand>& gArr2,
      const std::vector<u4::gelfand>& gArr3, Eigen::MatrixXd& b,
      u34utils::SO3_CGC& cgcs, std::vector<Eigen::MatrixXd>& E43E42E41,
      u4::WigU4U3_coeffs& coeffs, u3::WigU3U2_db& wigU3U2db)
  {
    int jj3 = gArr3[parent_g3].Lambdax2(), mm3 = gArr3[parent_g3].MLambdax2();

    // loop over daughter states
    for (int j = 0; j < same_weight_gArr3.size(); j++)
      if (u4::E14_13_12(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e41_13_12();
      else if (u4::E14_13_22(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e41_13_22();
      else if (u4::E14_23_12(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e41_23_12();
      else if (u4::E14_23_22(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e41_23_22();
      else if (u4::E14_33_12(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e41_33_12();
      else if (u4::E14_33_22(gArr3[same_weight_gArr3[j]], gArr3[parent_g3]))
        b(row, j) = gArr3[parent_g3].e41_33_22();
      else
        b(row, j) = 0.0;

    for (int j = 0; j < uncoupled_same_weight_gArr3.size(); j++)
    {
      int g1 = uncoupled_same_weight_gArr3[j].first,
          g2 = uncoupled_same_weight_gArr3[j].second;

      // act on the first state
      u4::gelfand g1_n13n12n11p1 = gArr1[g1];
      g1_n13n12n11p1.n13(g1_n13n12n11p1.n13() + 1);
      g1_n13n12n11p1.n12(g1_n13n12n11p1.n12() + 1);
      g1_n13n12n11p1.n11(g1_n13n12n11p1.n11() + 1);
      if (g1_n13n12n11p1.check())
      {
        u3::irrep u3a(g1_n13n12n11p1.n13(), g1_n13n12n11p1.n23(),
                      g1_n13n12n11p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n13n12n11p1.Lambdax2(), mm1 = g1_n13n12n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n13n12n11p1.n13(), g1_n13n12n11p1.n23(), g1_n13n12n11p1.n33(),
              g1_n13n12n11p1.n12(), g1_n13n12n11p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n13n12n11p1.e41_13_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g1_n13n22n11p1 = gArr1[g1];
      g1_n13n22n11p1.n13(g1_n13n22n11p1.n13() + 1);
      g1_n13n22n11p1.n22(g1_n13n22n11p1.n22() + 1);
      g1_n13n22n11p1.n11(g1_n13n22n11p1.n11() + 1);
      if (g1_n13n22n11p1.check())
      {
        u3::irrep u3a(g1_n13n22n11p1.n13(), g1_n13n22n11p1.n23(),
                      g1_n13n22n11p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n13n22n11p1.Lambdax2(), mm1 = g1_n13n22n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n13n22n11p1.n13(), g1_n13n22n11p1.n23(), g1_n13n22n11p1.n33(),
              g1_n13n22n11p1.n12(), g1_n13n22n11p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n13n22n11p1.e41_13_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g1_n23n12n11p1 = gArr1[g1];
      g1_n23n12n11p1.n23(g1_n23n12n11p1.n23() + 1);
      g1_n23n12n11p1.n12(g1_n23n12n11p1.n12() + 1);
      g1_n23n12n11p1.n11(g1_n23n12n11p1.n11() + 1);
      if (g1_n23n12n11p1.check())
      {
        u3::irrep u3a(g1_n23n12n11p1.n13(), g1_n23n12n11p1.n23(),
                      g1_n23n12n11p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n23n12n11p1.Lambdax2(), mm1 = g1_n23n12n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n23n12n11p1.n13(), g1_n23n12n11p1.n23(), g1_n23n12n11p1.n33(),
              g1_n23n12n11p1.n12(), g1_n23n12n11p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n23n12n11p1.e41_23_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g1_n23n22n11p1 = gArr1[g1];
      g1_n23n22n11p1.n23(g1_n23n22n11p1.n23() + 1);
      g1_n23n22n11p1.n22(g1_n23n22n11p1.n22() + 1);
      g1_n23n22n11p1.n11(g1_n23n22n11p1.n11() + 1);
      if (g1_n23n22n11p1.check())
      {
        u3::irrep u3a(g1_n23n22n11p1.n13(), g1_n23n22n11p1.n23(),
                      g1_n23n22n11p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n23n22n11p1.Lambdax2(), mm1 = g1_n23n22n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n23n22n11p1.n13(), g1_n23n22n11p1.n23(), g1_n23n22n11p1.n33(),
              g1_n23n22n11p1.n12(), g1_n23n22n11p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n23n22n11p1.e41_23_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g1_n33n12n11p1 = gArr1[g1];
      g1_n33n12n11p1.n33(g1_n33n12n11p1.n33() + 1);
      g1_n33n12n11p1.n12(g1_n33n12n11p1.n12() + 1);
      g1_n33n12n11p1.n11(g1_n33n12n11p1.n11() + 1);
      if (g1_n33n12n11p1.check())
      {
        u3::irrep u3a(g1_n33n12n11p1.n13(), g1_n33n12n11p1.n23(),
                      g1_n33n12n11p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n33n12n11p1.Lambdax2(), mm1 = g1_n33n12n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n33n12n11p1.n13(), g1_n33n12n11p1.n23(), g1_n33n12n11p1.n33(),
              g1_n33n12n11p1.n12(), g1_n33n12n11p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n33n12n11p1.e41_33_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g1_n33n22n11p1 = gArr1[g1];
      g1_n33n22n11p1.n33(g1_n33n22n11p1.n33() + 1);
      g1_n33n22n11p1.n22(g1_n33n22n11p1.n22() + 1);
      g1_n33n22n11p1.n11(g1_n33n22n11p1.n11() + 1);
      if (g1_n33n22n11p1.check())
      {
        u3::irrep u3a(g1_n33n22n11p1.n13(), g1_n33n22n11p1.n23(),
                      g1_n33n22n11p1.n33()),
            u3b(gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n33n22n11p1.Lambdax2(), mm1 = g1_n33n22n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              g1_n33n22n11p1.n13(), g1_n33n22n11p1.n23(), g1_n33n22n11p1.n33(),
              g1_n33n22n11p1.n12(), g1_n33n22n11p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g1_n33n22n11p1.e41_33_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      // act on the second state
      u4::gelfand g2_n13n12n11p1 = gArr2[g2];
      g2_n13n12n11p1.n13(g2_n13n12n11p1.n13() + 1);
      g2_n13n12n11p1.n12(g2_n13n12n11p1.n12() + 1);
      g2_n13n12n11p1.n11(g2_n13n12n11p1.n11() + 1);
      if (g2_n13n12n11p1.check())
      {
        u3::irrep u3b(g2_n13n12n11p1.n13(), g2_n13n12n11p1.n23(),
                      g2_n13n12n11p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n13n12n11p1.Lambdax2(), mm2 = g2_n13n12n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n13n12n11p1.n13(),
              g2_n13n12n11p1.n23(), g2_n13n12n11p1.n33(), g2_n13n12n11p1.n12(),
              g2_n13n12n11p1.n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n13n12n11p1.e41_13_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g2_n13n22n11p1 = gArr2[g2];
      g2_n13n22n11p1.n13(g2_n13n22n11p1.n13() + 1);
      g2_n13n22n11p1.n22(g2_n13n22n11p1.n22() + 1);
      g2_n13n22n11p1.n11(g2_n13n22n11p1.n11() + 1);
      if (g2_n13n22n11p1.check())
      {
        u3::irrep u3b(g2_n13n22n11p1.n13(), g2_n13n22n11p1.n23(),
                      g2_n13n22n11p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n13n22n11p1.Lambdax2(), mm2 = g2_n13n22n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n13n22n11p1.n13(),
              g2_n13n22n11p1.n23(), g2_n13n22n11p1.n33(), g2_n13n22n11p1.n12(),
              g2_n13n22n11p1.n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n13n22n11p1.e41_13_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g2_n23n12n11p1 = gArr2[g2];
      g2_n23n12n11p1.n23(g2_n23n12n11p1.n23() + 1);
      g2_n23n12n11p1.n12(g2_n23n12n11p1.n12() + 1);
      g2_n23n12n11p1.n11(g2_n23n12n11p1.n11() + 1);
      if (g2_n23n12n11p1.check())
      {
        u3::irrep u3b(g2_n23n12n11p1.n13(), g2_n23n12n11p1.n23(),
                      g2_n23n12n11p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n23n12n11p1.Lambdax2(), mm2 = g2_n23n12n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n23n12n11p1.n13(),
              g2_n23n12n11p1.n23(), g2_n23n12n11p1.n33(), g2_n23n12n11p1.n12(),
              g2_n23n12n11p1.n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n23n12n11p1.e41_23_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g2_n23n22n11p1 = gArr2[g2];
      g2_n23n22n11p1.n23(g2_n23n22n11p1.n23() + 1);
      g2_n23n22n11p1.n22(g2_n23n22n11p1.n22() + 1);
      g2_n23n22n11p1.n11(g2_n23n22n11p1.n11() + 1);
      if (g2_n23n22n11p1.check())
      {
        u3::irrep u3b(g2_n23n22n11p1.n13(), g2_n23n22n11p1.n23(),
                      g2_n23n22n11p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n23n22n11p1.Lambdax2(), mm2 = g2_n23n22n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n23n22n11p1.n13(),
              g2_n23n22n11p1.n23(), g2_n23n22n11p1.n33(), g2_n23n22n11p1.n12(),
              g2_n23n22n11p1.n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n23n22n11p1.e41_23_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g2_n33n12n11p1 = gArr2[g2];
      g2_n33n12n11p1.n33(g2_n33n12n11p1.n33() + 1);
      g2_n33n12n11p1.n12(g2_n33n12n11p1.n12() + 1);
      g2_n33n12n11p1.n11(g2_n33n12n11p1.n11() + 1);
      if (g2_n33n12n11p1.check())
      {
        u3::irrep u3b(g2_n33n12n11p1.n13(), g2_n33n12n11p1.n23(),
                      g2_n33n12n11p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n33n12n11p1.Lambdax2(), mm2 = g2_n33n12n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n33n12n11p1.n13(),
              g2_n33n12n11p1.n23(), g2_n33n12n11p1.n33(), g2_n33n12n11p1.n12(),
              g2_n33n12n11p1.n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n33n12n11p1.e41_33_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }

      u4::gelfand g2_n33n22n11p1 = gArr2[g2];
      g2_n33n22n11p1.n33(g2_n33n22n11p1.n33() + 1);
      g2_n33n22n11p1.n22(g2_n33n22n11p1.n22() + 1);
      g2_n33n22n11p1.n11(g2_n33n22n11p1.n11() + 1);
      if (g2_n33n22n11p1.check())
      {
        u3::irrep u3b(g2_n33n22n11p1.n13(), g2_n33n22n11p1.n23(),
                      g2_n33n22n11p1.n33()),
            u3a(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33()),
            u3c(gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                gArr3[parent_g3].n33());
        auto [exist, accU4] = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n33n22n11p1.Lambdax2(), mm2 = g2_n33n22n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (exist == true && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto [allowU3, accU3] = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n33n22n11p1.n13(),
              g2_n33n22n11p1.n23(), g2_n33n22n11p1.n33(), g2_n33n22n11p1.n12(),
              g2_n33n22n11p1.n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(allowU3 == true);

          double factor = g2_n33n22n11p1.e41_33_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          for (int e = 0; e < etamax; e++)
          {
            double y = 0.0;
            for (int r = 0; r < accU4.rows(); r++)
              y += accU4(r, e) * accU3(r);

            E43E42E41[e](row, j) += y * cgc * factor;
          }
        }
      }
    }
    row++;
    return;
  }

  // lower-weight reduced coefficients computation
  inline void LowerWeightReducedCoeffs(const std::vector<u4::gelfand>& gArr1,
                                       const std::vector<u4::gelfand>& gArr2,
                                       const std::vector<u4::gelfand>& gArr3,
                                       int etamax, int w3,
                                       u34utils::SO3_CGC& cgcs,
                                       WigU4U3_coeffs& coeffs,
                                       u3::WigU3U2_db wigU3U2db)
  {

    // trace states with same [w3,w2,w1] (z-weight)
    int W1max = gArr3[0].n14(), W1min = gArr3[0].n44(),
        W2max = gArr3[0].n14() + gArr3[0].n24(),
        W2min = gArr3[0].n34() + gArr3[0].n44();

    std::vector<int> same_weight_gArr3;

    for (int W1 = W1max; W1 >= W1min; W1--)
    {
      for (int W2 = W2max; W2 >= W2min; W2--)
      {
        std::vector<int> same_weight_gArr3_temp;

        for (int i = 0; i < gArr3.size(); i++)
          if (gArr3[i].e11() == W1 && gArr3[i].e22() == (W2 - W1) &&
              gArr3[i].e33() == (w3 - W2))
            same_weight_gArr3_temp.push_back(i);

        if (same_weight_gArr3_temp.size() > same_weight_gArr3.size() &&
            gArr3[same_weight_gArr3_temp.back()].n12() ==
                gArr3[same_weight_gArr3_temp.back()].n13() &&
            gArr3[same_weight_gArr3_temp.back()].n22() ==
                gArr3[same_weight_gArr3_temp.back()].n23() &&
            gArr3[same_weight_gArr3_temp.back()].n11() ==
                gArr3[same_weight_gArr3_temp.back()].n13())
          same_weight_gArr3 = same_weight_gArr3_temp;
      }
    }
    /*
        for (int i = 0; i < same_weight_gArr3.size(); i++)
          std::cout << gArr3[same_weight_gArr3[i]] << std::endl;
        std::cout << std::endl;
    */

    if (!same_weight_gArr3.empty())
    {
      // the coupled states having the same weight have the same expansion of
      // uncoupled states, whose indices are recorded
      std::vector<std::pair<int, int>> uncoupled_same_weight_gArr3 =
          expand_to_uncoupled(gArr3[same_weight_gArr3[0]], gArr1, gArr2);

      // find parent states and operators to get to lower weight states
      // (daughter states)
      std::vector<std::pair<int, int>> actOnParent;

      // loop over daughter states
      for (int i = 0; i < same_weight_gArr3.size(); i++)
      {
        // action of E43
        u4::gelfand g3_n13p1 = gArr3[same_weight_gArr3[i]];
        g3_n13p1.n13(g3_n13p1.n13() + 1);
        if (g3_n13p1.check())
        {
          auto j = u4::findGelfand(g3_n13p1, gArr3);
          std::pair<int, int> p(0, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }

        u4::gelfand g3_n23p1 = gArr3[same_weight_gArr3[i]];
        g3_n23p1.n23(g3_n23p1.n23() + 1);
        if (g3_n23p1.check())
        {
          auto j = u4::findGelfand(g3_n23p1, gArr3);
          std::pair<int, int> p(0, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }

        u4::gelfand g3_n33p1 = gArr3[same_weight_gArr3[i]];
        g3_n33p1.n33(g3_n33p1.n33() + 1);
        if (g3_n33p1.check())
        {
          auto j = u4::findGelfand(g3_n33p1, gArr3);
          std::pair<int, int> p(0, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }

        // action of E42
        u4::gelfand g3_n13n12p1 = gArr3[same_weight_gArr3[i]];
        g3_n13n12p1.n13(g3_n13n12p1.n13() + 1);
        g3_n13n12p1.n12(g3_n13n12p1.n12() + 1);
        if (g3_n13n12p1.check())
        {
          auto j = u4::findGelfand(g3_n13n12p1, gArr3);
          std::pair<int, int> p(1, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }

        u4::gelfand g3_n13n22p1 = gArr3[same_weight_gArr3[i]];
        g3_n13n22p1.n13(g3_n13n22p1.n13() + 1);
        g3_n13n22p1.n22(g3_n13n22p1.n22() + 1);
        if (g3_n13n22p1.check())
        {
          auto j = u4::findGelfand(g3_n13n22p1, gArr3);
          std::pair<int, int> p(1, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }

        u4::gelfand g3_n23n12p1 = gArr3[same_weight_gArr3[i]];
        g3_n23n12p1.n23(g3_n23n12p1.n23() + 1);
        g3_n23n12p1.n12(g3_n23n12p1.n12() + 1);
        if (g3_n23n12p1.check())
        {
          auto j = u4::findGelfand(g3_n23n12p1, gArr3);
          std::pair<int, int> p(1, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }

        u4::gelfand g3_n23n22p1 = gArr3[same_weight_gArr3[i]];
        g3_n23n22p1.n23(g3_n23n22p1.n23() + 1);
        g3_n23n22p1.n22(g3_n23n22p1.n22() + 1);
        if (g3_n23n22p1.check())
        {
          auto j = u4::findGelfand(g3_n23n22p1, gArr3);
          std::pair<int, int> p(1, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }

        u4::gelfand g3_n33n12p1 = gArr3[same_weight_gArr3[i]];
        g3_n33n12p1.n33(g3_n33n12p1.n33() + 1);
        g3_n33n12p1.n12(g3_n33n12p1.n12() + 1);
        if (g3_n33n12p1.check())
        {
          auto j = u4::findGelfand(g3_n33n12p1, gArr3);
          std::pair<int, int> p(1, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }

        u4::gelfand g3_n33n22p1 = gArr3[same_weight_gArr3[i]];
        g3_n33n22p1.n33(g3_n33n22p1.n33() + 1);
        g3_n33n22p1.n22(g3_n33n22p1.n22() + 1);
        if (g3_n33n22p1.check())
        {
          auto j = u4::findGelfand(g3_n33n22p1, gArr3);
          std::pair<int, int> p(1, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }

        // action of E41
        u4::gelfand g3_n13n12n11p1 = gArr3[same_weight_gArr3[i]];
        g3_n13n12n11p1.n13(g3_n13n12n11p1.n13() + 1);
        g3_n13n12n11p1.n12(g3_n13n12n11p1.n12() + 1);
        g3_n13n12n11p1.n11(g3_n13n12n11p1.n11() + 1);
        if (g3_n13n12n11p1.check())
        {
          auto j = u4::findGelfand(g3_n13n12n11p1, gArr3);
          std::pair<int, int> p(2, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }

        u4::gelfand g3_n13n22n11p1 = gArr3[same_weight_gArr3[i]];
        g3_n13n22n11p1.n13(g3_n13n22n11p1.n13() + 1);
        g3_n13n22n11p1.n22(g3_n13n22n11p1.n22() + 1);
        g3_n13n22n11p1.n11(g3_n13n22n11p1.n11() + 1);
        if (g3_n13n22n11p1.check())
        {
          auto j = u4::findGelfand(g3_n13n22n11p1, gArr3);
          std::pair<int, int> p(2, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }

        u4::gelfand g3_n23n12n11p1 = gArr3[same_weight_gArr3[i]];
        g3_n23n12n11p1.n23(g3_n23n12n11p1.n23() + 1);
        g3_n23n12n11p1.n12(g3_n23n12n11p1.n12() + 1);
        g3_n23n12n11p1.n11(g3_n23n12n11p1.n11() + 1);
        if (g3_n23n12n11p1.check())
        {
          auto j = u4::findGelfand(g3_n23n12n11p1, gArr3);
          std::pair<int, int> p(2, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }

        u4::gelfand g3_n23n22n11p1 = gArr3[same_weight_gArr3[i]];
        g3_n23n22n11p1.n23(g3_n23n22n11p1.n23() + 1);
        g3_n23n22n11p1.n22(g3_n23n22n11p1.n22() + 1);
        g3_n23n22n11p1.n11(g3_n23n22n11p1.n11() + 1);
        if (g3_n23n22n11p1.check())
        {
          auto j = u4::findGelfand(g3_n23n22n11p1, gArr3);
          std::pair<int, int> p(2, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }

        u4::gelfand g3_n33n12n11p1 = gArr3[same_weight_gArr3[i]];
        g3_n33n12n11p1.n33(g3_n33n12n11p1.n33() + 1);
        g3_n33n12n11p1.n12(g3_n33n12n11p1.n12() + 1);
        g3_n33n12n11p1.n11(g3_n33n12n11p1.n11() + 1);
        if (g3_n33n12n11p1.check())
        {
          auto j = u4::findGelfand(g3_n33n12n11p1, gArr3);
          std::pair<int, int> p(2, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }

        u4::gelfand g3_n33n22n11p1 = gArr3[same_weight_gArr3[i]];
        g3_n33n22n11p1.n33(g3_n33n22n11p1.n33() + 1);
        g3_n33n22n11p1.n22(g3_n33n22n11p1.n22() + 1);
        g3_n33n22n11p1.n11(g3_n33n22n11p1.n11() + 1);
        if (g3_n33n22n11p1.check())
        {
          auto j = u4::findGelfand(g3_n33n22n11p1, gArr3);
          std::pair<int, int> p(2, j);
          // std::cout << "p = " << p.first << std::endl;
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
            actOnParent.push_back(p);
        }
      }

      // matrix b is independent of the outer multiplicity
      Eigen::MatrixXd b(actOnParent.size(), same_weight_gArr3.size());

      // the lowering matrix from the action of E43, E42 and E41, which depends
      // on the outer multiplicity
      std::vector<Eigen::MatrixXd> E43E42E41(etamax);

      // set matrix elements to zero
      for (int e = 0; e < etamax; e++)
      {
        E43E42E41[e] = Eigen::MatrixXd(actOnParent.size(),
                                       uncoupled_same_weight_gArr3.size());
        E43E42E41[e].setZero();
      }

      int row = 0;
      // loop over daughter states to build these matrices
      for (int i = 0; i < actOnParent.size(); i++)
      {
        int opt = actOnParent[i].first, parent_g3 = actOnParent[i].second;

        if (opt == 0)
        {
          E43(parent_g3, row, etamax, same_weight_gArr3,
              uncoupled_same_weight_gArr3, gArr1, gArr2, gArr3, b, cgcs,
              E43E42E41, coeffs, wigU3U2db);
        }

        else if (opt == 1)
        {
          E42(parent_g3, row, etamax, same_weight_gArr3,
              uncoupled_same_weight_gArr3, gArr1, gArr2, gArr3, b, cgcs,
              E43E42E41, coeffs, wigU3U2db);
        }

        else if (opt == 2)
        {
          E41(parent_g3, row, etamax, same_weight_gArr3,
              uncoupled_same_weight_gArr3, gArr1, gArr2, gArr3, b, cgcs,
              E43E42E41, coeffs, wigU3U2db);
        }
      }

      Eigen::ColPivHouseholderQR<Eigen::MatrixXd> factorize;
      factorize.compute(b);

      // std::cout << b << std::endl;

      std::vector<Eigen::MatrixXd> coeffs_temp(etamax);

      for (int e = 0; e < etamax; e++)
      {
        // std::cout << E43E42E41[e] << std::endl;
        coeffs_temp[e] = factorize.solve(E43E42E41[e]);
        // std::cout << coeffs_temp[e] << std::endl;
      }

      // get the reduced coefficients
      for (int i = 0; i < same_weight_gArr3.size(); i++)
      {
        int i3 = same_weight_gArr3[i]; // fix the third state
        int jj3 = gArr3[i3].Lambdax2(), mm3 = gArr3[i3].MLambdax2();

        // going through expansion
        for (int j = 0; j < uncoupled_same_weight_gArr3.size(); j++)
        {
          int i1 = uncoupled_same_weight_gArr3[j].first,
              i2 = uncoupled_same_weight_gArr3[j].second;

          u3::irrep u3a(gArr1[i1].n13(), gArr1[i1].n23(), gArr1[i1].n33()),
              u3b(gArr2[i2].n13(), gArr2[i2].n23(), gArr2[i2].n33()),
              u3c(gArr3[i3].n13(), gArr3[i3].n23(), gArr3[i3].n33());

          auto rhomax = u3::Rho(u3a, u3b, u3c);

          if (rhomax > 0)
          {
            auto [missing, coeffs_] = coeffs.push(u3a, u3b, u3c);

            if (missing == false)
              continue;

            std::vector<Eigen::RowVectorXd> A_rows, B_rows;

            // vary U(2) and U(1) labels
            for (int k = j; k < uncoupled_same_weight_gArr3.size(); k++)
            {
              int i1_ = uncoupled_same_weight_gArr3[k].first,
                  i2_ = uncoupled_same_weight_gArr3[k].second;

              int jj1 = gArr1[i1_].Lambdax2(), mm1 = gArr1[i1_].MLambdax2(),
                  jj2 = gArr2[i2_].Lambdax2(), mm2 = gArr2[i2_].MLambdax2();

              if (gArr1[i1_].n13() == gArr1[i1].n13() &&
                  gArr1[i1_].n23() == gArr1[i1].n23() &&
                  gArr1[i1_].n33() == gArr1[i1].n33() &&
                  gArr2[i2_].n13() == gArr2[i2].n13() &&
                  gArr2[i2_].n23() == gArr2[i2].n23() &&
                  gArr2[i2_].n33() == gArr2[i2].n33() && (mm1 + mm2) == mm3 &&
                  jj3 <= (jj1 + jj2) && jj3 >= std::fabs(jj1 - jj2))
              {
                auto [allowU3, accU3] = wigU3U2db.get(
                    gArr1[i1_].n13(), gArr1[i1_].n23(), gArr1[i1_].n33(),
                    gArr1[i1_].n12(), gArr1[i1_].n22(), gArr2[i2_].n13(),
                    gArr2[i2_].n23(), gArr2[i2_].n33(), gArr2[i2_].n12(),
                    gArr2[i2_].n22(), gArr3[i3].n13(), gArr3[i3].n23(),
                    gArr3[i3].n33(), gArr3[i3].n12(), gArr3[i3].n22());

                if (allowU3 == false)
                  continue;

                double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

                Eigen::RowVectorXd A_row = accU3 * cgc;
                Eigen::RowVectorXd B_row(etamax);
                for (int e = 0; e < etamax; e++)
                  B_row(e) = coeffs_temp[e](i, k);
                /*
                                std::cout << gArr1[i1_] << " ; " << gArr2[i2_]
                   << " | " << gArr3[i3] << " -> "
                                          << B_row << std::endl;
                */
                // not consider vanishing rows in A (otherwise not invertible)
                if (A_row.norm() <= 1e-14)
                  continue;

                // only add independent row
                bool shouldAdd = true;
                for (int i = 0; i < A_rows.size(); i++)
                {
                  Eigen::MatrixXd M(2, A_row.size());
                  M.row(0) = A_row;
                  M.row(1) = A_rows[i];

                  if (M.fullPivLu().rank() < 2) // dependent
                  {
                    shouldAdd = false;
                    break; // no need to check further
                  }
                }

                if (shouldAdd == true)
                {
                  A_rows.push_back(A_row);
                  B_rows.push_back(B_row);
                }
              }
            }

            Eigen::MatrixXd A(A_rows.size(), rhomax), B(A_rows.size(), etamax);
            for (int i = 0; i < A_rows.size(); ++i)
            {
              A.row(i) = A_rows[i];
              B.row(i) = B_rows[i];
            }

            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> factorize;
            factorize.compute(A);
            coeffs_ = factorize.solve(B);
            for (int m = 0; m < coeffs_.rows(); m++)
              for (int n = 0; n < coeffs_.cols(); n++)
                if (std::fabs(coeffs_(m, n)) < 1e-14)
                  coeffs_(m, n) = 0.0;

            // std::cout << A << ";\n" << B << ";\n" << coeffs_ << "\n" <<
            // std::endl;
          }
        }
      }
    }

    return;
  }

  // all reduced coupling coefficients
  inline auto WigU4U3(const u4::irrep& u4A, const u4::irrep& u4B,
                      const u4::irrep& u4C, int etamax, int w3min,
                      const std::vector<u4::gelfand>& gArrA,
                      const std::vector<u4::gelfand>& gArrB,
                      const std::vector<u4::gelfand>& gArrC,
                      u34utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db)
  {
    WigU4U3_coeffs coeffs(u4A, u4B, u4C, etamax);

    u4::HighestWeightReducedCoeffs(gArrA, gArrB, gArrC, etamax, cgcs, coeffs,
                                   wigU3U2db);

#ifdef U4LIB_TIME_LOG
    auto t0 = std::chrono::high_resolution_clock::now();
#endif

    int w3max = u4C.n14() + u4C.n24() + u4C.n34();
    if (w3min < w3max)
    {
      if (w3min == -1) // calculate all
        w3min = u4C.n24() + u4C.n34() + u4C.n44();

      for (int w3 = w3max - 1; w3 >= w3min; w3--)
      {
        u4::LowerWeightReducedCoeffs(gArrA, gArrB, gArrC, etamax, w3, cgcs,
                                     coeffs, wigU3U2db);
      }
    }

#ifdef U4LIB_TIME_LOG
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = t1 - t0;
    std::cerr << "LowerWeightReducedCoeffs total time: " << duration.count()
              << " [s]" << std::endl;
#endif

    assert(coeffs.matrix().hasNaN() == false);

    return coeffs;
  }

#define U4LIB_ENABLE_WIGU4U3_CACHE
  class WigU4U3_db
  {
#ifdef U4LIB_ENABLE_WIGU4U3_CACHE
    using key_type = std::array<int, 9>;
    std::map<key_type, WigU4U3_coeffs> map_;
#endif

    u34utils::SO3_CGC& cgcs_;
    u3::WigU3U2_db& wigU3U2db_;

    auto calculate(const irrep& u4a, const irrep& u4b, const irrep& u4c)
    {
      auto etamax = Eta(u4a, u4b, u4c);

      auto gArrA = u4::gelfandBasis(u4a);
      auto gArrB = u4::gelfandBasis(u4b);
      auto gArrC = u4::gelfandBasis(u4c);

      return WigU4U3(u4a, u4b, u4c, etamax, -1, gArrA, gArrB, gArrC, cgcs_,
                     wigU3U2db_);
    }

  public:
    WigU4U3_db(u34utils::SO3_CGC& cgcs, u3::WigU3U2_db& wigU3U2db)
        : cgcs_(cgcs), wigU3U2db_(wigU3U2db)
    { }

    auto get(int n14a, int n24a, int n34a, int n44a, int n13a, int n23a,
             int n33a, int n14b, int n24b, int n34b, int n44b, int n13b,
             int n23b, int n33b, int n14c, int n24c, int n34c, int n44c,
             int n13c, int n23c, int n33c)
    {
      // normalized irreps:
      irrep u4a(n14a - n44a, n24a - n44a, n34a - n44a, 0);
      irrep u4b(n14b - n44b, n24b - n44b, n34b - n44b, 0);
      irrep u4c(n14c - n44c, n24c - n44c, n34c - n44c, 0);

      u3::irrep u3a(n13a - n44a, n23a - n44a, n33a - n44a);
      u3::irrep u3b(n13b - n44b, n23b - n44b, n33b - n44b);
      u3::irrep u3c(n13c - n44c, n23c - n44c, n33c - n44c);

      int etamax = Eta(u4a, u4b, u4c);
      int rhomax = u3::Rho(u3a, u3b, u3c);

      bool exist = false;

      Eigen::MatrixXd matrix;

      if (etamax > 0)
      {
#ifdef U4LIB_ENABLE_WIGU4U3_CACHE
        key_type key = {u4a.n14(), u4a.n24(), u4a.n34(), u4b.n14(), u4b.n24(),
                        u4b.n34(), u4c.n14(), u4c.n24(), u4c.n34()};

        auto iter = map_.find(key);
        if (iter == map_.end())
        {
          auto p = map_.try_emplace(key, calculate(u4a, u4b, u4c));
          assert(p.second);
          iter = p.first;
        }

        const auto& coeffs_ref = iter->second;
#else
        auto coeffs = calculate(u4a, u4b, u4c);
        const auto& coeffs_ref = coeffs;
#endif
        std::tie(exist, matrix) = coeffs_ref.access(u3a, u3b, u3c);

        if (exist)
        {
          assert(matrix.rows() == rhomax);
          assert(matrix.cols() == etamax);
        }
      }

      return std::pair(exist, matrix);
    }
  };
} // namespace u4

#endif

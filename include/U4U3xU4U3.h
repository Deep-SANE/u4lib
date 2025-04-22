#ifndef U4LIB_U4U3U2XU4U3U2_H
#define U4LIB_U4U3U2XU4U3U2_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <tuple>
#include <utility>
#include <unordered_map>
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

#include "U4decomposition.h"
#include "so3Coeffs/SO3_CGC.h"
#include "U4basis.h"
//#include "utils/nullspaceSolver.h"

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
    auto access(const u3::irrep& u3a, const u3::irrep& u3b,
                const u3::irrep& u3c)
    {
      auto rhomax = u3::Rho(u3a, u3b, u3c);

      assert(rhomax > 0);

      key_type key = {u3a.n13(), u3a.n23(), u3a.n33(), u3b.n13(), u3b.n23(),
                      u3b.n33(), u3c.n13(), u3c.n23(), u3c.n33()};

      std::size_t first_row;

      auto [iter, inserted] = map_.try_emplace(key);
      if (inserted)
      {
        first_row = coeffs_.rows();
        coeffs_.conservativeResize(coeffs_.rows() + rhomax, Eigen::NoChange);
        iter->second = first_row;
      }
      else
        first_row = iter->second;

      assert(first_row + rhomax <= coeffs_.rows());

      return std::pair(inserted, coeffs_.block(first_row, 0, rhomax, etamax_));
    }
  };

  // expansion of a coupled gelfand basis state (u4c) in terms of those
  // uncoupled ones based on the weight selection rule; hence, this expansion is
  // the same for those other coupled states that have the same weight with the
  // state being expanded. Here indices of the uncoupled states are stored.
  std::vector<std::pair<int, int>>
  expand_to_uncoupled(const u4::gelfand& g3,
                      const std::vector<u4::gelfand>& gArr1,
                      const std::vector<u4::gelfand>& gArr2)
  {
    std::vector<std::pair<int, int>> indices;
    for (int i = 0; i < gArr1.size(); i++)
    {
      for (int j = 0; j < gArr2.size(); j++)
      {
        if ((gArr1[i].e33() + gArr2[j].e33()) == g3.e33() &&
            (gArr1[i].e22() + gArr2[j].e22()) == g3.e22() &&
            (gArr1[i].e11() + gArr2[j].e11()) == g3.e11())
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
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat(row_num, col_num);
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
  // \   n13a  n23a  n33a      n13b  n23b n33b   |||   n14c  n24c n34c   /_eta,rho

  void HighestWeightReducedCoeffs(const std::vector<u4::gelfand>& gArr1,
                                  const std::vector<u4::gelfand>& gArr2,
                                  const std::vector<u4::gelfand>& gArr3,
                                  //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& coeffs)
                                  u4::SO3_CGC& cgcs, WigU4U3_coeffs& coeffs)
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

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> nullspace =
        null_space(mat);

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

    for (int i = 0; i < indices.size(); i++)
    {
      int i1 = indices[i].first, i2 = indices[i].second;

      auto acc = coeffs.access(gArr1[i1].n13(), gArr1[i1].n23(), gArr1[i1].n33(),
                               gArr2[i2].n13(), gArr2[i2].n23(), gArr2[i2].n33(),
                               gArr3[0].n13(), gArr3[0].n23(), gArr3[0].n33());

      if (acc.computed == false)
      {
        int jj1 = gArr1[i1].Lambdax2(), mm1 = gArr1[i1].MLambdax2(),
            jj2 = gArr2[i2].Lambdax2(), mm2 = gArr2[i2].MLambdax2(),
            jj3 = gArr3[0].Lambdax2(), mm3 = gArr3[0].MLambdax2();

        u3::irrep u3_1(gArr1[i1].n13() - gArr1[i1].n33(), 
                       gArr1[i1].n23() - gArr1[i1].n33(), 0),
                  u3_2(gArr2[i2].n13() - gArr2[i2].n33(), 
                       gArr2[i2].n23() - gArr2[i2].n33(), 0),
                  u3_3(gArr3[0].n13() - gArr1[i1].n33() - gArr2[i2].n33(), 
                       gArr3[0].n23() - gArr1[i1].n33() - gArr2[i2].n33(), 
                       gArr3[0].n33() - gArr1[i1].n33() - gArr2[i2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[0].n12() + gArr3[0].n22() 
                   - 2 * gArr1[i1].n33() - 2 * gArr2[i2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[i1].n12() - gArr1[i1].n33(), 
                                      gArr1[i1].n22() - gArr1[i1].n33(), 
                                      gArr2[i2].n12() - gArr2[i2].n33(), 
                                      gArr2[i2].n22() - gArr2[i2].n33(), 
                                      gArr3[0].n12() - gArr3[0].n33(), 
                                      gArr3[0].n22() - gArr3[0].n33());

          Eigen::MatrixXd A(rhomax, rhomax), B(rhomax, etamax);

          int r = 0;

          for (int j = i; j < indices.size(); j++)
          {
            if ()
            r++;
            if (r == rhomax)
              break;
          }

          coeffs.set_row(acc, nullspace.row(i) / cgc);
        }
      }
    }

    return;
  }

  // lower-weight reduced coefficients
  // / n14a n24a n34a n44a ; n14b n24b n34b n44b ||| n14c n24c n34c n44c \
  // \   n13a  n23a  n33a      n13b  n23b n33b   |||   n13c  n23c n33c   /_eta,rho

  // build with E43
  void E43(int parent_g3, int& row, int etamax,
           std::vector<int>& same_weight_gArr3,
           std::vector<std::pair<int, int>>& uncoupled_same_weight_gArr3,
           const std::vector<u4::gelfand>& gArr1,
           const std::vector<u4::gelfand>& gArr2,
           const std::vector<u4::gelfand>& gArr3, Eigen::MatrixXd& b,
           u4::SO3_CGC& cgcs, std::vector<Eigen::MatrixXd>& E43E42E41,
           WigU4U3_coeffs& coeffs)
  {
    int jj3 = gArr3[parent_g3].Lambdax2(), 
        mm3 = gArr3[parent_g3].MLambdax2();

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
        int jj1 = g1_n13p1.Lambdax2(), mm1 = g1_n13p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n13p1.n13() - g1_n13p1.n33(), 
                       g1_n13p1.n23() - g1_n13p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n13p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n13p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n13p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n13p1.e43_13();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n13p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n13p1.n12() - g1_n13p1.n33(), 
                                      g1_n13p1.n22() - g1_n13p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n13p1.n13(), g1_n13p1.n23(), g1_n13p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g1_n23p1 = gArr1[g1];
      g1_n23p1.n23(g1_n23p1.n23() + 1);
      if (g1_n23p1.check())
      {
        int jj1 = g1_n23p1.Lambdax2(), mm1 = g1_n23p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n23p1.n13() - g1_n23p1.n33(), 
                       g1_n23p1.n23() - g1_n23p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n23p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n23p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n23p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n23p1.e43_23();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n23p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n23p1.n12() - g1_n23p1.n33(), 
                                      g1_n23p1.n22() - g1_n23p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n23p1.n13(), g1_n23p1.n23(), g1_n23p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g1_n33p1 = gArr1[g1];
      g1_n33p1.n33(g1_n33p1.n33() + 1);
      if (g1_n33p1.check())
      {
        int jj1 = g1_n33p1.Lambdax2(), mm1 = g1_n33p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n33p1.n13() - g1_n33p1.n33(), 
                       g1_n33p1.n23() - g1_n33p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n33p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n33p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n33p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n33p1.e43_33();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n33p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n33p1.n12() - g1_n33p1.n33(), 
                                      g1_n33p1.n22() - g1_n33p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n33p1.n13(), g1_n33p1.n23(), g1_n33p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      // act on the second state
      u3::gelfand g2_n13p1 = gArr2[g2];
      g2_n13p1.n13(g2_n13p1.n13() + 1);
      if (g2_n13p1.check())
      {
        int jj2 = g2_n13p1.Lambdax2(), mm2 = g2_n13p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n13p1.n13() - g2_n13p1.n33(), 
                       g2_n13p1.n23() - g2_n13p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n13p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n13p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n13p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n13p1.e43_13();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n13p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n13p1.n12() - g2_n13p1.n33(), 
                                      g2_n13p1.n22() - g2_n13p1.n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
                                     g2_n13p1.n13(), g2_n13p1.n23(), g2_n13p1.n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u3::gelfand g2_n23p1 = gArr2[g2];
      g2_n23p1.n23(g2_n23p1.n23() + 1);
      if (g2_n23p1.check())
      {
        int jj2 = g2_n23p1.Lambdax2(), mm2 = g2_n23p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n23p1.n13() - g2_n23p1.n33(), 
                       g2_n23p1.n23() - g2_n23p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n23p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n23p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n23p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n23p1.e43_23();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n23p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n23p1.n12() - g2_n23p1.n33(), 
                                      g2_n23p1.n22() - g2_n23p1.n33(),  
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
                                     g2_n13p1.n13(), g2_n13p1.n23(), g2_n13p1.n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u3::gelfand g2_n33p1 = gArr2[g2];
      g2_n33p1.n33(g2_n33p1.n33() + 1);
      if (g2_n33p1.check())
      {
        int jj2 = g2_n33p1.Lambdax2(), mm2 = g2_n33p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n33p1.n13() - g2_n33p1.n33(), 
                       g2_n33p1.n23() - g2_n33p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n33p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n33p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n33p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n33p1.e43_33();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n33p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n33p1.n12() - g2_n33p1.n33(), 
                                      g2_n33p1.n22() - g2_n33p1.n33(),  
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
                                     g2_n13p1.n13(), g2_n13p1.n23(), g2_n13p1.n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }
    }
    row++;
    return;
  }

  // build with E42
  void E42(int parent_g3, int& row, int etamax,
           std::vector<int>& same_weight_gArr3,
           std::vector<std::pair<int, int>>& uncoupled_same_weight_gArr3,
           const std::vector<u4::gelfand>& gArr1,
           const std::vector<u4::gelfand>& gArr2,
           const std::vector<u4::gelfand>& gArr3, Eigen::MatrixXd& b,
           u4::SO3_CGC& cgcs, std::vector<Eigen::MatrixXd>& E43E42E41,
           WigU4U3_coeffs& coeffs)
  {
    int jj3 = gArr3[parent_g3].Lambdax2(), 
        mm3 = gArr3[parent_g3].MLambdax2();

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
        int jj1 = g1_n13n12p1.Lambdax2(), mm1 = g1_n13n12p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n13n12p1.n13() - g1_n13n12p1.n33(), 
                       g1_n13n12p1.n23() - g1_n13n12p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n13n12p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n13n12p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n13n12p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n13n12p1.e42_13_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n13n12p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n13n12p1.n12() - g1_n13n12p1.n33(), 
                                      g1_n13n12p1.n22() - g1_n13n12p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n13n12p1.n13(), g1_n13n12p1.n23(), g1_n13n12p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g1_n13n22p1 = gArr1[g1];
      g1_n13n22p1.n13(g1_n13n22p1.n13() + 1);
      g1_n13n22p1.n22(g1_n13n22p1.n22() + 1);
      if (g1_n13n22p1.check())
      {
        int jj1 = g1_n13n22p1.Lambdax2(), mm1 = g1_n13n22p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n13n22p1.n13() - g1_n13n22p1.n33(), 
                       g1_n13n22p1.n23() - g1_n13n22p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n13n22p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n13n22p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n13n22p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n13n22p1.e42_13_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n13n22p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n13n22p1.n12() - g1_n13n22p1.n33(), 
                                      g1_n13n22p1.n22() - g1_n13n22p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n13n22p1.n13(), g1_n13n22p1.n23(), g1_n13n22p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g1_n23n12p1 = gArr1[g1];
      g1_n23n12p1.n23(g1_n23n12p1.n23() + 1);
      g1_n23n12p1.n12(g1_n23n12p1.n12() + 1);
      if (g1_n23n12p1.check())
      {
        int jj1 = g1_n23n12p1.Lambdax2(), mm1 = g1_n23n12p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n23n12p1.n13() - g1_n23n12p1.n33(), 
                       g1_n23n12p1.n23() - g1_n23n12p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n23n12p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n23n12p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n23n12p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n23n12p1.e42_23_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n23n12p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n23n12p1.n12() - g1_n23n12p1.n33(), 
                                      g1_n23n12p1.n22() - g1_n23n12p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n23n12p1.n13(), g1_n23n12p1.n23(), g1_n23n12p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g1_n23n22p1 = gArr1[g1];
      g1_n23n22p1.n23(g1_n23n22p1.n23() + 1);
      g1_n23n22p1.n22(g1_n23n22p1.n22() + 1);
      if (g1_n23n22p1.check())
      {
        int jj1 = g1_n23n22p1.Lambdax2(), mm1 = g1_n23n22p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n23n22p1.n13() - g1_n23n22p1.n33(), 
                       g1_n23n22p1.n23() - g1_n23n22p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n23n22p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n23n22p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n23n22p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n23n22p1.e42_23_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n23n22p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n23n22p1.n12() - g1_n23n22p1.n33(), 
                                      g1_n23n22p1.n22() - g1_n23n22p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n23n22p1.n13(), g1_n23n22p1.n23(), g1_n23n22p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g1_n33n12p1 = gArr1[g1];
      g1_n33n12p1.n33(g1_n33n12p1.n33() + 1);
      g1_n33n12p1.n12(g1_n33n12p1.n12() + 1);
      if (g1_n33n12p1.check())
      {
        int jj1 = g1_n33n12p1.Lambdax2(), mm1 = g1_n33n12p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n33n12p1.n13() - g1_n33n12p1.n33(), 
                       g1_n33n12p1.n23() - g1_n33n12p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n33n12p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n33n12p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n33n12p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n33n12p1.e42_33_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n33n12p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n33n12p1.n12() - g1_n33n12p1.n33(), 
                                      g1_n33n12p1.n22() - g1_n33n12p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n33n12p1.n13(), g1_n33n12p1.n23(), g1_n33n12p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g1_n33n22p1 = gArr1[g1];
      g1_n33n22p1.n33(g1_n33n22p1.n33() + 1);
      g1_n33n22p1.n22(g1_n33n22p1.n22() + 1);
      if (g1_n33n22p1.check())
      {
        int jj1 = g1_n33n22p1.Lambdax2(), mm1 = g1_n33n22p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n33n22p1.n13() - g1_n33n22p1.n33(), 
                       g1_n33n22p1.n23() - g1_n33n22p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n33n22p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n33n22p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n33n22p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n33n22p1.e42_33_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n33n22p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n33n22p1.n12() - g1_n33n22p1.n33(), 
                                      g1_n33n22p1.n22() - g1_n33n22p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n33n22p1.n13(), g1_n33n22p1.n23(), g1_n33n22p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      // act on the second state
      u4::gelfand g2_n13n12p1 = gArr2[g2];
      g2_n13n12p1.n13(g2_n13n12p1.n13() + 1);
      g2_n13n12p1.n12(g2_n13n12p1.n12() + 1);
      if (g2_n13n12p1.check())
      {
        int jj2 = g2_n13n12p1.Lambdax2(), mm2 = g2_n13n12p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n13n12p1.n13() - g2_n13n12p1.n33(), 
                       g2_n13n12p1.n23() - g2_n13n12p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n13n12p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n13n12p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n13n12p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n13n12p1.e42_13_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n13n12p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n13n12p1.n12() - g2_n13n12p1.n33(), 
                                      g2_n13n12p1.n22() - g2_n13n12p1.n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(), 
                                     g2_n13n12p1.n13(), g2_n13n12p1.n23(), g2_n13n12p1.n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g2_n13n22p1 = gArr2[g2];
      g2_n13n22p1.n13(g2_n13n22p1.n13() + 1);
      g2_n13n22p1.n22(g2_n13n22p1.n22() + 1);
      if (g2_n13n22p1.check())
      {
        int jj2 = g2_n13n22p1.Lambdax2(), mm2 = g2_n13n22p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n13n22p1.n13() - g2_n13n22p1.n33(), 
                       g2_n13n22p1.n23() - g2_n13n22p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n13n22p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n13n22p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n13n22p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n13n22p1.e42_13_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n13n22p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n13n22p1.n12() - g2_n13n22p1.n33(), 
                                      g2_n13n22p1.n22() - g2_n13n22p1.n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(), 
                                     g2_n13n22p1.n13(), g2_n13n22p1.n23(), g2_n13n22p1.n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g2_n23n12p1 = gArr2[g2];
      g2_n23n12p1.n23(g2_n23n12p1.n23() + 1);
      g2_n23n12p1.n12(g2_n23n12p1.n12() + 1);
      if (g2_n23n12p1.check())
      {
        int jj2 = g2_n23n12p1.Lambdax2(), mm2 = g2_n23n12p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n23n12p1.n13() - g2_n23n12p1.n33(), 
                       g2_n23n12p1.n23() - g2_n23n12p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n23n12p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n23n12p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n23n12p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n23n12p1.e42_23_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n23n12p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n23n12p1.n12() - g2_n23n12p1.n33(), 
                                      g2_n23n12p1.n22() - g2_n23n12p1.n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(), 
                                     g2_n23n12p1.n13(), g2_n23n12p1.n23(), g2_n23n12p1.n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g2_n23n22p1 = gArr2[g2];
      g2_n23n22p1.n23(g2_n23n22p1.n23() + 1);
      g2_n23n22p1.n22(g2_n23n22p1.n22() + 1);
      if (g2_n23n22p1.check())
      {
        int jj2 = g2_n23n22p1.Lambdax2(), mm2 = g2_n23n22p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n23n22p1.n13() - g2_n23n22p1.n33(), 
                       g2_n23n22p1.n23() - g2_n23n22p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n23n22p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n23n22p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n23n22p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n23n22p1.e42_23_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n23n22p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n23n22p1.n12() - g2_n23n22p1.n33(), 
                                      g2_n23n22p1.n22() - g2_n23n22p1.n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(), 
                                     g2_n23n22p1.n13(), g2_n23n22p1.n23(), g2_n23n22p1.n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g2_n33n12p1 = gArr2[g2];
      g2_n33n12p1.n33(g2_n33n12p1.n33() + 1);
      g2_n33n12p1.n12(g2_n33n12p1.n12() + 1);
      if (g2_n33n12p1.check())
      {
        int jj2 = g2_n33n12p1.Lambdax2(), mm2 = g2_n33n12p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n33n12p1.n13() - g2_n33n12p1.n33(), 
                       g2_n33n12p1.n23() - g2_n33n12p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n33n12p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n33n12p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n33n12p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n33n12p1.e42_33_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n33n12p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n33n12p1.n12() - g2_n33n12p1.n33(), 
                                      g2_n33n12p1.n22() - g2_n33n12p1.n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(), 
                                     g2_n33n12p1.n13(), g2_n33n12p1.n23(), g2_n33n12p1.n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g2_n33n22p1 = gArr2[g2];
      g2_n33n22p1.n33(g2_n33n22p1.n33() + 1);
      g2_n33n22p1.n22(g2_n33n22p1.n22() + 1);
      if (g2_n33n22p1.check())
      {
        int jj2 = g2_n33n22p1.Lambdax2(), mm2 = g2_n33n22p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n33n22p1.n13() - g2_n33n22p1.n33(), 
                       g2_n33n22p1.n23() - g2_n33n22p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n33n22p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n33n22p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n33n22p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n33n22p1.e42_33_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n33n22p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n33n22p1.n12() - g2_n33n22p1.n33(), 
                                      g2_n33n22p1.n22() - g2_n33n22p1.n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(), 
                                     g2_n33n22p1.n13(), g2_n33n22p1.n23(), g2_n33n22p1.n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }
    }
    row++;
    return;
  }

  // build with E41
  void E41(int parent_g3, int& row, int etamax,
           std::vector<int>& same_weight_gArr3,
           std::vector<std::pair<int, int>>& uncoupled_same_weight_gArr3,
           const std::vector<u4::gelfand>& gArr1,
           const std::vector<u4::gelfand>& gArr2,
           const std::vector<u4::gelfand>& gArr3, Eigen::MatrixXd& b,
           u4::SO3_CGC& cgcs, std::vector<Eigen::MatrixXd>& E43E42E41,
           WigU4U3_coeffs& coeffs)
  {
    int jj3 = gArr3[parent_g3].Lambdax2(), 
        mm3 = gArr3[parent_g3].MLambdax2();

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
        int jj1 = g1_n13n12n11p1.Lambdax2(), mm1 = g1_n13n12n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n13n12n11p1.n13() - g1_n13n12n11p1.n33(), 
                       g1_n13n12n11p1.n23() - g1_n13n12n11p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n13n12n11p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n13n12n11p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n13n12n11p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n13n12n11p1.e41_13_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n13n12n11p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n13n12n11p1.n12() - g1_n13n12n11p1.n33(), 
                                      g1_n13n12n11p1.n22() - g1_n13n12n11p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n13n12n11p1.n13(), g1_n13n12n11p1.n23(), 
                                     g1_n13n12n11p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g1_n13n22n11p1 = gArr1[g1];
      g1_n13n22n11p1.n13(g1_n13n22n11p1.n13() + 1);
      g1_n13n22n11p1.n22(g1_n13n22n11p1.n22() + 1);
      g1_n13n22n11p1.n11(g1_n13n22n11p1.n11() + 1);
      if (g1_n13n22n11p1.check())
      {
        int jj1 = g1_n13n22n11p1.Lambdax2(), mm1 = g1_n13n22n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n13n22n11p1.n13() - g1_n13n22n11p1.n33(), 
                       g1_n13n22n11p1.n23() - g1_n13n22n11p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n13n22n11p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n13n22n11p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n13n22n11p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n13n22n11p1.e41_13_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n13n22n11p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n13n22n11p1.n12() - g1_n13n22n11p1.n33(), 
                                      g1_n13n22n11p1.n22() - g1_n13n22n11p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n13n22n11p1.n13(), g1_n13n22n11p1.n23(), 
                                     g1_n13n22n11p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g1_n23n12n11p1 = gArr1[g1];
      g1_n23n12n11p1.n23(g1_n23n12n11p1.n23() + 1);
      g1_n23n12n11p1.n12(g1_n23n12n11p1.n12() + 1);
      g1_n23n12n11p1.n11(g1_n23n12n11p1.n11() + 1);
      if (g1_n23n12n11p1.check())
      {
        int jj1 = g1_n23n12n11p1.Lambdax2(), mm1 = g1_n23n12n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n23n12n11p1.n13() - g1_n23n12n11p1.n33(), 
                       g1_n23n12n11p1.n23() - g1_n23n12n11p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n23n12n11p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n23n12n11p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n23n12n11p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n23n12n11p1.e41_23_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n23n12n11p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n23n12n11p1.n12() - g1_n23n12n11p1.n33(), 
                                      g1_n23n12n11p1.n22() - g1_n23n12n11p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n23n12n11p1.n13(), g1_n23n12n11p1.n23(), 
                                     g1_n23n12n11p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g1_n23n22n11p1 = gArr1[g1];
      g1_n23n22n11p1.n23(g1_n23n22n11p1.n23() + 1);
      g1_n23n22n11p1.n22(g1_n23n22n11p1.n22() + 1);
      g1_n23n22n11p1.n11(g1_n23n22n11p1.n11() + 1);
      if (g1_n23n22n11p1.check())
      {
        int jj1 = g1_n23n22n11p1.Lambdax2(), mm1 = g1_n23n22n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n23n22n11p1.n13() - g1_n23n22n11p1.n33(), 
                       g1_n23n22n11p1.n23() - g1_n23n22n11p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n23n22n11p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n23n22n11p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n23n22n11p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n23n22n11p1.e41_23_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n23n22n11p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n23n22n11p1.n12() - g1_n23n22n11p1.n33(), 
                                      g1_n23n22n11p1.n22() - g1_n23n22n11p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n23n22n11p1.n13(), g1_n23n22n11p1.n23(), 
                                     g1_n23n22n11p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g1_n33n12n11p1 = gArr1[g1];
      g1_n33n12n11p1.n33(g1_n33n12n11p1.n33() + 1);
      g1_n33n12n11p1.n12(g1_n33n12n11p1.n12() + 1);
      g1_n33n12n11p1.n11(g1_n33n12n11p1.n11() + 1);
      if (g1_n33n12n11p1.check())
      {
        int jj1 = g1_n33n12n11p1.Lambdax2(), mm1 = g1_n33n12n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n33n12n11p1.n13() - g1_n33n12n11p1.n33(), 
                       g1_n33n12n11p1.n23() - g1_n33n12n11p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n33n12n11p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n33n12n11p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n33n12n11p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n33n12n11p1.e41_33_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n33n12n11p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n33n12n11p1.n12() - g1_n33n12n11p1.n33(), 
                                      g1_n33n12n11p1.n22() - g1_n33n12n11p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n33n12n11p1.n13(), g1_n33n12n11p1.n23(), 
                                     g1_n33n12n11p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g1_n33n22n11p1 = gArr1[g1];
      g1_n33n22n11p1.n33(g1_n33n22n11p1.n33() + 1);
      g1_n33n22n11p1.n22(g1_n33n22n11p1.n22() + 1);
      g1_n33n22n11p1.n11(g1_n33n22n11p1.n11() + 1);
      if (g1_n33n22n11p1.check())
      {
        int jj1 = g1_n33n22n11p1.Lambdax2(), mm1 = g1_n33n22n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        u3::irrep u3_1(g1_n33n22n11p1.n13() - g1_n33n22n11p1.n33(), 
                       g1_n33n22n11p1.n23() - g1_n33n22n11p1.n33(), 0),
                  u3_2(gArr2[g2].n13() - gArr2[g2].n33(), 
                       gArr2[g2].n23() - gArr2[g2].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g1_n33n22n11p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n23() - g1_n33n22n11p1.n33() - gArr2[g2].n33(), 
                       gArr3[parent_g3].n33() - g1_n33n22n11p1.n33() - gArr2[g2].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g1_n33n22n11p1.e41_33_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g1_n33n22n11p1.n33() - 2 * gArr2[g2].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(g1_n33n22n11p1.n12() - g1_n33n22n11p1.n33(), 
                                      g1_n33n22n11p1.n22() - g1_n33n22n11p1.n33(), 
                                      gArr2[g2].n12() - gArr2[g2].n33(), 
                                      gArr2[g2].n22() - gArr2[g2].n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(g1_n33n22n11p1.n13(), g1_n33n22n11p1.n23(), 
                                     g1_n33n22n11p1.n33(), 
                                     gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(), 
                                     gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
                                     gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      // act on the second state
      u4::gelfand g2_n13n12n11p1 = gArr2[g2];
      g2_n13n12n11p1.n13(g2_n13n12n11p1.n13() + 1);
      g2_n13n12n11p1.n12(g2_n13n12n11p1.n12() + 1);
      g2_n13n12n11p1.n11(g2_n13n12n11p1.n11() + 1);
      if (g2_n13n12n11p1.check())
      {
        int jj2 = g2_n13n12n11p1.Lambdax2(), mm2 = g2_n13n12n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n13n12n11p1.n13() - g2_n13n12n11p1.n33(), 
                       g2_n13n12n11p1.n23() - g2_n13n12n11p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n13n12n11p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n13n12n11p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n13n12n11p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n13n12n11p1.e41_13_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n13n12n11p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n13n12n11p1.n12() - g2_n13n12n11p1.n33(), 
                                      g2_n13n12n11p1.n22() - g2_n13n12n11p1.n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(), 
                                     g2_n13n12n11p1.n13(), g2_n13n12n11p1.n23(), 
                                     g2_n13n12n11p1.n33(), gArr3[parent_g3].n13(), 
                                     gArr3[parent_g3].n23(), gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g2_n13n22n11p1 = gArr2[g2];
      g2_n13n22n11p1.n13(g2_n13n22n11p1.n13() + 1);
      g2_n13n22n11p1.n22(g2_n13n22n11p1.n22() + 1);
      g2_n13n22n11p1.n11(g2_n13n22n11p1.n11() + 1);
      if (g2_n13n22n11p1.check())
      {
        int jj2 = g2_n13n22n11p1.Lambdax2(), mm2 = g2_n13n22n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n13n22n11p1.n13() - g2_n13n22n11p1.n33(), 
                       g2_n13n22n11p1.n23() - g2_n13n22n11p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n13n22n11p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n13n22n11p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n13n22n11p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n13n22n11p1.e41_13_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n13n22n11p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n13n22n11p1.n12() - g2_n13n22n11p1.n33(), 
                                      g2_n13n22n11p1.n22() - g2_n13n22n11p1.n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(), 
                                     g2_n13n22n11p1.n13(), g2_n13n22n11p1.n23(), 
                                     g2_n13n22n11p1.n33(), gArr3[parent_g3].n13(), 
                                     gArr3[parent_g3].n23(), gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g2_n23n12n11p1 = gArr2[g2];
      g2_n23n12n11p1.n23(g2_n23n12n11p1.n23() + 1);
      g2_n23n12n11p1.n12(g2_n23n12n11p1.n12() + 1);
      g2_n23n12n11p1.n11(g2_n23n12n11p1.n11() + 1);
      if (g2_n23n12n11p1.check())
      {
        int jj2 = g2_n23n12n11p1.Lambdax2(), mm2 = g2_n23n12n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n23n12n11p1.n13() - g2_n23n12n11p1.n33(), 
                       g2_n23n12n11p1.n23() - g2_n23n12n11p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n23n12n11p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n23n12n11p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n23n12n11p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n23n12n11p1.e41_23_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n23n12n11p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n23n12n11p1.n12() - g2_n23n12n11p1.n33(), 
                                      g2_n23n12n11p1.n22() - g2_n23n12n11p1.n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(), 
                                     g2_n23n12n11p1.n13(), g2_n23n12n11p1.n23(), 
                                     g2_n23n12n11p1.n33(), gArr3[parent_g3].n13(), 
                                     gArr3[parent_g3].n23(), gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g2_n23n22n11p1 = gArr2[g2];
      g2_n23n22n11p1.n23(g2_n23n22n11p1.n23() + 1);
      g2_n23n22n11p1.n22(g2_n23n22n11p1.n22() + 1);
      g2_n23n22n11p1.n11(g2_n23n22n11p1.n11() + 1);
      if (g2_n23n22n11p1.check())
      {
        int jj2 = g2_n23n22n11p1.Lambdax2(), mm2 = g2_n23n22n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n23n22n11p1.n13() - g2_n23n22n11p1.n33(), 
                       g2_n23n22n11p1.n23() - g2_n23n22n11p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n23n22n11p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n23n22n11p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n23n22n11p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n23n22n11p1.e41_23_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n23n22n11p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n23n22n11p1.n12() - g2_n23n22n11p1.n33(), 
                                      g2_n23n22n11p1.n22() - g2_n23n22n11p1.n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(), 
                                     g2_n23n22n11p1.n13(), g2_n23n22n11p1.n23(), 
                                     g2_n23n22n11p1.n33(), gArr3[parent_g3].n13(), 
                                     gArr3[parent_g3].n23(), gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g2_n33n12n11p1 = gArr2[g2];
      g2_n33n12n11p1.n33(g2_n33n12n11p1.n33() + 1);
      g2_n33n12n11p1.n12(g2_n33n12n11p1.n12() + 1);
      g2_n33n12n11p1.n11(g2_n33n12n11p1.n11() + 1);
      if (g2_n33n12n11p1.check())
      {
        int jj2 = g2_n33n12n11p1.Lambdax2(), mm2 = g2_n33n12n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n33n12n11p1.n13() - g2_n33n12n11p1.n33(), 
                       g2_n33n12n11p1.n23() - g2_n33n12n11p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n33n12n11p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n33n12n11p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n33n12n11p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n33n12n11p1.e41_33_12();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n33n12n11p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n33n12n11p1.n12() - g2_n33n12n11p1.n33(), 
                                      g2_n33n12n11p1.n22() - g2_n33n12n11p1.n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(), 
                                     g2_n33n12n11p1.n13(), g2_n33n12n11p1.n23(), 
                                     g2_n33n12n11p1.n33(), gArr3[parent_g3].n13(), 
                                     gArr3[parent_g3].n23(), gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }

      u4::gelfand g2_n33n22n11p1 = gArr2[g2];
      g2_n33n22n11p1.n33(g2_n33n22n11p1.n33() + 1);
      g2_n33n22n11p1.n22(g2_n33n22n11p1.n22() + 1);
      g2_n33n22n11p1.n11(g2_n33n22n11p1.n11() + 1);
      if (g2_n33n22n11p1.check())
      {
        int jj2 = g2_n33n22n11p1.Lambdax2(), mm2 = g2_n33n22n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        u3::irrep u3_2(g2_n33n22n11p1.n13() - g2_n33n22n11p1.n33(), 
                       g2_n33n22n11p1.n23() - g2_n33n22n11p1.n33(), 0),
                  u3_1(gArr1[g1].n13() - gArr1[g1].n33(), 
                       gArr1[g1].n23() - gArr1[g1].n33(), 0),
                  u3_3(gArr3[parent_g3].n13() - g2_n33n22n11p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n23() - g2_n33n22n11p1.n33() - gArr1[g1].n33(), 
                       gArr3[parent_g3].n33() - g2_n33n22n11p1.n33() - gArr1[g1].n33());
        int rhomax = u3::Rho(u3_1, u3_2, u3_3);

        if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
            (jj1 + jj2) >= jj3 && rhomax > 0)
        {
          double factor = g2_n33n22n11p1.e42_33_22();

          double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

          std::vector<u3::gelfand> gArr3_1 = u3::gelfandBasis(u3_1),
                                   gArr3_2 = u3::gelfandBasis(u3_2),
                                   gArr3_3 = u3::gelfandBasis(u3_3);

          int w2 = gArr3[parent_g3].n12() + gArr3[parent_g3].n22() 
                   - 2 * g2_n33n22n11p1.n33() - 2 * gArr1[g1].n33();

          auto wigU3U2 = u3::WigU3U2(u3_1, u3_2, u3_3, rhomax, w2, 
                                     gArr3_1, gArr3_2, gArr3_3);
          // after this, u3_3, gArr3_3 and wigU3U2 are normalized

          auto accU3 = wigU3U2.access(gArr1[g1].n12() - gArr1[g1].n33(), 
                                      gArr1[g1].n22() - gArr1[g1].n33(), 
                                      g2_n33n22n11p1.n12() - g2_n33n22n11p1.n33(), 
                                      g2_n33n22n11p1.n22() - g2_n33n22n11p1.n33(), 
                                      gArr3[parent_g3].n12() - gArr3[parent_g3].n33(), 
                                      gArr3[parent_g3].n22() - gArr3[parent_g3].n33());

          auto accU4 = coeffs.access(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(), 
                                     g2_n33n22n11p1.n13(), g2_n33n22n11p1.n23(), 
                                     g2_n33n22n11p1.n33(), gArr3[parent_g3].n13(), 
                                     gArr3[parent_g3].n23(), gArr3[parent_g3].n33());

          for (int e = 0; e < etamax; e++)            
            for (int r = 0; r < rhomax; r++)
              E43E42E41[e](row, j) += coeffs.get(accU4, e) * wigU3U2.get(accU3, r) * cgc * factor;
        }
      }
    }
    row++;
    return;
  }

  // lower-weight coefficients < G1; G2 | G3 >
  void LowerWeightReducedCoeffs(const std::vector<u4::gelfand>& gArr1,
                                const std::vector<u4::gelfand>& gArr2,
                                const std::vector<u4::gelfand>& gArr3,
                                int etamax, std::vector<int>& same_weight_gArr3,
                                u4::SO3_CGC& cgcs, WigU4U3_coeffs& coeffs)
  {

    // trace states with same [w3,w2,w1] (z-weight)
    int w3 = gArr3[same_weight_gArr3.back()].n13()
           + gArr3[same_weight_gArr3.back()].n23()
           + gArr3[same_weight_gArr3.back()].n33(),
        w2 = gArr3[same_weight_gArr3.back()].n12()
           + gArr3[same_weight_gArr3.back()].n22(),
        w1 = gArr3[same_weight_gArr3.back()].n11();
              
    for (int i = 0; i < same_weight_gArr3.size(); i++)
    {
      for (int j = same_weight_gArr3[i]; j < gArr3.size(); j++)
        if (gArr3[j].n13() + gArr3[j].n23() + gArr3[j].n33() == w3
            && gArr3[j].n12() + gArr3[j].n22() == w2 && gArr3[j].n11() == w1)
        {
          same_weight_gArr3[i] = j;
          break;
        }
    }

    if (!same_weight_gArr3.empty())
    {
      // the coupled states having the same weight have the same expansion of
      // uncoupled states, whose indices are recorded
      std::vector<std::pair<int, int>> uncoupled_same_weight_gArr3 =
          expand_to_uncoupled(gArr3[same_weight_gArr3[0]], gArr1, gArr2);

      // matrix b is independent of the outer multiplicity
      Eigen::MatrixXd b(same_weight_gArr3.size(), same_weight_gArr3.size());

      // the lowering matrix from the action of E43, E42 and E41, which depends on
      // the outer multiplicity
      std::vector<Eigen::MatrixXd> E43E42E41(etamax);

      // set matrix elements to zero
      for (int e = 0; e < etamax; e++)
      {
        E43E42E41[e] = Eigen::MatrixXd(same_weight_gArr3.size(),
                                    uncoupled_same_weight_gArr3.size());
        E43E42E41[e].setZero();
      }

      // find parent states and operators to get to lower weight states
      // (daughter states)
      std::vector<std::pair<int, int>> actOnParent;

      int row = 0;

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
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E43(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
        }

        u4::gelfand g3_n23p1 = gArr3[same_weight_gArr3[i]];
        g3_n23p1.n23(g3_n23p1.n23() + 1);
        if (g3_n23p1.check())
        {
          auto j = u4::findGelfand(g3_n23p1, gArr3);
          std::pair<int, int> p(0, j);
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E43(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
        }

        u4::gelfand g3_n33p1 = gArr3[same_weight_gArr3[i]];
        g3_n33p1.n33(g3_n33p1.n33() + 1);
        if (g3_n33p1.check())
        {
          auto j = u4::findGelfand(g3_n33p1, gArr3);
          std::pair<int, int> p(0, j);
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E43(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
        }

        // action of E42
        u4::gelfand g3_n13n12p1 = gArr3[same_weight_gArr3[i]];
        g3_n13n12p1.n13(g3_n13n12p1.n13() + 1);
        g3_n13n12p1.n12(g3_n13n12p1.n12() + 1);
        if (g3_n13n12p1.check())
        {
          auto j = u4::findGelfand(g3_n13n12p1, gArr3);
          std::pair<int, int> p(1, j);
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E42(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
        }

        u4::gelfand g3_n13n22p1 = gArr3[same_weight_gArr3[i]];
        g3_n13n22p1.n13(g3_n13n22p1.n13() + 1);
        g3_n13n22p1.n22(g3_n13n22p1.n22() + 1);
        if (g3_n13n22p1.check())
        {
          auto j = u4::findGelfand(g3_n13n22p1, gArr3);
          std::pair<int, int> p(1, j);
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E42(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
        }

        u4::gelfand g3_n23n12p1 = gArr3[same_weight_gArr3[i]];
        g3_n23n12p1.n23(g3_n23n12p1.n23() + 1);
        g3_n23n12p1.n12(g3_n23n12p1.n12() + 1);
        if (g3_n23n12p1.check())
        {
          auto j = u4::findGelfand(g3_n23n12p1, gArr3);
          std::pair<int, int> p(1, j);
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E42(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
        }

        u4::gelfand g3_n23n22p1 = gArr3[same_weight_gArr3[i]];
        g3_n23n22p1.n23(g3_n23n22p1.n23() + 1);
        g3_n23n22p1.n22(g3_n23n22p1.n22() + 1);
        if (g3_n23n22p1.check())
        {
          auto j = u4::findGelfand(g3_n23n22p1, gArr3);
          std::pair<int, int> p(1, j);
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E42(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
        }

        u4::gelfand g3_n33n12p1 = gArr3[same_weight_gArr3[i]];
        g3_n33n12p1.n33(g3_n33n12p1.n33() + 1);
        g3_n33n12p1.n12(g3_n33n12p1.n12() + 1);
        if (g3_n33n12p1.check())
        {
          auto j = u4::findGelfand(g3_n33n12p1, gArr3);
          std::pair<int, int> p(1, j);
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E42(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
        }

        u4::gelfand g3_n33n22p1 = gArr3[same_weight_gArr3[i]];
        g3_n33n22p1.n33(g3_n33n22p1.n33() + 1);
        g3_n33n22p1.n22(g3_n33n22p1.n22() + 1);
        if (g3_n33n22p1.check())
        {
          auto j = u4::findGelfand(g3_n33n22p1, gArr3);
          std::pair<int, int> p(1, j);
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E42(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
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
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E41(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
        }

        u4::gelfand g3_n13n22n11p1 = gArr3[same_weight_gArr3[i]];
        g3_n13n22n11p1.n13(g3_n13n22n11p1.n13() + 1);
        g3_n13n22n11p1.n22(g3_n13n22n11p1.n22() + 1);
        g3_n13n22n11p1.n11(g3_n13n22n11p1.n11() + 1);
        if (g3_n13n22n11p1.check())
        {
          auto j = u4::findGelfand(g3_n13n22n11p1, gArr3);
          std::pair<int, int> p(2, j);
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E41(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
        }

        u4::gelfand g3_n23n12n11p1 = gArr3[same_weight_gArr3[i]];
        g3_n23n12n11p1.n23(g3_n23n12n11p1.n23() + 1);
        g3_n23n12n11p1.n12(g3_n23n12n11p1.n12() + 1);
        g3_n23n12n11p1.n11(g3_n23n12n11p1.n11() + 1);
        if (g3_n23n12n11p1.check())
        {
          auto j = u4::findGelfand(g3_n23n12n11p1, gArr3);
          std::pair<int, int> p(2, j);
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E41(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
        }

        u4::gelfand g3_n23n22n11p1 = gArr3[same_weight_gArr3[i]];
        g3_n23n22n11p1.n23(g3_n23n22n11p1.n23() + 1);
        g3_n23n22n11p1.n22(g3_n23n22n11p1.n22() + 1);
        g3_n23n22n11p1.n11(g3_n23n22n11p1.n11() + 1);
        if (g3_n23n22n11p1.check())
        {
          auto j = u4::findGelfand(g3_n23n22n11p1, gArr3);
          std::pair<int, int> p(2, j);
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E41(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
        }

        u4::gelfand g3_n33n12n11p1 = gArr3[same_weight_gArr3[i]];
        g3_n33n12n11p1.n33(g3_n33n12n11p1.n33() + 1);
        g3_n33n12n11p1.n12(g3_n33n12n11p1.n12() + 1);
        g3_n33n12n11p1.n11(g3_n33n12n11p1.n11() + 1);
        if (g3_n33n12n11p1.check())
        {
          auto j = u4::findGelfand(g3_n33n12n11p1, gArr3);
          std::pair<int, int> p(2, j);
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E41(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
        }

        u4::gelfand g3_n33n22n11p1 = gArr3[same_weight_gArr3[i]];
        g3_n33n22n11p1.n33(g3_n33n22n11p1.n33() + 1);
        g3_n33n22n11p1.n22(g3_n33n22n11p1.n22() + 1);
        g3_n33n22n11p1.n11(g3_n33n22n11p1.n11() + 1);
        if (g3_n33n22n11p1.check())
        {
          auto j = u4::findGelfand(g3_n33n22n11p1, gArr3);
          std::pair<int, int> p(2, j);
          auto it = std::find(actOnParent.begin(), actOnParent.end(), p);
          if (it == actOnParent.end())
          {
            actOnParent.push_back(p);
            E41(j, row, etamax, same_weight_gArr3, uncoupled_same_weight_gArr3,
                gArr1, gArr2, gArr3, b, cgcs, E43E42E41, coeffs);
            continue;
          }
        }
      }

      Eigen::ColPivHouseholderQR<Eigen::MatrixXd> factorize;
      factorize.compute(b);

      std::vector<Eigen::MatrixXd> coeffs_temp(etamax);

      for (int e = 0; e < etamax; e++)
        coeffs_temp[e] = factorize.solve(E43E42E41[e]);

      /*
      Eigen::RowVectorXd coeffs_row(etamax); // define here to avoid allocation

      for (int i = 0; i < same_weight_gArr3.size(); i++)
      {
        for (int j = 0; j < uncoupled_same_weight_gArr3.size(); j++)
        {

          int i1 = uncoupled_same_weight_gArr3[j].first,
              i2 = uncoupled_same_weight_gArr3[j].second,
              i3 = same_weight_gArr3[i];

          int i1_hw = i1 - gArr1[i1].n12() + gArr1[i1].n11(),
              i2_hw = i2 - gArr2[i2].n12() + gArr2[i2].n11(),
              i3_hw = i3 - gArr3[i3].n12() + gArr3[i3].n11();

          const auto& gA = gArr1[i1_hw];
          const auto& gB = gArr2[i2_hw];
          const auto& gC = gArr3[i3_hw];

          auto acc = coeffs.access(gA.n12(), gA.n22(), gB.n12(), gB.n22(),
                                   gC.n12(), gC.n22());

          if (acc.computed == false)
          {
            int jj1 = gArr1[i1].Lambdax2(), jj2 = gArr2[i2].Lambdax2(),
                jj3 = gArr3[i3].Lambdax2(), mm1 = gArr1[i1].MLambdax2(),
                mm2 = gArr2[i2].MLambdax2(), mm3 = gArr3[i3].MLambdax2();

            if ((mm1 + mm2) == mm3 && std::fabs(jj1 - jj2) <= jj3 &&
                (jj1 + jj2) >= jj3)
            {
              double cgc = cgcs.compute(jj1, jj2, jj3, mm1, mm2, mm3);

              for (int r = 0; r < rhomax; r++)
              {
                if (std::fabs(coeffs_temp[r](i, j)) < 1e-15)
                  coeffs_temp[r](i, j) = 0.0;

                assert(cgc != 0.0);

                coeffs_row(r) = coeffs_temp[r](i, j) / cgc;
              }

              coeffs.set_row(acc, coeffs_row);
            }
          }
        }
      }
      */
    }
    return;
  }

  // all reduced coupling coefficients
  auto WigU4U3(const u4::irrep& u4A, const u4::irrep& u4B, u4::irrep& u4C,
               int etamax, int w3min, std::vector<u4::gelfand>& gArrA,
               std::vector<u4::gelfand>& gArrB, std::vector<u4::gelfand>& gArrC)
  {
    WigU4U3_coeffs coeffs(u4A, u4B, u4C, etamax);

    u4::SO3_CGC cgcs;

    u4::HighestWeightReducedCoeffs(gArrA, gArrB, gArrC, cgcs, coeffs);

#ifdef U4LIB_TIME_LOG
    auto t0 = std::chrono::high_resolution_clock::now();
#endif

    int w3max = u4C.n14() + u4C.n24() + u4C.n34();
    if (w3min < w3max)
    {
      if (w3min == -1) // calculate all
        w3min = u4C.n24() + u4C.n34() + u4C.n44();

      std::vector<std::vector<int>> all_same_weight_gArrC(w3max - w3min + 1);
      for (int i = 0; i < gArrC.size(); i++)
        if (gArrC[i].n12() == gArrC[i].n13() && gArrC[i].n11() == gArrC[i].n13()
            && gArrC[i].n22() == gArrC[i].n23())
          all_same_weight_gArrC[w3max - gArrC[i].n13() - gArrC[i].n23() 
                                - gArrC[i].n33()].push_back(i);

      for (int w3 = w3max - 1; w3 >= w3min; w3--)
      {
        u4::LowerWeightReducedCoeffs(gArrA, gArrB, gArrC, etamax,
                                     all_same_weight_gArrC[w3max - w3], cgcs,
                                     coeffs);
      }
    }

#ifdef U4LIB_TIME_LOG
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = t1 - t0;
    std::cerr << "LowerWeightReducedCoeffs total time: " << duration.count()
              << " [s]" << std::endl;
#endif

    u4C.normalize();
    for (int i = 0; i < gArrC.size(); i++)
      gArrC[i].normalize();

    // once bases are normalized, coeffs needs to be as well:
    coeffs.normalize();

    return coeffs;
  }

} // namespace u4

#endif

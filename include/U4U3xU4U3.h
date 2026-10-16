#ifndef U4LIB_U4U3XU4U3_H
#define U4LIB_U4U3XU4U3_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <memory>
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
#include <Eigen/Sparse>   // Eigen::Triplet — single-pass E34E23E12 build (both paths)

#include <U3basis.h>
#include <U3decomposition.h>
#include <U3U2xU3U2.h>

#include <so3Coeffs/SO3_CGC.h>

#include <nullspace_solver_qr.h>
#if defined(NULLSPACE_SOLVER_SLEPC)
#elif defined(NULLSPACE_SOLVER_SNS)
#include <nullspace_solver_sns.h>
#endif

#include "U4basis.h"
#include "U4decomposition.h"

#ifdef TIME_WIGU4U3   // flag in Definitions.h (pulled in via U4basis.h)
#include <chrono>
#include <iostream>
#endif

namespace u4
{
  // This should be computed in only one thread
  class WigU4U3_coeffs
  {
    u4::irrep u4a_, u4b_, u4c_;

    std::size_t etamax_;

    // using key_type = std::array<int, 9>;
    using key_type = std::pair<uint64_t, uint8_t>;

    struct hasher
    {
      /*
      static void hash_combine(std::size_t& seed, int value)
      {
        seed ^=
            std::hash<int>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
      */
      template<typename T>
      static void hash_combine(std::size_t& seed, const T& value) noexcept
      {
          seed ^= std::hash<T>{}(value)
                + 0x9e3779b97f4a7c15ULL
                + (seed << 6)
                + (seed >> 2);
      }

      std::size_t operator()(const key_type& key) const
      {
        std::size_t seed = 0;

        hash_combine(seed, key.first);
        hash_combine(seed, key.second);

        /*
        hash_combine(seed, key[0]);
        hash_combine(seed, key[1]);
        hash_combine(seed, key[2]);
        hash_combine(seed, key[3]);
        hash_combine(seed, key[4]);
        hash_combine(seed, key[5]);
        hash_combine(seed, key[6]);
        hash_combine(seed, key[7]);
        hash_combine(seed, key[8]);
        */

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

      std::array<utils::UInt8, 9> safe_key = {
                            u3a.n13(), u3a.n23(), u3a.n33(), 
                            u3b.n13(), u3b.n23(), u3b.n33(), 
                            u3c.n13(), u3c.n23(), u3c.n33()};

      uint64_t key_1 = 
        (uint64_t(safe_key[0]) << 56) | (uint64_t(safe_key[1]) << 48) |
        (uint64_t(safe_key[2]) << 40) | (uint64_t(safe_key[3]) << 32) |
        (uint64_t(safe_key[4]) << 24) | (uint64_t(safe_key[5]) << 16) |
        (uint64_t(safe_key[6]) << 8) | uint64_t(safe_key[7]);

      key_type key = {key_1, safe_key[8]};

      /*
      key_type key = {u3a.n13(), u3a.n23(), u3a.n33(), u3b.n13(), u3b.n23(),
                      u3b.n33(), u3c.n13(), u3c.n23(), u3c.n33()};
      */

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
      // bool exist = false;
      Eigen::MatrixXd X;

      auto rhomax = u3::Rho(u3a, u3b, u3c);
      if (rhomax > 0)
      {
        std::array<utils::UInt8, 9> safe_key = {
                            u3a.n13(), u3a.n23(), u3a.n33(), 
                            u3b.n13(), u3b.n23(), u3b.n33(), 
                            u3c.n13(), u3c.n23(), u3c.n33()};

        uint64_t key_1 = 
          (uint64_t(safe_key[0]) << 56) | (uint64_t(safe_key[1]) << 48) |
          (uint64_t(safe_key[2]) << 40) | (uint64_t(safe_key[3]) << 32) |
          (uint64_t(safe_key[4]) << 24) | (uint64_t(safe_key[5]) << 16) |
          (uint64_t(safe_key[6]) << 8) | uint64_t(safe_key[7]);

        key_type key = {key_1, safe_key[8]};

        /*
        key_type key = {u3a.n13(), u3a.n23(), u3a.n33(), u3b.n13(), u3b.n23(),
                        u3b.n33(), u3c.n13(), u3c.n23(), u3c.n33()};
        */

        auto iter = map_.find(key);
        if (iter != map_.end())
        {
          X = coeffs_.block(iter->second, 0, rhomax, etamax_);
          // exist = true;
        }
      }

      return X;
      // return std::pair(exist, X);
    }

    // Non-owning view of the (rhomax x etamax) block for a U(3) triple --
    // same lookup as access() but with no MatrixXd copy. The block is
    // contiguous (row-major storage, full width), so it maps directly.
    // size() == 0 signals "not found" (as with access()). Valid while the
    // coeffs object is alive; hot consumers hold it via
    // WigU4U3_db::get_coeffs and resolve blocks locally.
    using BlockView = Eigen::Map<const Eigen::Matrix<
        double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

    BlockView find_block(const u3::irrep& u3a, const u3::irrep& u3b,
                         const u3::irrep& u3c) const
    {
      auto rhomax = u3::Rho(u3a, u3b, u3c);
      if (rhomax > 0)
      {
        std::array<utils::UInt8, 9> safe_key = {
                            u3a.n13(), u3a.n23(), u3a.n33(),
                            u3b.n13(), u3b.n23(), u3b.n33(),
                            u3c.n13(), u3c.n23(), u3c.n33()};

        uint64_t key_1 =
          (uint64_t(safe_key[0]) << 56) | (uint64_t(safe_key[1]) << 48) |
          (uint64_t(safe_key[2]) << 40) | (uint64_t(safe_key[3]) << 32) |
          (uint64_t(safe_key[4]) << 24) | (uint64_t(safe_key[5]) << 16) |
          (uint64_t(safe_key[6]) << 8) | uint64_t(safe_key[7]);

        key_type key = {key_1, safe_key[8]};

        auto iter = map_.find(key);
        if (iter != map_.end())
          return BlockView(coeffs_.data() + iter->second * etamax_,
                           rhomax, etamax_);
      }

      return BlockView(nullptr, 0, 0);
    }

    int size() const { return map_.size(); }

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

    /*
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

#ifdef NULLSPACE_SPARSE
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
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr1[g1].e34_13());
#else
          mat(i, j) = gArr1[g1].e34_13();
#endif
        else if (u4::E34_23(gArr1[g1], gArr1[new_g1]) && new_g2 == g2)
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr1[g1].e34_23());
#else
          mat(i, j) = gArr1[g1].e34_23();
#endif
        else if (u4::E34_33(gArr1[g1], gArr1[new_g1]) && new_g2 == g2)
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr1[g1].e34_33());
#else
          mat(i, j) = gArr1[g1].e34_33();
#endif
        else if (new_g1 == g1 && u4::E34_13(gArr2[g2], gArr2[new_g2]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr2[g2].e34_13());
#else
          mat(i, j) = gArr2[g2].e34_13();
#endif
        else if (new_g1 == g1 && u4::E34_23(gArr2[g2], gArr2[new_g2]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr2[g2].e34_23());
#else
          mat(i, j) = gArr2[g2].e34_23();
#endif
        else if (new_g1 == g1 && u4::E34_33(gArr2[g2], gArr2[new_g2]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr2[g2].e34_33());
#else
          mat(i, j) = gArr2[g2].e34_33();
#endif

        // action of E23
        else if (u4::E23_12(gArr1[g1], gArr1[new_g1]) && new_g2 == g2)
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr1[g1].e23_12());
#else
          mat(i, j) = gArr1[g1].e23_12();
#endif
        else if (u4::E23_22(gArr1[g1], gArr1[new_g1]) && new_g2 == g2)
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr1[g1].e23_22());
#else
          mat(i, j) = gArr1[g1].e23_22();
#endif
        else if (new_g1 == g1 && u4::E23_12(gArr2[g2], gArr2[new_g2]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr2[g2].e23_12());
#else
          mat(i, j) = gArr2[g2].e23_12();
#endif
        else if (new_g1 == g1 && u4::E23_22(gArr2[g2], gArr2[new_g2]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr2[g2].e23_22());
#else
          mat(i, j) = gArr2[g2].e23_22();
#endif

        // action of E12
        else if (u4::E12(gArr1[g1], gArr1[new_g1]) && new_g2 == g2)
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr1[g1].e12());
#else
          mat(i, j) = gArr1[g1].e12();
#endif
        else if (new_g1 == g1 && u4::E12(gArr2[g2], gArr2[new_g2]))
#ifdef NULLSPACE_SPARSE
          ija.emplace_back(i, j, gArr2[g2].e12());
#else
          mat(i, j) = gArr2[g2].e12();
#endif

        else
#ifdef NULLSPACE_SPARSE
          ;
#else
          mat(i, j) = 0.0;
#endif
      }
    }

#ifdef NULLSPACE_SPARSE
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
    */

    // Single pass: each valid raise of an inner Gelfand label of state 1 or
    // state 2 yields exactly one parent (row) plus its matrix element, so the
    // triplets are emitted directly. parent_row maps a parent state to its row
    // index, assigned on first appearance -- an O(log n) map lookup replaces the
    // old O(n) std::find dedup, and the separate fill pass is gone. Raise <->
    // operator correspondence: n11<->E12, n12<->E23_12, n22<->E23_22,
    // n13<->E34_13, n23<->E34_23, n33<->E34_33. Each (parent,column) is reachable
    // by at most one raise, so no triplet is emitted twice.
    std::map<std::pair<int, int>, int> parent_row;
    std::vector<Eigen::Triplet<double>> ija;
    int row_num = 0;

    auto add = [&](int pi, int pj, int col, double val)
    {
      auto [it, inserted] = parent_row.try_emplace({pi, pj}, row_num);
      if (inserted) row_num++;
      ija.emplace_back(it->second, col, val);
    };

    for (int k = 0; k < col_num; k++)
    {
      int g1 = indices[k].first, g2 = indices[k].second;
      const u4::gelfand& G1 = gArr1[g1];
      const u4::gelfand& G2 = gArr2[g2];

      // parents of the first state (operator acts on g1, g2 unchanged)
      u4::gelfand t = G1; t.n11(t.n11() + 1); 
      if (t.check()) add(u4::findGelfand(t, gArr1), g2, k, G1.e12());
      t = G1; t.n12(t.n12() + 1);
      if (t.check()) add(u4::findGelfand(t, gArr1), g2, k, G1.e23_12());
      t = G1; t.n22(t.n22() + 1);
      if (t.check()) add(u4::findGelfand(t, gArr1), g2, k, G1.e23_22());
      t = G1; t.n13(t.n13() + 1);
      if (t.check()) add(u4::findGelfand(t, gArr1), g2, k, G1.e34_13());
      t = G1; t.n23(t.n23() + 1);
      if (t.check()) add(u4::findGelfand(t, gArr1), g2, k, G1.e34_23());
      t = G1; t.n33(t.n33() + 1);
      if (t.check()) add(u4::findGelfand(t, gArr1), g2, k, G1.e34_33());

      // parents of the second state (operator acts on g2, g1 unchanged)
      t = G2; t.n11(t.n11() + 1);
      if (t.check()) add(g1, u4::findGelfand(t, gArr2), k, G2.e12());
      t = G2; t.n12(t.n12() + 1);
      if (t.check()) add(g1, u4::findGelfand(t, gArr2), k, G2.e23_12());
      t = G2; t.n22(t.n22() + 1);
      if (t.check()) add(g1, u4::findGelfand(t, gArr2), k, G2.e23_22());
      t = G2; t.n13(t.n13() + 1);
      if (t.check()) add(g1, u4::findGelfand(t, gArr2), k, G2.e34_13());
      t = G2; t.n23(t.n23() + 1);
      if (t.check()) add(g1, u4::findGelfand(t, gArr2), k, G2.e34_23());
      t = G2; t.n33(t.n33() + 1);
      if (t.check()) add(g1, u4::findGelfand(t, gArr2), k, G2.e34_33());
    }

#ifdef NULLSPACE_SPARSE
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
#else
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(row_num, col_num);
    for (const auto& t : ija) mat(t.row(), t.col()) = t.value();
#endif

    return mat;
  }

  // highest-weight reduced coefficients
  // / n14a n24a n34a n44a ; n14b n24b n34b n44b ||| n14c n24c n34c n44c \
  // \   n13a  n23a  n33a      n13b  n23b n33b   |||   n14c  n24c n34c /_eta,rho

  inline void HighestWeightReducedCoeffs(const std::vector<u4::gelfand>& gArr1,
                                         const std::vector<u4::gelfand>& gArr2,
                                         const std::vector<u4::gelfand>& gArr3,
                                         int etamax, u4::WigU4U3_coeffs& coeffs)
  {
    auto& cgcs = utils::cgcs();
    auto& wigU3U2db = u3::wigU3U2db();

    std::vector<std::pair<int, int>> indices =
        expand_to_uncoupled(gArr3[0], gArr1, gArr2);

    auto mat = E34E23E12(gArr1, gArr2, indices);

#ifdef U4LIB_TIME_LOG
    auto t0 = std::chrono::high_resolution_clock::now();
#endif

#ifdef U4LIB_NULLSPACE_LOG
    std::cout << "mat: " << mat.rows() << " x " << mat.cols() << std::endl;
    std::cout << "\telements: " << mat.rows() * mat.cols() << std::endl;

#ifdef NULLSPACE_SPARSE
    long nnz = mat.nonZeros();
#else
    long nnz = 0;
    for (auto x : mat.reshaped())
      if (x != 0.0)
        nnz++;
#endif
    std::cout << "\tnonzero elements: " << nnz << std::endl;
#endif

#if defined(NULLSPACE_SOLVER_SNS)
    Eigen::MatrixXd nullspace = utils::null_space_sns(mat, etamax);
#else
    Eigen::MatrixXd nullspace = utils::null_space(mat);
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
            auto accU3 = wigU3U2db.get(
                gArr1[i1_].n13(), gArr1[i1_].n23(), gArr1[i1_].n33(),
                gArr1[i1_].n12(), gArr1[i1_].n22(), gArr2[i2_].n13(),
                gArr2[i2_].n23(), gArr2[i2_].n33(), gArr2[i2_].n12(),
                gArr2[i2_].n22(), gArr3[0].n13(), gArr3[0].n23(),
                gArr3[0].n33(), gArr3[0].n12(), gArr3[0].n22());

            if (accU3.size() == 0) continue;

            double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
      std::vector<Eigen::MatrixXd>& E43E42E41,
      u4::WigU4U3_coeffs& coeffs)
  {
    auto& cgcs = utils::cgcs();
    auto& wigU3U2db = u3::wigU3U2db();

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n13p1.Lambdax2(), mm1 = g1_n13p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n13p1.n13(), g1_n13p1.n23(), g1_n13p1.n33(), g1_n13p1.n12(),
              g1_n13p1.n22(), gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(),
              gArr2[g2].n12(), gArr2[g2].n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n13p1.e43_13();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n23p1.Lambdax2(), mm1 = g1_n23p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n23p1.n13(), g1_n23p1.n23(), g1_n23p1.n33(), g1_n23p1.n12(),
              g1_n23p1.n22(), gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(),
              gArr2[g2].n12(), gArr2[g2].n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n23p1.e43_23();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n33p1.Lambdax2(), mm1 = g1_n33p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n33p1.n13(), g1_n33p1.n23(), g1_n33p1.n33(), g1_n33p1.n12(),
              g1_n33p1.n22(), gArr2[g2].n13(), gArr2[g2].n23(), gArr2[g2].n33(),
              gArr2[g2].n12(), gArr2[g2].n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n33p1.e43_33();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n13p1.Lambdax2(), mm2 = g2_n13p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 =
              wigU3U2db.get(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
                            gArr1[g1].n12(), gArr1[g1].n22(), g2_n13p1.n13(),
                            g2_n13p1.n23(), g2_n13p1.n33(), g2_n13p1.n12(),
                            g2_n13p1.n22(), gArr3[parent_g3].n13(),
                            gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
                            gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n13p1.e43_13();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n23p1.Lambdax2(), mm2 = g2_n23p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 =
              wigU3U2db.get(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
                            gArr1[g1].n12(), gArr1[g1].n22(), g2_n23p1.n13(),
                            g2_n23p1.n23(), g2_n23p1.n33(), g2_n23p1.n12(),
                            g2_n23p1.n22(), gArr3[parent_g3].n13(),
                            gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
                            gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n23p1.e43_23();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n33p1.Lambdax2(), mm2 = g2_n33p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 =
              wigU3U2db.get(gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
                            gArr1[g1].n12(), gArr1[g1].n22(), g2_n33p1.n13(),
                            g2_n33p1.n23(), g2_n33p1.n33(), g2_n33p1.n12(),
                            g2_n33p1.n22(), gArr3[parent_g3].n13(),
                            gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
                            gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n33p1.e43_33();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
      std::vector<Eigen::MatrixXd>& E43E42E41,
      u4::WigU4U3_coeffs& coeffs)
  {
    auto& cgcs = utils::cgcs();
    auto& wigU3U2db = u3::wigU3U2db();

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n13n12p1.Lambdax2(), mm1 = g1_n13n12p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n13n12p1.n13(), g1_n13n12p1.n23(), g1_n13n12p1.n33(),
              g1_n13n12p1.n12(), g1_n13n12p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n13n12p1.e42_13_12();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n13n22p1.Lambdax2(), mm1 = g1_n13n22p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n13n22p1.n13(), g1_n13n22p1.n23(), g1_n13n22p1.n33(),
              g1_n13n22p1.n12(), g1_n13n22p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n13n22p1.e42_13_22();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n23n12p1.Lambdax2(), mm1 = g1_n23n12p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n23n12p1.n13(), g1_n23n12p1.n23(), g1_n23n12p1.n33(),
              g1_n23n12p1.n12(), g1_n23n12p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n23n12p1.e42_23_12();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n23n22p1.Lambdax2(), mm1 = g1_n23n22p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n23n22p1.n13(), g1_n23n22p1.n23(), g1_n23n22p1.n33(),
              g1_n23n22p1.n12(), g1_n23n22p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n23n22p1.e42_23_22();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n33n12p1.Lambdax2(), mm1 = g1_n33n12p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n33n12p1.n13(), g1_n33n12p1.n23(), g1_n33n12p1.n33(),
              g1_n33n12p1.n12(), g1_n33n12p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n33n12p1.e42_33_12();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n33n22p1.Lambdax2(), mm1 = g1_n33n22p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n33n22p1.n13(), g1_n33n22p1.n23(), g1_n33n22p1.n33(),
              g1_n33n22p1.n12(), g1_n33n22p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n33n22p1.e42_33_22();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n13n12p1.Lambdax2(), mm2 = g2_n13n12p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n13n12p1.n13(),
              g2_n13n12p1.n23(), g2_n13n12p1.n33(), g2_n13n12p1.n12(),
              g2_n13n12p1.n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n13n12p1.e42_13_12();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n13n22p1.Lambdax2(), mm2 = g2_n13n22p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n13n22p1.n13(),
              g2_n13n22p1.n23(), g2_n13n22p1.n33(), g2_n13n22p1.n12(),
              g2_n13n22p1.n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n13n22p1.e42_13_22();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n23n12p1.Lambdax2(), mm2 = g2_n23n12p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n23n12p1.n13(),
              g2_n23n12p1.n23(), g2_n23n12p1.n33(), g2_n23n12p1.n12(),
              g2_n23n12p1.n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n23n12p1.e42_23_12();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n23n22p1.Lambdax2(), mm2 = g2_n23n22p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n23n22p1.n13(),
              g2_n23n22p1.n23(), g2_n23n22p1.n33(), g2_n23n22p1.n12(),
              g2_n23n22p1.n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n23n22p1.e42_23_22();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n33n12p1.Lambdax2(), mm2 = g2_n33n12p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n33n12p1.n13(),
              g2_n33n12p1.n23(), g2_n33n12p1.n33(), g2_n33n12p1.n12(),
              g2_n33n12p1.n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n33n12p1.e42_33_12();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n33n22p1.Lambdax2(), mm2 = g2_n33n22p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n33n22p1.n13(),
              g2_n33n22p1.n23(), g2_n33n22p1.n33(), g2_n33n22p1.n12(),
              g2_n33n22p1.n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n33n22p1.e42_33_22();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
      std::vector<Eigen::MatrixXd>& E43E42E41,
      u4::WigU4U3_coeffs& coeffs)
  {
    auto& cgcs = utils::cgcs();
    auto& wigU3U2db = u3::wigU3U2db();

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n13n12n11p1.Lambdax2(), mm1 = g1_n13n12n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n13n12n11p1.n13(), g1_n13n12n11p1.n23(), g1_n13n12n11p1.n33(),
              g1_n13n12n11p1.n12(), g1_n13n12n11p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n13n12n11p1.e41_13_12();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n13n22n11p1.Lambdax2(), mm1 = g1_n13n22n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n13n22n11p1.n13(), g1_n13n22n11p1.n23(), g1_n13n22n11p1.n33(),
              g1_n13n22n11p1.n12(), g1_n13n22n11p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n13n22n11p1.e41_13_22();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n23n12n11p1.Lambdax2(), mm1 = g1_n23n12n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n23n12n11p1.n13(), g1_n23n12n11p1.n23(), g1_n23n12n11p1.n33(),
              g1_n23n12n11p1.n12(), g1_n23n12n11p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n23n12n11p1.e41_23_12();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n23n22n11p1.Lambdax2(), mm1 = g1_n23n22n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n23n22n11p1.n13(), g1_n23n22n11p1.n23(), g1_n23n22n11p1.n33(),
              g1_n23n22n11p1.n12(), g1_n23n22n11p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n23n22n11p1.e41_23_22();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n33n12n11p1.Lambdax2(), mm1 = g1_n33n12n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n33n12n11p1.n13(), g1_n33n12n11p1.n23(), g1_n33n12n11p1.n33(),
              g1_n33n12n11p1.n12(), g1_n33n12n11p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n33n12n11p1.e41_33_12();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj1 = g1_n33n22n11p1.Lambdax2(), mm1 = g1_n33n22n11p1.MLambdax2(),
            jj2 = gArr2[g2].Lambdax2(), mm2 = gArr2[g2].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              g1_n33n22n11p1.n13(), g1_n33n22n11p1.n23(), g1_n33n22n11p1.n33(),
              g1_n33n22n11p1.n12(), g1_n33n22n11p1.n22(), gArr2[g2].n13(),
              gArr2[g2].n23(), gArr2[g2].n33(), gArr2[g2].n12(),
              gArr2[g2].n22(), gArr3[parent_g3].n13(), gArr3[parent_g3].n23(),
              gArr3[parent_g3].n33(), gArr3[parent_g3].n12(),
              gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g1_n33n22n11p1.e41_33_22();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n13n12n11p1.Lambdax2(), mm2 = g2_n13n12n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n13n12n11p1.n13(),
              g2_n13n12n11p1.n23(), g2_n13n12n11p1.n33(), g2_n13n12n11p1.n12(),
              g2_n13n12n11p1.n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n13n12n11p1.e41_13_12();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n13n22n11p1.Lambdax2(), mm2 = g2_n13n22n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n13n22n11p1.n13(),
              g2_n13n22n11p1.n23(), g2_n13n22n11p1.n33(), g2_n13n22n11p1.n12(),
              g2_n13n22n11p1.n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n13n22n11p1.e41_13_22();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n23n12n11p1.Lambdax2(), mm2 = g2_n23n12n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n23n12n11p1.n13(),
              g2_n23n12n11p1.n23(), g2_n23n12n11p1.n33(), g2_n23n12n11p1.n12(),
              g2_n23n12n11p1.n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n23n12n11p1.e41_23_12();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n23n22n11p1.Lambdax2(), mm2 = g2_n23n22n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n23n22n11p1.n13(),
              g2_n23n22n11p1.n23(), g2_n23n22n11p1.n33(), g2_n23n22n11p1.n12(),
              g2_n23n22n11p1.n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n23n22n11p1.e41_23_22();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n33n12n11p1.Lambdax2(), mm2 = g2_n33n12n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n33n12n11p1.n13(),
              g2_n33n12n11p1.n23(), g2_n33n12n11p1.n33(), g2_n33n12n11p1.n12(),
              g2_n33n12n11p1.n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n33n12n11p1.e41_33_12();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
        auto accU4 = coeffs.access(u3a, u3b, u3c);

        int jj2 = g2_n33n22n11p1.Lambdax2(), mm2 = g2_n33n22n11p1.MLambdax2(),
            jj1 = gArr1[g1].Lambdax2(), mm1 = gArr1[g1].MLambdax2();

        if (accU4.size() > 0 && (mm1 + mm2) == mm3 &&
            std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
        {
          auto accU3 = wigU3U2db.get(
              gArr1[g1].n13(), gArr1[g1].n23(), gArr1[g1].n33(),
              gArr1[g1].n12(), gArr1[g1].n22(), g2_n33n22n11p1.n13(),
              g2_n33n22n11p1.n23(), g2_n33n22n11p1.n33(), g2_n33n22n11p1.n12(),
              g2_n33n22n11p1.n22(), gArr3[parent_g3].n13(),
              gArr3[parent_g3].n23(), gArr3[parent_g3].n33(),
              gArr3[parent_g3].n12(), gArr3[parent_g3].n22());
          assert(accU3.size() > 0);

          double factor = g2_n33n22n11p1.e41_33_22();

          double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
                                       WigU4U3_coeffs& coeffs)
  {
    auto& cgcs = utils::cgcs();
    auto& wigU3U2db = u3::wigU3U2db();

    /*
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
    */

    // Keep all GT states at the common point on this fixed-w3 slice.
    const int W1 = gArr3[0].n24();
    const int W2 = W1 + gArr3[0].n34();
    std::vector<int> same_weight_gArr3;
    for (int i = 0; i < gArr3.size(); i++)
    {
      if (gArr3[i].n11() == W1 &&
          gArr3[i].n12() + gArr3[i].n22() == W2 &&
          gArr3[i].n13() + gArr3[i].n23() + gArr3[i].n33() == w3)
        same_weight_gArr3.push_back(i);
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
              uncoupled_same_weight_gArr3, gArr1, gArr2, gArr3, b, 
              E43E42E41, coeffs);
        }

        else if (opt == 1)
        {
          E42(parent_g3, row, etamax, same_weight_gArr3,
              uncoupled_same_weight_gArr3, gArr1, gArr2, gArr3, b,
              E43E42E41, coeffs);
        }

        else if (opt == 2)
        {
          E41(parent_g3, row, etamax, same_weight_gArr3,
              uncoupled_same_weight_gArr3, gArr1, gArr2, gArr3, b,
              E43E42E41, coeffs);
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

            if (missing == false) continue;

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
                auto accU3 = wigU3U2db.get(
                    gArr1[i1_].n13(), gArr1[i1_].n23(), gArr1[i1_].n33(),
                    gArr1[i1_].n12(), gArr1[i1_].n22(), gArr2[i2_].n13(),
                    gArr2[i2_].n23(), gArr2[i2_].n33(), gArr2[i2_].n12(),
                    gArr2[i2_].n22(), gArr3[i3].n13(), gArr3[i3].n23(),
                    gArr3[i3].n33(), gArr3[i3].n12(), gArr3[i3].n22());

                if (accU3.size() == 0) continue;

                double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

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
                      const std::vector<u4::gelfand>& gArrC)
  {
    WigU4U3_coeffs coeffs(u4A, u4B, u4C, etamax);

    u4::HighestWeightReducedCoeffs(gArrA, gArrB, gArrC, etamax, coeffs);

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
        u4::LowerWeightReducedCoeffs(gArrA, gArrB, gArrC, etamax, w3, coeffs);
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

  // database for holding coupling coefficients
  class WigU4U3_db
  {
    using db_value_type = Eigen::MatrixXd;

    // using ht_key_type = std::array<uint8_t, 9>;
    using ht_key_type = std::pair<uint64_t, uint8_t>;
    // Shared-pointer values: get_coeffs hands the whole table out with a
    // refcount bump (one concurrent-map visit per U(4) irrep triple); hot
    // consumers (WigU4SU2SU2/U6/Z6) then resolve U(3) blocks locally via
    // WigU4U3_coeffs::find_block. Mirrors u3lib's WigU3U2_db.
    using ht_value_type = std::shared_ptr<const WigU4U3_coeffs>;

#ifdef U4LIB_ENABLE_WIGU4U3_CACHE
    utils::Cache<ht_key_type, ht_value_type> cache_;
#endif

    inline static db_value_type* p_result_ = nullptr;
#pragma omp threadprivate(p_result_)

  public:
    static void initialize()
    {
      assert(p_result_ == nullptr);
      p_result_ = new db_value_type;
    }

    static void finalize()
    {
      assert(p_result_ != nullptr);
      delete p_result_;
      p_result_ = nullptr;
    }

    static void initialize_thread()
    {
#ifdef USE_OPENMP
      if (omp_get_thread_num() == 0)
      {
        assert(p_result_ != nullptr);
      }
      else
      {
        assert(p_result_ == nullptr);
        p_result_ = new db_value_type;
      }
#endif
    }

    static void finalize_thread()
    {
#ifdef USE_OPENMP
      if (omp_get_thread_num() != 0)
      {
        assert(p_result_ != nullptr);
        delete p_result_;
        p_result_ = nullptr;
      }
#endif
    }  
    
  private:
    // This class is a singleton, and cannot be instantiated directly:
    WigU4U3_db() = default;
    WigU4U3_db(const WigU4U3_db&) = delete;
    WigU4U3_db& operator=(const WigU4U3_db&) = delete;

    static ht_value_type compute(const u4::irrep& u4a, 
      const u4::irrep& u4b, const u4::irrep& u4c, int etamax)
    {
      /*
      irrep u4a(ht_key[0], ht_key[1], ht_key[2], 0);
      irrep u4b(ht_key[3], ht_key[4], ht_key[5], 0);
      irrep u4c(ht_key[6], ht_key[7], ht_key[8], 0);
      */
#ifdef TIME_WIGU4U3
      static std::chrono::duration<double> wigu4u3_time(0);
      static double wigu4u3_last_print = 0.0;
      auto start = std::chrono::high_resolution_clock::now();
#endif

      auto gArrA = u4::gelfandBasis(u4a);
      auto gArrB = u4::gelfandBasis(u4b);
      auto gArrC = u4::gelfandBasis(u4c);

      auto result = std::make_shared<const WigU4U3_coeffs>(
          WigU4U3(u4a, u4b, u4c, etamax, -1, gArrA, gArrB, gArrC));

#ifdef TIME_WIGU4U3
      auto end = std::chrono::high_resolution_clock::now();
      wigu4u3_time += end - start; // duration in seconds

      // Print occasionally
      if (wigu4u3_time.count() - wigu4u3_last_print >= 10.0)
      {
        std::cout << "Accumulated WigU4U3 time: " << wigu4u3_time.count()
                  << " s\n";
        wigu4u3_last_print = wigu4u3_time.count();
      }
#endif

      return result;
    }

  public:
    // Fetch the whole coefficient table for a U(4) irrep triple: one
    // concurrent-map visit, shared-pointer handout (no copy). Returns nullptr
    // when the coupling is forbidden (etamax == 0). Hot consumers hold the
    // pointer and resolve U(3) blocks locally with find_block().
    std::shared_ptr<const WigU4U3_coeffs> get_coeffs(
      u4::irrep u4a, u4::irrep u4b, u4::irrep u4c)
    {
      u4a.normalize();
      u4b.normalize();
      u4c.normalize();

      auto etamax = Eta(u4a, u4b, u4c);
      if (etamax == 0)
        return nullptr;

      std::array<utils::UInt8, 9>
        ht_key_safe = {u4a.n14(), u4a.n24(), u4a.n34(),
                       u4b.n14(), u4b.n24(), u4b.n34(),
                       u4c.n14(), u4c.n24(), u4c.n34()};

      uint64_t key_1 =
        (uint64_t(ht_key_safe[0]) << 56) | (uint64_t(ht_key_safe[1]) << 48) |
        (uint64_t(ht_key_safe[2]) << 40) | (uint64_t(ht_key_safe[3]) << 32) |
        (uint64_t(ht_key_safe[4]) << 24) | (uint64_t(ht_key_safe[5]) << 16) |
        (uint64_t(ht_key_safe[6]) << 8) | uint64_t(ht_key_safe[7]);

      ht_key_type ht_key = {key_1, ht_key_safe[8]};

      std::shared_ptr<const WigU4U3_coeffs> result;

#ifdef U4LIB_ENABLE_WIGU4U3_CACHE
      auto producer = [&u4a, &u4b, &u4c, etamax](ht_key_type)
      {
        return compute(u4a, u4b, u4c, etamax);
      };

      auto accessor = [&result](const auto& element)
      { result = element.second; };

      cache_.visit(ht_key, producer, accessor);
#else
      result = compute(u4a, u4b, u4c, etamax);
#endif

      return result;
    }

    db_value_type get(
      int n14a, int n24a, int n34a, int n44a, int n13a, int n23a, int n33a,
      int n14b, int n24b, int n34b, int n44b, int n13b, int n23b, int n33b,
      int n14c, int n24c, int n34c, int n44c, int n13c, int n23c, int n33c)
    {
      assert(n14a >= n24a);
      assert(n24a >= n34a);
      assert(n34a >= n44a);
      assert(n14b >= n24b);
      assert(n24b >= n34b);
      assert(n34b >= n44b);
      assert(n14c >= n24c);
      assert(n24c >= n34c);
      assert(n34c >= n44c);

      irrep u4a(n14a - n44a, n24a - n44a, n34a - n44a, 0);
      irrep u4b(n14b - n44b, n24b - n44b, n34b - n44b, 0);
      irrep u4c(n14c - n44c, n24c - n44c, n34c - n44c, 0);

      u3::irrep u3a(n13a - n44a, n23a - n44a, n33a - n44a);
      u3::irrep u3b(n13b - n44b, n23b - n44b, n33b - n44b);
      u3::irrep u3c(n13c - n44c, n23c - n44c, n33c - n44c);

      auto rhomax = u3::Rho(u3a, u3b, u3c);
      if (rhomax == 0) return db_value_type();

      auto coeffs = get_coeffs(u4a, u4b, u4c);
      if (!coeffs)   // etamax == 0
        return db_value_type();

      db_value_type result = coeffs->access(u3a, u3b, u3c);

      if (result.size() > 0)
        assert(result.rows() == rhomax);

      return result;
    }

    U4U3<coeffs_t> get_U4U3(
      int n14a, int n24a, int n34a, int n44a, int n13a, int n23a, int n33a,
      int n14b, int n24b, int n34b, int n44b, int n13b, int n23b, int n33b,
      int n14c, int n24c, int n34c, int n44c, int n13c, int n23c, int n33c)
    {
      Eigen::MatrixXd mat = get(n14a, n24a, n34a, n44a, n13a, n23a, n33a,
                                n14b, n24b, n34b, n44b, n13b, n23b, n33b,
                                n14c, n24c, n34c, n44c, n13c, n23c, n33c);
      // cols() == etamax; keep >= 1 so an empty (forbidden) result does not
      // divide by zero in U4U3::rhomax().
      const size_t etamax =
          mat.cols() > 0 ? static_cast<size_t>(mat.cols()) : size_t{1};
      return U4U3<coeffs_t>(std::move(mat), etamax);
    }

    friend WigU4U3_db& wigU4U3db();

#ifdef U4LIB_ENABLE_WIGU4U3_CACHE
    void clear() { cache_.clear(); }
#endif
  };

  inline WigU4U3_db& wigU4U3db()
  {
    static WigU4U3_db wigU4U3db;

    return wigU4U3db;
  }

} // namespace u4

#endif

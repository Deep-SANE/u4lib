// protection against multiple header file inclusion
#ifndef U4LIB_U4_U9COEFFS_H
#define U4LIB_U4_U9COEFFS_H

#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "U4decomposition.h"
#include "U4_U6Z6Coeffs.h"
#include "U4basis.h"

#include <qcx_ND.h>
#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>

namespace u4
{

  // input by users: u4a, u4b, u4c, u4d, u4ab, u4cd, u4ac, u4bd, u4abcd
  // all normalized
  inline std::vector<double>
  U9(const u4::irrep& u4A, const std::vector<u4::gelfand>& gArrA,
     const u4::irrep& u4B, const std::vector<u4::gelfand>& gArrB,
     const u4::irrep& u4C, const std::vector<u4::gelfand>& gArrC,
     const u4::irrep& u4D, const std::vector<u4::gelfand>& gArrD,
     const u4::irrep& u4AB, const std::vector<u4::gelfand>& gArrAB,
     const u4::irrep& u4CD, const std::vector<u4::gelfand>& gArrCD,
     const u4::irrep& u4AC, const std::vector<u4::gelfand>& gArrAC,
     const u4::irrep& u4BD, const std::vector<u4::gelfand>& gArrBD,
     const u4::irrep& u4ABCD, const std::vector<u4::gelfand>& gArrABCD,
     int etaABmax, int etaCDmax, int etaAB_CDmax, int etaACmax, int etaBDmax,
     int etaAC_BDmax, qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah& racahs,
     u3::WigU3U2_db& wigU3U2db, u4::WigU4U3_db& wigU4U3db)
  {
    std::vector<double> U(etaACmax * etaBDmax * etaAB_CDmax * etaABmax *
                              etaCDmax * etaAC_BDmax,
                          0.0);

    // couple u4AB and u4C
    auto [u4AB_C, etamaxsAB_C] = u4::decomposition(u4AB, u4C);

    /* an array to store the couple I-II-III irreps
    std::vector<u4::irrep> u4ABC;
    std::vector<int> etaAB_Cmax, etaAC_Bmax;    

    for (int i = 0; i < u4AB_C.size(); i++)
    {
      int etamaxAC_B = u4::Eta(u4AC, u4B, u4AB_C[i]);
      if (etamaxAC_B > 0)
      {
        u4ABC.push_back(u4AB_C[i]);
        etaAB_Cmax.push_back(etamaxsAB_C[i]);
        etaAC_Bmax.push_back(etamaxAC_B);
      }
    } */

    //for (int i = 0; i < u4ABC.size(); i++)
    for (int i = 0; i < u4AB_C.size(); i++)
    {
      int etamaxAC_B = u4::Eta(u4AC, u4B, u4AB_C[i]);
      if (etamaxAC_B == 0)
        continue;

      int etaABC_Dmax = u4::Eta(u4AB_C[i], u4D, u4ABCD);
      if (etaABC_Dmax == 0)
        continue;

      std::vector<u4::gelfand> gArrABC = u4::gelfandBasis(u4AB_C[i]);

      std::vector<double> U6_1 =
          u4::U6(u4AC, gArrAC, u4B, gArrB, u4ABCD, gArrABCD, u4D, gArrD,
                 u4AB_C[i], gArrABC, etamaxAC_B, etaABC_Dmax, u4BD, gArrBD,
                 etaBDmax, etaAC_BDmax, cgcs, racahs, wigU3U2db, wigU4U3db);

      std::vector<double> U6_2 =
          u4::U6(u4AB, gArrAB, u4C, gArrC, u4ABCD, gArrABCD, u4D, gArrD,
                 u4AB_C[i], gArrABC, etamaxsAB_C[i], etaABC_Dmax, u4CD, gArrCD,
                 etaCDmax, etaAB_CDmax, cgcs, racahs, wigU3U2db, wigU4U3db);

      std::vector<double> Z6 =
          u4::Z6(u4B, gArrB, u4A, gArrA, u4AB_C[i], gArrABC, u4C, gArrC, u4AB,
                 gArrAB, etaABmax, etamaxsAB_C[i], u4AC, gArrAC, etaACmax,
                 etamaxAC_B, cgcs, racahs, wigU3U2db, wigU4U3db);

      for (int eAC = 0; eAC < etaACmax; eAC++)
      {
        for (int eBD = 0; eBD < etaBDmax; eBD++)
        {
          for (int eAB_CD = 0; eAB_CD < etaAB_CDmax; eAB_CD++)
          {
            for (int eAB = 0; eAB < etaABmax; eAB++)
            {
              for (int eCD = 0; eCD < etaCDmax; eCD++)
              {
                for (int eAC_BD = 0; eAC_BD < etaAC_BDmax; eAC_BD++)
                {
                  for (int eAC_B = 0; eAC_B < etamaxAC_B; eAC_B++)
                  {
                    for (int eAB_C = 0; eAB_C < etamaxsAB_C[i]; eAB_C++)
                    {
                      for (int eABC_D = 0; eABC_D < etaABC_Dmax; eABC_D++)
                      {
                        U[((((eAC * etaBDmax + eBD) * etaAB_CDmax + eAB_CD) *
                        etaABmax + eAB) * etaCDmax + eCD) * etaAC_BDmax + eAC_BD] 
                        += U6_1[((eAC_B * etaABC_Dmax + eABC_D) * etaBDmax +
                                  eBD) * etaAC_BDmax + eAC_BD] 
                         * Z6[((eAB * etamaxsAB_C[i] + eAB_C) * etaACmax +
                                eAC) * etamaxAC_B + eAC_B] 
                         * U6_2[((eAB_C * etaABC_Dmax + eABC_D) * etaCDmax +
                                  eCD) * etaAB_CDmax + eAB_CD];
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

    for (int i = 0; i < U.size(); i++)
      if (std::fabs(U[i]) < 1e-14)
        U[i] = 0.0;

    return U;
  }

  class U4U9
  {
    std::vector<double> coeffs_;
    qcx_utils::NDIndex ndi_;

  public:
    U4U9()
    {
      ndi_.add_dimensions(0, 0, 0, 0, 0, 0);
      assert(coeffs_.size() == ndi_.cardinality());
    }

    U4U9(const std::vector<double>& coeffs, size_t etaACmax, size_t etaBDmax,
         size_t etaABCDmax, size_t etaABmax, size_t etaCDmax, size_t etaACBDmax)
        : coeffs_(coeffs)
    {
      ndi_.add_dimensions(etaACmax, etaBDmax, etaABCDmax, etaABmax, etaCDmax,
                          etaACBDmax);
      assert(coeffs_.size() == ndi_.cardinality());
    }

    size_t size() const { return coeffs_.size(); }

    bool empty() const { return size() == 0; }

    size_t etaACmax() const { return ndi_.dimension(0); }
    size_t etaBDmax() const { return ndi_.dimension(1); }
    size_t etaABCDmax() const { return ndi_.dimension(2); }
    size_t etaABmax() const { return ndi_.dimension(3); }
    size_t etaCDmax() const { return ndi_.dimension(4); }
    size_t etaACBDmax() const { return ndi_.dimension(5); }

    double operator[](size_t i) const
    {
      assert(i < coeffs_.size());
      return coeffs_[i];
    }

    double get(size_t etaAC, size_t etaBD, size_t etaABCD, size_t etaAB,
               size_t etaCD, size_t etaACBD)
    {
      std::array multi_index = {etaAC, etaBD, etaABCD, etaAB, etaCD, etaACBD};
      size_t i = ndi_.to_linear(multi_index);
      return operator[](i);
    }
  };

#define U4LIB_ENABLE_U4U9_CACHE
  class U4U9_db
  {
#ifdef U4LIB_ENABLE_U4U9_CACHE
    using key_type = std::array<int, 27>;
    std::map<key_type, std::vector<double>> map_;

    std::vector<double> empty_;
#endif

    qcx_utils::SO3_CGC& cgcs_;
    qcx_utils::SO3_Racah& racahs_;
    u3::WigU3U2_db& wigU3U2db_;
    u4::WigU4U3_db& wigU4U3db_;

    auto calculate(const u4::irrep& u4A, const u4::irrep& u4B,
                   const u4::irrep& u4C, const u4::irrep& u4D,
                   const u4::irrep& u4AB, const u4::irrep& u4CD,
                   const u4::irrep& u4AC, const u4::irrep& u4BD,
                   const u4::irrep& u4ABCD, int etaABmax, int etaCDmax,
                   int etaAB_CDmax, int etaACmax, int etaBDmax, int etaAC_BDmax)
    {
      std::vector<u4::gelfand> gArrA = u4::gelfandBasis(u4A),
                               gArrB = u4::gelfandBasis(u4B),
                               gArrC = u4::gelfandBasis(u4C),
                               gArrD = u4::gelfandBasis(u4D),
                               gArrAB = u4::gelfandBasis(u4AB),
                               gArrCD = u4::gelfandBasis(u4CD),
                               gArrAC = u4::gelfandBasis(u4AC),
                               gArrBD = u4::gelfandBasis(u4BD),
                               gArrABCD = u4::gelfandBasis(u4ABCD);

      return U9(u4A, gArrA, u4B, gArrB, u4C, gArrC, u4D, gArrD, u4AB, gArrAB,
                u4CD, gArrCD, u4AC, gArrAC, u4BD, gArrBD, u4ABCD, gArrABCD,
                etaABmax, etaCDmax, etaAB_CDmax, etaACmax, etaBDmax,
                etaAC_BDmax, cgcs_, racahs_, wigU3U2db_, wigU4U3db_);
    }

  public:
    U4U9_db(qcx_utils::SO3_CGC& cgcs, qcx_utils::SO3_Racah& racahs,
            u3::WigU3U2_db& wigU3U2db, u4::WigU4U3_db& wigU4U3db)
        : cgcs_(cgcs),
          racahs_(racahs),
          wigU3U2db_(wigU3U2db),
          wigU4U3db_(wigU4U3db)
    { }

    U4U9 get_U4U9(u4::irrep u4A, u4::irrep u4B, u4::irrep u4C, u4::irrep u4D,
                  u4::irrep u4AB, u4::irrep u4CD, u4::irrep u4AC,
                  u4::irrep u4BD, u4::irrep u4ABCD, int etaACmax, int etaBDmax,
                  int etaABCDmax, int etaABmax, int etaCDmax, int etaACBDmax)
    {
      decltype(auto) coeffs =
          get(u4A, u4B, u4C, u4D, u4AB, u4CD, u4AC, u4BD, u4ABCD, etaABmax,
              etaCDmax, etaABCDmax, etaACmax, etaBDmax, etaACBDmax);

      return {coeffs,
              static_cast<size_t>(etaACmax),
              static_cast<size_t>(etaBDmax),
              static_cast<size_t>(etaABCDmax),
              static_cast<size_t>(etaABmax),
              static_cast<size_t>(etaCDmax),
              static_cast<size_t>(etaACBDmax)};
    }
#ifdef U4LIB_ENABLE_U4U9_CACHE
    const std::vector<double>&
#else
    std::vector<double>
#endif
    get(u4::irrep u4A, u4::irrep u4B, u4::irrep u4C, u4::irrep u4D,
        u4::irrep u4AB, u4::irrep u4CD, u4::irrep u4AC, u4::irrep u4BD,
        u4::irrep u4ABCD, int etaACmax, int etaBDmax, int etaABCDmax,
        int etaABmax, int etaCDmax, int etaACBDmax)
    {
      u4A.normalize();
      u4B.normalize();
      u4C.normalize();
      u4D.normalize();
      u4AB.normalize();
      u4CD.normalize();
      u4AC.normalize();
      u4BD.normalize();
      u4ABCD.normalize();

      assert(etaABmax == Eta(u4A, u4B, u4AB));
      assert(etaCDmax == Eta(u4C, u4D, u4CD));
      assert(etaABCDmax == Eta(u4AB, u4CD, u4ABCD));
      assert(etaACmax == Eta(u4A, u4C, u4AC));
      assert(etaBDmax == Eta(u4B, u4D, u4BD));
      assert(etaACBDmax == Eta(u4AC, u4BD, u4ABCD));

      bool flag = etaABmax > 0 && etaCDmax > 0 && etaACmax > 0 &&
                  etaBDmax > 0 && etaABCDmax > 0 && etaACBDmax > 0;

#ifdef U4LIB_ENABLE_U4U9_CACHE
      if (flag)
      {
        key_type key = {u4A.n14(),    u4A.n24(),    u4A.n34(),   u4B.n14(),
                        u4B.n24(),    u4B.n34(),    u4C.n14(),   u4C.n24(),
                        u4C.n34(),    u4D.n14(),    u4D.n24(),   u4D.n34(),
                        u4AB.n14(),   u4AB.n24(),   u4AB.n34(),  u4CD.n14(),
                        u4CD.n24(),   u4CD.n34(),   u4AC.n14(),  u4AC.n24(),
                        u4AC.n34(),   u4BD.n14(),   u4BD.n24(),  u4BD.n34(),
                        u4ABCD.n14(), u4ABCD.n24(), u4ABCD.n34()};

        auto iter = map_.find(key);
        if (iter == map_.end())
        {
          auto p = map_.try_emplace(
              key, calculate(u4A, u4B, u4C, u4D, u4AB, u4CD, u4AC, u4BD, u4ABCD,
                             etaABmax, etaCDmax, etaABCDmax, etaACmax,
                             etaBDmax, etaACBDmax));
          assert(p.second);
          iter = p.first;
        }

        return iter->second;
      }
      else
        return empty_;
#else
      if (flag)
        return calculate(u4A, u4B, u4C, u4D, u4AB, u4CD, u4AC, u4BD, u4ABCD,
                         etaABmax, etaCDmax, etaABCDmax, etaACmax, etaBDmax,
                         etaACBDmax);
      else
        return {};
#endif
    }

    auto get(int n14a, int n24a, int n34a, int n44a, int n14b, int n24b,
             int n34b, int n44b, int n14c, int n24c, int n34c, int n44c,
             int n14d, int n24d, int n34d, int n44d, int n14ab, int n24ab,
             int n34ab, int n44ab, int n14cd, int n24cd, int n34cd, int n44cd,
             int n14ac, int n24ac, int n34ac, int n44ac, int n14bd, int n24bd,
             int n34bd, int n44bd, int n14abcd, int n24abcd, int n34abcd,
             int n44abcd)
    {
      u4::irrep u4A(n14a - n44a, n24a - n44a, n34a - n44a, 0), 
                u4B(n14b - n44b, n24b - n44b, n34b - n44b, 0), 
                u4C(n14c - n44c, n24c - n44c, n34c - n44c, 0), 
                u4D(n14d - n44d, n24d - n44d, n34d - n44d, 0), 
                u4AB(n14ab - n44ab, n24ab - n44ab, n34ab - n44ab, 0), 
                u4CD(n14cd - n44cd, n24cd - n44cd, n34cd - n44cd, 0), 
                u4AC(n14ac - n44ac, n24ac - n44ac, n34ac - n44ac, 0),
                u4BD(n14bd - n44bd, n24bd - n44bd, n34bd - n44bd, 0),
                u4ABCD(n14abcd - n44abcd, n24abcd - n44abcd, n34abcd - n44abcd, 0);

      int etaABmax = Eta(u4A,u4B,u4AB), etaCDmax = Eta(u4C,u4D,u4CD),
          etaACmax = Eta(u4A,u4C,u4AC), etaBDmax = Eta(u4B,u4D,u4BD),
          etaAB_CDmax = Eta(u4AB,u4CD,u4ABCD), 
          etaAC_BDmax = Eta(u4AC,u4BD,u4ABCD);

      bool flag = false;
      std::vector<double> c;

      if (etaABmax > 0 && etaCDmax > 0 && etaACmax > 0 && etaBDmax > 0
          && etaAB_CDmax > 0 && etaAC_BDmax > 0)
      {
#ifdef U4LIB_ENABLE_U4U9_CACHE
        key_type key = {u4A.n14(), u4A.n24(), u4A.n34(),
                        u4B.n14(), u4B.n24(), u4B.n34(),
                        u4C.n14(), u4C.n24(), u4C.n34(),
                        u4D.n14(), u4D.n24(), u4D.n34(),
                        u4AB.n14(), u4AB.n24(), u4AB.n34(),
                        u4CD.n14(), u4CD.n24(), u4CD.n34(),
                        u4AC.n14(), u4AC.n24(), u4AC.n34(),
                        u4BD.n14(), u4BD.n24(), u4BD.n34(),
                        u4ABCD.n14(), u4ABCD.n24(), u4ABCD.n34()};
        auto iter = map_.find(key);
        if (iter == map_.end())
        {
          auto p = map_.try_emplace(key, 
            calculate(u4A, u4B, u4C, u4D, u4AB, u4CD, u4AC, u4BD, u4ABCD, 
              etaABmax,etaCDmax,etaAB_CDmax,etaACmax,etaBDmax,etaAC_BDmax));
          assert(p.second);
          iter = p.first;
        }

        c = iter->second;
#else
        c = calculate(u4A, u4B, u4C, u4D, u4AB, u4CD, u4AC, u4BD, u4ABCD, 
              etaABmax,etaCDmax,etaAB_CDmax,etaACmax,etaBDmax,etaAC_BDmax);
#endif
        flag = true;
      }

      return std::make_pair(flag, c);
    }
  };

} // namespace u4

#endif

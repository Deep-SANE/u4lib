// protection against multiple header file inclusion
#ifndef U4LIB_U4_U9COEFFS_H
#define U4LIB_U4_U9COEFFS_H

#include <cmath>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

// #define TIME_U4U9
#ifdef TIME_U4U9
#include <chrono>
#endif

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
     int etaAC_BDmax)
  {
#ifdef TIME_U4U9
    static std::chrono::duration<double> u4u9_time(0);
    static double u4u9_last_print = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
#endif

    auto& u4u6db = u4::u4u6db();
    auto& u4z6db = u4::u4z6db();

    std::vector<double> U(etaACmax * etaBDmax * etaAB_CDmax * etaABmax *
                              etaCDmax * etaAC_BDmax,
                          0.0);

    // couple u4AB and u4C
    auto [u4AB_C, etamaxsAB_C] = u4::decomposition(u4AB, u4C);

    for (int i = 0; i < u4AB_C.size(); i++)
    {
      int etamaxAC_B = u4::Eta(u4AC, u4B, u4AB_C[i]);
      if (etamaxAC_B == 0)
        continue;

      int etaABC_Dmax = u4::Eta(u4AB_C[i], u4D, u4ABCD);
      if (etaABC_Dmax == 0)
        continue;

      std::vector<u4::gelfand> gArrABC = u4::gelfandBasis(u4AB_C[i]);

#ifdef U4LIB_ENABLE_U4U6_CACHE
      auto U6_1 = u4u6db.get(
        u4AC.n14(), u4AC.n24(), u4AC.n34(), u4AC.n44(),
        u4B.n14(), u4B.n24(), u4B.n34(), u4B.n44(),
        u4ABCD.n14(), u4ABCD.n24(), u4ABCD.n34(), u4ABCD.n44(),
        u4D.n14(), u4D.n24(), u4D.n34(), u4D.n44(),               
        u4AB_C[i].n14(), u4AB_C[i].n24(), u4AB_C[i].n34(), u4AB_C[i].n44(),
        u4BD.n14(), u4BD.n24(), u4BD.n34(), u4BD.n44());
      if (U6_1.empty()) continue;

      auto U6_2 = u4u6db.get(
        u4AB.n14(), u4AB.n24(), u4AB.n34(), u4AB.n44(),
        u4C.n14(), u4C.n24(), u4C.n34(), u4C.n44(),
        u4ABCD.n14(), u4ABCD.n24(), u4ABCD.n34(), u4ABCD.n44(),
        u4D.n14(), u4D.n24(), u4D.n34(), u4D.n44(),  
        u4AB_C[i].n14(), u4AB_C[i].n24(), u4AB_C[i].n34(), u4AB_C[i].n44(),
        u4CD.n14(), u4CD.n24(), u4CD.n34(), u4CD.n44());
      if (U6_2.empty()) continue;
#else
      std::vector<double> U6_1 =
          u4::U6(u4AC, gArrAC, u4B, gArrB, u4ABCD, gArrABCD, u4D, gArrD,
                 u4AB_C[i], gArrABC, etamaxAC_B, etaABC_Dmax, u4BD, gArrBD,
                 etaBDmax, etaAC_BDmax);

      std::vector<double> U6_2 =
          u4::U6(u4AB, gArrAB, u4C, gArrC, u4ABCD, gArrABCD, u4D, gArrD,
                 u4AB_C[i], gArrABC, etamaxsAB_C[i], etaABC_Dmax, u4CD, gArrCD,
                 etaCDmax, etaAB_CDmax);
#endif

#ifdef U4LIB_ENABLE_U4Z6_CACHE
      auto Z6 = u4z6db.get(
        u4B.n14(), u4B.n24(), u4B.n34(), u4B.n44(),
        u4A.n14(), u4A.n24(), u4A.n34(), u4A.n44(),
        u4AB_C[i].n14(), u4AB_C[i].n24(), u4AB_C[i].n34(), u4AB_C[i].n44(),
        u4C.n14(), u4C.n24(), u4C.n34(), u4C.n44(),
        u4AB.n14(), u4AB.n24(), u4AB.n34(), u4AB.n44(),
        u4AC.n14(), u4AC.n24(), u4AC.n34(), u4AC.n44());
      if (Z6.empty()) continue;
#else
      std::vector<double> Z6 =
          u4::Z6(u4B, gArrB, u4A, gArrA, u4AB_C[i], gArrABC, u4C, gArrC, u4AB,
                 gArrAB, etaABmax, etamaxsAB_C[i], u4AC, gArrAC, etaACmax,
                 etamaxAC_B);
#endif

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

#ifdef TIME_U4U9
    auto end = std::chrono::high_resolution_clock::now();
    u4u9_time += end - start;  // duration in seconds

    // Print occasionally 
    if (u4u9_time.count() - u4u9_last_print >= 1.0) 
    {
        std::cout << "Accumulated U4U9 time: " << u4u9_time.count() << " s\n";
        u4u9_last_print = u4u9_time.count();
    }
#endif

    return U;
  }

  template <typename T>
  class U4U9
  {
    T coeffs_;
    qcx_utils::NDIndex6 ndi_;

  public:
    U4U9()
    {
      ndi_.add_dimensions(0, 0, 0, 0, 0, 0);
      assert(coeffs_.size() == ndi_.cardinality());
    }

    U4U9(T&& coeffs, size_t etaACmax, size_t etaBDmax,
         size_t etaABCDmax, size_t etaABmax, size_t etaCDmax, size_t etaACBDmax)
        : coeffs_(std::move(coeffs))
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
    using key_type = std::array<uint8_t, 27>;
    using value_type = std::vector<double>;

#ifdef U4LIB_ENABLE_U4U9_CACHE
    qcx_utils::Cache<key_type, value_type> cache_;
    using return_type = std::span<const double>;
#else
    using return_type = std::vector<double>;
#endif

    inline static value_type* p_result_ = nullptr;
#pragma omp threadprivate(p_result_)

  public:
    static void initialize()
    {
      assert(p_result_ == nullptr);
      p_result_ = new value_type;
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
        p_result_ = new value_type;
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
    U4U9_db() = default;
    U4U9_db(const U4U9_db&) = delete;
    U4U9_db& operator=(const U4U9_db&) = delete;

    static value_type compute(
      const u4::irrep& u4A, const u4::irrep& u4B,
      const u4::irrep& u4C, const u4::irrep& u4D,
      const u4::irrep& u4AB, const u4::irrep& u4CD,
      const u4::irrep& u4AC, const u4::irrep& u4BD,
      const u4::irrep& u4ABCD, 
      int etaABmax, int etaCDmax, int etaABCDmax, 
      int etaACmax, int etaBDmax, int etaACBDmax)
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
                etaABmax, etaCDmax, etaABCDmax, etaACmax, etaBDmax,
                etaACBDmax);
    }

  public:
    // pass by references to avoid copies
    U4U9<return_type> get_U4U9(
      const u4::irrep& u4A, const u4::irrep& u4B,
      const u4::irrep& u4C, const u4::irrep& u4D,
      const u4::irrep& u4AB, const u4::irrep& u4CD,
      const u4::irrep& u4AC, const u4::irrep& u4BD, 
      const u4::irrep& u4ABCD,
      int etaACmax, int etaBDmax, int etaABCDmax,
      int etaABmax, int etaCDmax, int etaACBDmax)
    {
      if ((etaACmax == 0) || (etaBDmax == 0) || (etaABCDmax == 0) ||
          (etaABmax == 0) || (etaCDmax == 0) || (etaACBDmax == 0))
        return {};

      auto coeffs =
          get(u4A, u4B, u4C, u4D, u4AB, u4CD, u4AC, u4BD, u4ABCD, etaABmax,
              etaCDmax, etaABCDmax, etaACmax, etaBDmax, etaACBDmax);

      return {std::move(coeffs),
              static_cast<size_t>(etaACmax),
              static_cast<size_t>(etaBDmax),
              static_cast<size_t>(etaABCDmax),
              static_cast<size_t>(etaABmax),
              static_cast<size_t>(etaCDmax),
              static_cast<size_t>(etaACBDmax)};
    }

    // pass by values to normalize irreps
    return_type get(u4::irrep u4A, u4::irrep u4B, u4::irrep u4C, u4::irrep u4D,
                    u4::irrep u4AB, u4::irrep u4CD, u4::irrep u4AC,
                    u4::irrep u4BD, u4::irrep u4ABCD, int etaABmax,
                    int etaCDmax, int etaABCDmax, int etaACmax, int etaBDmax,
                    int etaACBDmax)
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

      if ((etaABmax == 0) || (etaCDmax == 0) || (etaABCDmax == 0) ||
          (etaACmax == 0) || (etaBDmax == 0) || (etaACBDmax == 0))
        return {};

#ifdef U4LIB_ENABLE_U4U9_CACHE
      std::array<qcx_utils::UInt8, 27> safe_key = {
        u4A.n14(),    u4A.n24(),    u4A.n34(),   u4B.n14(),
        u4B.n24(),    u4B.n34(),    u4C.n14(),   u4C.n24(),
        u4C.n34(),    u4D.n14(),    u4D.n24(),   u4D.n34(),
        u4AB.n14(),   u4AB.n24(),   u4AB.n34(),  u4CD.n14(),
        u4CD.n24(),   u4CD.n34(),   u4AC.n14(),  u4AC.n24(),
        u4AC.n34(),   u4BD.n14(),   u4BD.n24(),  u4BD.n34(),
        u4ABCD.n14(), u4ABCD.n24(), u4ABCD.n34()};

      key_type key = {safe_key[0], safe_key[1], safe_key[2],
                      safe_key[3], safe_key[4], safe_key[5],
                      safe_key[6], safe_key[7], safe_key[8],
                      safe_key[9], safe_key[10], safe_key[11],
                      safe_key[12], safe_key[13], safe_key[14],
                      safe_key[15], safe_key[16], safe_key[17],
                      safe_key[18], safe_key[19], safe_key[20],
                      safe_key[21], safe_key[22], safe_key[23],
                      safe_key[24], safe_key[25], safe_key[26]};

      auto producer = [&u4A, &u4B, &u4C, &u4D, &u4AB, &u4CD, &u4AC, &u4BD, &u4ABCD, 
                       &etaABmax, &etaCDmax, &etaABCDmax, 
                       &etaACmax, &etaBDmax, &etaACBDmax](key_type)
      { 
        return compute(u4A, u4B, u4C, u4D, u4AB, u4CD, u4AC, u4BD, u4ABCD, 
                       etaABmax, etaCDmax, etaABCDmax, 
                       etaACmax, etaBDmax, etaACBDmax);
      };

      auto accessor = [](const auto& element) { *p_result_ = element.second; };

      cache_.visit(key, producer, accessor);
#else
      *p_result_ = compute(u4A, u4B, u4C, u4D, u4AB, u4CD, u4AC, u4BD, u4ABCD, 
                           etaABmax, etaCDmax, etaABCDmax, 
                           etaACmax, etaBDmax, etaACBDmax);
#endif

      return *p_result_;
    }

    // TO BE REMOVED IN THE FUTURE:
    value_type get(
      int n14a, int n24a, int n34a, int n44a, int n14b, int n24b,
      int n34b, int n44b, int n14c, int n24c, int n34c, int n44c,
      int n14d, int n24d, int n34d, int n44d, int n14ab, int n24ab,
      int n34ab, int n44ab, int n14cd, int n24cd, int n34cd, int n44cd,
      int n14ac, int n24ac, int n34ac, int n44ac, int n14bd, int n24bd,
      int n34bd, int n44bd, int n14abcd, int n24abcd, int n34abcd, int n44abcd)
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
          etaABCDmax = Eta(u4AB,u4CD,u4ABCD), 
          etaACBDmax = Eta(u4AC,u4BD,u4ABCD);

      if ((etaABmax == 0) || (etaCDmax == 0) || (etaABCDmax == 0) ||
          (etaACmax == 0) || (etaBDmax == 0) || (etaACBDmax == 0))
        return {};

#ifdef U4LIB_ENABLE_U4U9_CACHE
      std::array<qcx_utils::UInt8, 27> safe_key = {
        u4A.n14(),    u4A.n24(),    u4A.n34(),   u4B.n14(),
        u4B.n24(),    u4B.n34(),    u4C.n14(),   u4C.n24(),
        u4C.n34(),    u4D.n14(),    u4D.n24(),   u4D.n34(),
        u4AB.n14(),   u4AB.n24(),   u4AB.n34(),  u4CD.n14(),
        u4CD.n24(),   u4CD.n34(),   u4AC.n14(),  u4AC.n24(),
        u4AC.n34(),   u4BD.n14(),   u4BD.n24(),  u4BD.n34(),
        u4ABCD.n14(), u4ABCD.n24(), u4ABCD.n34()};

      key_type key = {safe_key[0], safe_key[1], safe_key[2],
                      safe_key[3], safe_key[4], safe_key[5],
                      safe_key[6], safe_key[7], safe_key[8],
                      safe_key[9], safe_key[10], safe_key[11],
                      safe_key[12], safe_key[13], safe_key[14],
                      safe_key[15], safe_key[16], safe_key[17],
                      safe_key[18], safe_key[19], safe_key[20],
                      safe_key[21], safe_key[22], safe_key[23],
                      safe_key[24], safe_key[25], safe_key[26]};

      auto producer = [&u4A, &u4B, &u4C, &u4D, &u4AB, &u4CD, &u4AC, &u4BD, &u4ABCD, 
                       &etaABmax, &etaCDmax, &etaABCDmax, 
                       &etaACmax, &etaBDmax, &etaACBDmax](key_type)
      { 
        return compute(u4A, u4B, u4C, u4D, u4AB, u4CD, u4AC, u4BD, u4ABCD, 
                       etaABmax, etaCDmax, etaABCDmax, 
                       etaACmax, etaBDmax, etaACBDmax);
      };

      auto accessor = [](const auto& element) { *p_result_ = element.second; };

      cache_.visit(key, producer, accessor);
#else
      *p_result_ = compute(u4A, u4B, u4C, u4D, u4AB, u4CD, u4AC, u4BD, u4ABCD, 
                           etaABmax, etaCDmax, etaABCDmax, 
                           etaACmax, etaBDmax, etaACBDmax);
#endif

      return *p_result_;
    }

    friend U4U9_db& u4u9db();
  };

  inline U4U9_db& u4u9db()
  {
    static U4U9_db u4u9db;

    return u4u9db;
  }

} // namespace u4

#endif

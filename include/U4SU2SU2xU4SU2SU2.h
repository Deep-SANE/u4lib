#ifndef U4LIB_U4SU2SU2XU4SU2SU2_H
#define U4LIB_U4SU2SU2XU4SU2SU2_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "U4basis.h"
#include "U4SU2SU2_to_U4U3U2U1.h"
#include "U4U3xU4U3.h"

#ifdef TIME_WIGU4SU2SU2   // flag in Definitions.h (pulled in via U4basis.h)
#include <chrono>
#endif

#include <qcx_Cache.h>
#include <qcx_Int.h>
#include <so3Coeffs/SO3_CGC.h>

namespace u4
{
  // generate all Wigner U(4)->SU(2)xSU(2) coefficients for multiple
  // values of ss1, tt1, ss2, tt2, ss3, tt3
  inline auto
  WigU4SU2SU2(int etamax, 
              const std::vector<u4::physicalvector>& pArrA,
              const std::vector<u4::physicalvector>& pArrB,
              const std::vector<u4::physicalvector>& pArrC)
  {
    auto& cgcs = qcx_utils::cgcs();
    auto& wigU3U2db = u3::wigU3U2db();
    auto& wigU4U3db = u4::wigU4U3db();

    // build Gelfand basis
    std::vector<u4::gelfand> gArrA = u4::gelfandBasis(pArrA[0].irr()),
                             gArrB = u4::gelfandBasis(pArrB[0].irr()),
                             gArrC = u4::gelfandBasis(pArrC[0].irr());

    // the U4U3 table is keyed by the U(4) irrep triple only -- fetch it once
    // for the whole function and resolve U(3) blocks locally
    auto u4CoeffsTable =
        wigU4U3db.get_coeffs(pArrA[0].irr(), pArrB[0].irr(), pArrC[0].irr());
    assert(u4CoeffsTable);

    // find transformation for physical states belong to the 1st irrep
    std::vector<std::vector<int>> canoBasisA(pArrA.size());
    std::vector<Eigen::MatrixXd> transformationA(pArrA.size());
    for (int i = 0; i < pArrA.size(); i++)
      std::tie(canoBasisA[i], transformationA[i]) =
        u4::phys_to_cano(pArrA[i], pArrA[i].SS(), pArrA[i].TT(), gArrA);

    // find transformation for physical states belong to the 3rd irrep
    std::vector<std::vector<int>> canoBasisC(pArrC.size());
    std::vector<Eigen::MatrixXd> transformationC(pArrC.size());
    for (int i = 0; i < pArrC.size(); i++)
      std::tie(canoBasisC[i], transformationC[i]) =
        u4::phys_to_cano(pArrC[i], pArrC[i].SS(), pArrC[i].TT(), gArrC);

    std::vector<std::tuple<int, int, int>> tArr;
    std::vector<U4SU2SU2<coeffs_t>> coeffs;

    int i, j, k;
    for (j = 0; j < pArrB.size(); j++)
    {
      const auto& pB = pArrB[j];
      int SSB = pB.SS(), TTB = pB.TT(), ZB = pB.zeta();

      auto [canoBasisB, transformationB] = u4::phys_to_cano_allM(pB, gArrB);    
 
      for (i = 0; i < pArrA.size(); i++)
      {
        const auto& pA = pArrA[i];
        int SSA = pA.SS(), TTA = pA.TT(), ZA = pA.zeta();
        const auto& canoA = canoBasisA[i];
        const auto& transA = transformationA[i];

        for (k = 0; k < pArrC.size(); k++)
        {
          const auto& pC = pArrC[k];
          int SSC = pC.SS(), TTC = pC.TT(), ZC = pC.zeta();
          const auto& canoC = canoBasisC[k];
          const auto& transC = transformationC[k];

          if (std::fabs(SSA - SSB) <= SSC && (SSA + SSB) >= SSC && 
              std::fabs(TTA - TTB) <= TTC && (TTA + TTB) >= TTC)
          {
            tArr.emplace_back(i,j,k);

            U4SU2SU2<coeffs_t> coeffs_temp(ZA, ZB, ZC, etamax);

            int mmsB = SSC - SSA, mmtB = TTC - TTA;
            int indexS = (SSB - mmsB) / 2, indexT = (TTB - mmtB) / 2;
            const auto& canoB = canoBasisB[indexS][indexT];
            const auto& transB = transformationB[indexS][indexT];

            double cgcS = cgcs.get({SSA, SSB, SSC, SSA, mmsB, SSC}),
                   cgcT = cgcs.get({TTA, TTB, TTC, TTA, mmtB, TTC});

            for (int iA = 0; iA < canoA.size(); iA++)
            {
              const auto& gA = gArrA[canoA[iA]];

              for (int iB = 0; iB < canoB.size(); iB++)
              {
                const auto& gB = gArrB[canoB[iB]];

                for (int iC = 0; iC < canoC.size(); iC++)
                {
                  const auto& gC = gArrC[canoC[iC]];

                  auto accU4 = u4CoeffsTable->find_block(
                    u3::irrep(gA.n13() - gA.n44(), gA.n23() - gA.n44(),
                              gA.n33() - gA.n44()),
                    u3::irrep(gB.n13() - gB.n44(), gB.n23() - gB.n44(),
                              gB.n33() - gB.n44()),
                    u3::irrep(gC.n13() - gC.n44(), gC.n23() - gC.n44(),
                              gC.n33() - gC.n44()));
                  if (accU4.size() == 0) continue;

                  int jj1 = gA.Lambdax2(), mm1 = gA.MLambdax2(),
                      jj2 = gB.Lambdax2(), mm2 = gB.MLambdax2(),
                      jj3 = gC.Lambdax2(), mm3 = gC.MLambdax2();

                  if ((mm1 + mm2) == mm3 && 
                       std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
                  {
                    double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

                    auto accU3 = wigU3U2db.get(
                      gA.n13(), gA.n23(), gA.n33(), gA.n12(), gA.n22(), 
                      gB.n13(), gB.n23(), gB.n33(), gB.n12(), gB.n22(),
                      gC.n13(), gC.n23(), gC.n33(), gC.n12(), gC.n22());
                    if (accU3.size() == 0) continue;

                    for (int e = 0; e < etamax; e++)   
                      for (int z1 = 0; z1 < ZA; z1++)
                        for (int z2 = 0; z2 < ZB; z2++)
                          for (int z3 = 0; z3 < ZC; z3++)
                          {
                            double y = 0.0;
                            for (int r = 0; r < accU4.rows(); r++)
                              y += accU4(r,e) * accU3(r);
                            
                            coeffs_temp(z1, z2, z3, e)
                              += y * cgc * transA(iA, z1)
                               * transB(iB, z2) * transC(iC, z3);
                          }
                  }
                }
              }
            }

            /*
            coeffs_temp /= (cgcS * cgcT);
            for (int row = 0; row < coeffs_temp.rows(); row++)
              for (int col = 0; col < coeffs_temp.cols(); col++)
                if (std::fabs(coeffs_temp(row,col)) <= 1e-12)
                  coeffs_temp(row,col) = 0.0;
            */

            for (size_t i = 0; i < coeffs_temp.size(); i++)
            {
              if (std::fabs(coeffs_temp[i]) <= 1e-12) coeffs_temp[i] = 0.0;
              else coeffs_temp[i] /= (cgcS * cgcT);
            }

            coeffs.push_back(coeffs_temp);
          }
        }
      }
    }
    return std::make_pair(tArr, coeffs);
  }

  // database for holding coupling coefficents
  class WigU4SU2SU2_db
  {
    // using key_type = std::array<uint8_t, 15>;
    using key_type = std::array<uint64_t, 2>;
    using value_type = coeffs_t; //Eigen::MatrixXd;

#ifdef U4LIB_ENABLE_WIGU4SU2SU2_CACHE
    qcx_utils::Cache<key_type, value_type> cache_;
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
    WigU4SU2SU2_db() = default;
    WigU4SU2SU2_db(const WigU4SU2SU2_db&) = delete;
    WigU4SU2SU2_db& operator=(const WigU4SU2SU2_db&) = delete;

    static value_type compute(int etamax, const u4::physicalvector& pA,
                              const u4::physicalvector& pB,
                              const u4::physicalvector& pC)
    {
#ifdef TIME_WIGU4SU2SU2
      static std::chrono::duration<double> wigu4su2su2_time(0);
      static double wigu4su2su2_last_print = 0.0;
      auto start = std::chrono::high_resolution_clock::now();
#endif

      auto& cgcs = qcx_utils::cgcs();
      auto& wigU3U2db = u3::wigU3U2db();
      auto& wigU4U3db = u4::wigU4U3db();

      // build Gelfand basis
      u4::gelfands gArrA = u4::gelfandBasis(pA.irr()),
                   gArrB = u4::gelfandBasis(pB.irr()),
                   gArrC = u4::gelfandBasis(pC.irr());

      int SSA = pA.SS(), TTA = pA.TT(), ZA = pA.zeta();
      int SSB = pB.SS(), TTB = pB.TT(), ZB = pB.zeta();
      int SSC = pC.SS(), TTC = pC.TT(), ZC = pC.zeta();

      // find transformation for physical states belong to the 1st irrep
      auto [canoA, transA] = u4::phys_to_cano(pA, SSA, TTA, gArrA);

      // find transformation for physical states belong to the 3rd irrep
      auto [canoC, transC] = u4::phys_to_cano(pC, SSC, TTC, gArrC);

      // find transformation for physical states belong to the 2nd irrep
      int mmsB = SSC - SSA, mmtB = TTC - TTA;
      auto [canoB, transB] = u4::phys_to_cano(pB, mmsB, mmtB, gArrB);

      double cgcS = cgcs.get({SSA, SSB, SSC, SSA, mmsB, SSC}),
             cgcT = cgcs.get({TTA, TTB, TTC, TTA, mmtB, TTC});

      U4SU2SU2<coeffs_t> coeffs(ZA, ZB, ZC, etamax);

      // The U4U3 cache key is the U(4) irrep triple, constant for the whole
      // Gelfand loop nest: fetch the table once, resolve U(3) blocks locally
      // (no per-iteration concurrent-map visits, no MatrixXd copies).
      auto u4Coeffs = wigU4U3db.get_coeffs(pA.irr(), pB.irr(), pC.irr());
      assert(u4Coeffs);   // etamax > 0 guaranteed by the caller

      for (int iA = 0; iA < canoA.size(); iA++)
      {
        const auto& gA = gArrA[canoA[iA]];
        int jj1 = gA.Lambdax2(), mm1 = gA.MLambdax2();

        for (int iB = 0; iB < canoB.size(); iB++)
        {
          const auto& gB = gArrB[canoB[iB]];
          int jj2 = gB.Lambdax2(), mm2 = gB.MLambdax2();

          for (int iC = 0; iC < canoC.size(); iC++)
          {
            const auto& gC = gArrC[canoC[iC]];
            int jj3 = gC.Lambdax2(), mm3 = gC.MLambdax2();

            // Cheap CG selection (weight + triangle) before the expensive
            // wigU4U3 / wigU3U2db lookups: cgc == 0 -> no contribution.
            double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});
            if (std::fabs(cgc) == 0.0) continue;

            auto accU4 = u4Coeffs->find_block(
              u3::irrep(gA.n13() - gA.n44(), gA.n23() - gA.n44(), gA.n33() - gA.n44()),
              u3::irrep(gB.n13() - gB.n44(), gB.n23() - gB.n44(), gB.n33() - gB.n44()),
              u3::irrep(gC.n13() - gC.n44(), gC.n23() - gC.n44(), gC.n33() - gC.n44()));
            if (accU4.size() == 0) continue;

            auto accU3 = wigU3U2db.get(
              gA.n13(), gA.n23(), gA.n33(), gA.n12(), gA.n22(),
              gB.n13(), gB.n23(), gB.n33(), gB.n12(), gB.n22(),
              gC.n13(), gC.n23(), gC.n33(), gC.n12(), gC.n22());
            if (accU3.size() == 0) continue;

            // y(e) = cgc * sum_r accU4(r,e)*accU3(r) depends only on e, so
            // compute it once per e instead of per (z1,z2,z3); hoist trans.
            for (int e = 0; e < etamax; e++)
            {
              double y = 0.0;
              for (int r = 0; r < accU4.rows(); r++)
                y += accU4(r, e) * accU3(r);
              double yc = y * cgc;

              for (int z1 = 0; z1 < ZA; z1++)
              {
                double a = transA(iA, z1);
                for (int z2 = 0; z2 < ZB; z2++)
                {
                  double ab = a * transB(iB, z2);
                  for (int z3 = 0; z3 < ZC; z3++)
                    coeffs(z1, z2, z3, e) += yc * ab * transC(iC, z3);
                }
              }
            }
          }
        }
      }
      /*
      coeffs /= (cgcS * cgcT);
      for (int row = 0; row < coeffs.rows(); row++)
        for (int col = 0; col < coeffs.cols(); col++)
          if (std::fabs(coeffs(row,col)) <= 1e-12)
            coeffs(row,col) = 0.0;
      */

      for (size_t i = 0; i < coeffs.size(); i++)
      {
        if (std::fabs(coeffs[i]) <= 1e-12) coeffs[i] = 0.0;
        else coeffs[i] /= (cgcS * cgcT);
      }

#ifdef TIME_WIGU4SU2SU2
      auto end = std::chrono::high_resolution_clock::now();
      wigu4su2su2_time += end - start; // duration in seconds

      // Print occasionally
      if (wigu4su2su2_time.count() - wigu4su2su2_last_print >= 10.0)
      {
        std::cout << "Accumulated WigU4SU2SU2 time: " << wigu4su2su2_time.count()
                  << " s\n";
        wigu4su2su2_last_print = wigu4su2su2_time.count();
      }
#endif

      return coeffs.data();
    }

  public:
    U4SU2SU2<value_type> get_U4SU2SU2(int P1a, int P2a, int P3a, int SSa, int TTa,
                             int P1b, int P2b, int P3b, int SSb, int TTb,
                             int P1c, int P2c, int P3c, int SSc, int TTc)
    {
      // normalized irreps:
      u4::irrep u4a(P1a, P2a, P3a), u4b(P1b, P2b, P3b), u4c(P1c, P2c, P3c);
      int etamax = Eta(u4a, u4b, u4c);
      u4::physicalvector pA(u4a,SSa,TTa), pB(u4b,SSb,TTb), pC(u4c,SSc,TTc);

      if ((etamax == 0) || (pA.zeta() == 0) || (pB.zeta() == 0) ||
          (pC.zeta() == 0) || (u2::mult(SSa, SSb, SSc) == 0) || 
          (u2::mult(TTa, TTb, TTc) == 0))
        return {};

#ifdef U4LIB_ENABLE_WIGU4SU2SU2_CACHE
      std::array<qcx_utils::UInt8, 15> safe_key = {P1a, P2a, P3a, SSa, TTa,
                                                   P1b, P2b, P3b, SSb, TTb,
                                                   P1c, P2c, P3c, SSc, TTc};

      uint64_t key_1 = 
        (uint64_t(safe_key[0]) << 56) | (uint64_t(safe_key[1]) << 48) |
        (uint64_t(safe_key[2]) << 40) | (uint64_t(safe_key[3]) << 32) |
        (uint64_t(safe_key[4]) << 24) | (uint64_t(safe_key[5]) << 16) |
        (uint64_t(safe_key[6]) << 8) | uint64_t(safe_key[7]);

      uint64_t key_2 = 
        (uint64_t(safe_key[8]) << 56) | (uint64_t(safe_key[9]) << 48) |
        (uint64_t(safe_key[10]) << 40) | (uint64_t(safe_key[11]) << 32) |
        (uint64_t(safe_key[12]) << 24) | (uint64_t(safe_key[13]) << 16) |
        (uint64_t(safe_key[14]) << 8);

      key_type key = {key_1, key_2};
      /*
      key_type key = {safe_key[0],  safe_key[1],  safe_key[2],  safe_key[3],
                      safe_key[4],  safe_key[5],  safe_key[6],  safe_key[7],
                      safe_key[8],  safe_key[9],  safe_key[10], safe_key[11],
                      safe_key[12], safe_key[13], safe_key[14]};
      */
      auto producer = [etamax, &pA, &pB, &pC](key_type)
      { return compute(etamax, pA, pB, pC); };

      auto accessor = [](const auto& element) { *p_result_ = element.second; };

      cache_.visit(key, producer, accessor);
#else
      *p_result_ = compute(etamax, pA, pB, pC);
#endif
      return {std::move(*p_result_), 
              static_cast<size_t>(pA.zeta()),
              static_cast<size_t>(pB.zeta()),
              static_cast<size_t>(pC.zeta()), 
              static_cast<size_t>(etamax)};
    }

    value_type get(int P1a, int P2a, int P3a, int SSa, int TTa,
                            int P1b, int P2b, int P3b, int SSb, int TTb,
                            int P1c, int P2c, int P3c, int SSc, int TTc)
    {
      // normalized irreps:
      u4::irrep u4a(P1a, P2a, P3a), u4b(P1b, P2b, P3b), u4c(P1c, P2c, P3c);
      int etamax = Eta(u4a, u4b, u4c);
      u4::physicalvector pA(u4a,SSa,TTa), pB(u4b,SSb,TTb), pC(u4c,SSc,TTc);

      if ((etamax == 0) || (pA.zeta() == 0) || (pB.zeta() == 0) ||
          (pC.zeta() == 0) || (u2::mult(SSa, SSb, SSc) == 0) || 
          (u2::mult(TTa, TTb, TTc) == 0))
        return {};

#ifdef U4LIB_ENABLE_WIGU4SU2SU2_CACHE
      std::array<qcx_utils::UInt8, 15> safe_key = {P1a, P2a, P3a, SSa, TTa,
                                                   P1b, P2b, P3b, SSb, TTb,
                                                   P1c, P2c, P3c, SSc, TTc};

      uint64_t key_1 = 
        (uint64_t(safe_key[0]) << 56) | (uint64_t(safe_key[1]) << 48) |
        (uint64_t(safe_key[2]) << 40) | (uint64_t(safe_key[3]) << 32) |
        (uint64_t(safe_key[4]) << 24) | (uint64_t(safe_key[5]) << 16) |
        (uint64_t(safe_key[6]) << 8) | uint64_t(safe_key[7]);

      uint64_t key_2 = 
        (uint64_t(safe_key[8]) << 56) | (uint64_t(safe_key[9]) << 48) |
        (uint64_t(safe_key[10]) << 40) | (uint64_t(safe_key[11]) << 32) |
        (uint64_t(safe_key[12]) << 24) | (uint64_t(safe_key[13]) << 16) |
        (uint64_t(safe_key[14]) << 8) | 0;

      key_type key = {key_1, key_2};
      /*
      key_type key = {safe_key[0],  safe_key[1],  safe_key[2],  safe_key[3],
                      safe_key[4],  safe_key[5],  safe_key[6],  safe_key[7],
                      safe_key[8],  safe_key[9],  safe_key[10], safe_key[11],
                      safe_key[12], safe_key[13], safe_key[14]};
      */
      auto producer = [etamax, &pA, &pB, &pC](key_type)
      { return compute(etamax, pA, pB, pC); };

      auto accessor = [](const auto& element) { *p_result_ = element.second; };

      cache_.visit(key, producer, accessor);
#else
      *p_result_ = compute(etamax, pA, pB, pC);
#endif
      return *p_result_;
    }

    friend WigU4SU2SU2_db& wigU4SU2SU2db();

#ifdef U4LIB_ENABLE_WIGU4SU2SU2_CACHE
    void clear() { cache_.clear(); }
#endif
  };

  inline WigU4SU2SU2_db& wigU4SU2SU2db()
  {
    static WigU4SU2SU2_db wigU4SU2SU2db;

    return wigU4SU2SU2db;
  }

} // namespace u4

#endif

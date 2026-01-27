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
    std::vector<Eigen::MatrixXd> coeffs;

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

            Eigen::MatrixXd coeffs_temp(ZA * ZB * ZC, etamax);
            coeffs_temp.setZero();

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

                  auto [allowU4, accU4] = wigU4U3db.get(
                    gA.n14(), gA.n24(), gA.n34(), gA.n44(), 
                    gA.n13(), gA.n23(), gA.n33(),
                    gB.n14(), gB.n24(), gB.n34(), gB.n44(),
                    gB.n13(), gB.n23(), gB.n33(),
                    gC.n14(), gC.n24(), gC.n34(), gC.n44(),
                    gC.n13(), gC.n23(), gC.n33());
                  if (allowU4 == false) continue;

                  int jj1 = gA.Lambdax2(), mm1 = gA.MLambdax2(),
                      jj2 = gB.Lambdax2(), mm2 = gB.MLambdax2(),
                      jj3 = gC.Lambdax2(), mm3 = gC.MLambdax2();

                  if ((mm1 + mm2) == mm3 && 
                       std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
                  {
                    double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

                    auto [allowU3, accU3] = wigU3U2db.get(
                      gA.n13(), gA.n23(), gA.n33(), gA.n12(), gA.n22(), 
                      gB.n13(), gB.n23(), gB.n33(), gB.n12(), gB.n22(),
                      gC.n13(), gC.n23(), gC.n33(), gC.n12(), gC.n22());
                    if (allowU3 == false) continue;

                    for (int e = 0; e < etamax; e++)   
                      for (int z1 = 0; z1 < ZA; z1++)
                        for (int z2 = 0; z2 < ZB; z2++)
                          for (int z3 = 0; z3 < ZC; z3++)
                          {
                            double y = 0.0;
                            for (int r = 0; r < accU4.rows(); r++)
                              y += accU4(r,e) * accU3(r);
                            
                            coeffs_temp((z1 * ZB + z2) * ZC + z3, e)
                              += y * cgc * transA(iA, z1)
                               * transB(iB, z2) * transC(iC, z3);
                          }
                  }
                }
              }
            }

            coeffs_temp /= (cgcS * cgcT);
            for (int row = 0; row < coeffs_temp.rows(); row++)
              for (int col = 0; col < coeffs_temp.cols(); col++)
                if (std::fabs(coeffs_temp(row,col)) <= 1e-12)
                  coeffs_temp(row,col) = 0.0;

            coeffs.push_back(coeffs_temp);
          }
        }
      }
    }
    return std::make_pair(tArr, coeffs);
  }

#define U4LIB_ENABLE_WIGU4SU2SU2_CACHE

  class WigU4SU2SU2_db
  {
    using key_type = std::array<uint8_t, 15>;
    using value_type = Eigen::MatrixXd;

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
      auto& cgcs = qcx_utils::cgcs();
      auto& wigU3U2db = u3::wigU3U2db();
      auto& wigU4U3db = u4::wigU4U3db();

      // build Gelfand basis
      std::vector<u4::gelfand> gArrA = u4::gelfandBasis(pA.irr()),
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

      Eigen::MatrixXd coeffs(ZA * ZB * ZC, etamax);
      coeffs.setZero();

      for (int iA = 0; iA < canoA.size(); iA++)
      {
        const auto& gA = gArrA[canoA[iA]];

        for (int iB = 0; iB < canoB.size(); iB++)
        {
          const auto& gB = gArrB[canoB[iB]];

          for (int iC = 0; iC < canoC.size(); iC++)
          {
            const auto& gC = gArrC[canoC[iC]];

            auto [allowU4, accU4] = wigU4U3db.get(
              gA.n14(), gA.n24(), gA.n34(), gA.n44(), gA.n13(), gA.n23(), gA.n33(),
              gB.n14(), gB.n24(), gB.n34(), gB.n44(), gB.n13(), gB.n23(), gB.n33(),
              gC.n14(), gC.n24(), gC.n34(), gC.n44(), gC.n13(), gC.n23(), gC.n33());
            if (allowU4 == false) continue;

            int jj1 = gA.Lambdax2(), mm1 = gA.MLambdax2(),
                jj2 = gB.Lambdax2(), mm2 = gB.MLambdax2(),
                jj3 = gC.Lambdax2(), mm3 = gC.MLambdax2();

            if ((mm1 + mm2) == mm3 && 
                 std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
            {
              double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

              auto [allowU3, accU3] = wigU3U2db.get(
                gA.n13(), gA.n23(), gA.n33(), gA.n12(), gA.n22(), 
                gB.n13(), gB.n23(), gB.n33(), gB.n12(), gB.n22(),
                gC.n13(), gC.n23(), gC.n33(), gC.n12(), gC.n22());
              if (allowU3 == false) continue;

              for (int e = 0; e < etamax; e++)   
                for (int z1 = 0; z1 < ZA; z1++)
                  for (int z2 = 0; z2 < ZB; z2++)
                    for (int z3 = 0; z3 < ZC; z3++)
                    {
                      double y = 0.0;
                      for (int r = 0; r < accU4.rows(); r++)
                        y += accU4(r,e) * accU3(r);
                            
                      coeffs((z1 * ZB + z2) * ZC + z3, e) += y * cgc 
                            * transA(iA, z1) * transB(iB, z2) * transC(iC, z3);
                    }
            }
          }
        }
      }

      coeffs /= (cgcS * cgcT);
      for (int row = 0; row < coeffs.rows(); row++)
        for (int col = 0; col < coeffs.cols(); col++)
          if (std::fabs(coeffs(row,col)) <= 1e-12)
            coeffs(row,col) = 0.0;

      return coeffs;
    }

  public:
    std::pair<bool, value_type> get(int P1a, int P2a, int P3a, int SSa, int TTa,
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
        return {false, value_type()};

#ifdef U4LIB_ENABLE_WIGU4SU2SU2_CACHE
      std::array<qcx_utils::UInt8, 15> safe_key = {P1a, P2a, P3a, SSa, TTa,
                                                   P1b, P2b, P3b, SSb, TTb,
                                                   P1c, P2c, P3c, SSc, TTc};

      key_type key = {safe_key[0],  safe_key[1],  safe_key[2],  safe_key[3],
                      safe_key[4],  safe_key[5],  safe_key[6],  safe_key[7],
                      safe_key[8],  safe_key[9],  safe_key[10], safe_key[11],
                      safe_key[12], safe_key[13], safe_key[14]};

      auto producer = [etamax, &pA, &pB, &pC](key_type)
      { return compute(etamax, pA, pB, pC); };

      auto accessor = [](const auto& element) { *p_result_ = element.second; };

      cache_.visit(key, producer, accessor);
#else
      *p_result_ = compute(etamax, pA, pB, pC);
#endif
      return {true, *p_result_};
    }

    friend WigU4SU2SU2_db& wigU4SU2SU2db();
  };

  inline WigU4SU2SU2_db& wigU4SU2SU2db()
  {
    static WigU4SU2SU2_db wigU4SU2SU2db;

    return wigU4SU2SU2db;
  }

} // namespace u4

#endif

/*
  // generate all Wigner U(4)->SU(2)xSU(2) coefficients for multiple
  // values of ss1, tt1, ss2, tt2, ss3, tt3
  inline auto
  WigU4SU2SU2(int etamax, 
              const std::vector<u4::physicalvector>& pArrA,
              const std::vector<u4::physicalvector>& pArrB,
              const std::vector<u4::physicalvector>& pArrC)
  {
    // build Gelfand basis
    std::vector<u4::gelfand> gArrA = u4::gelfandBasis(pArrA[0].irr),
                             gArrB = u4::gelfandBasis(pArrB[0].irr),
                             gArrC = u4::gelfandBasis(pArrC[0].irr);

    // find transformation for physical states belong to the 1st irrep
    std::vector<std::vector<int>> canoBasisA(pArrA.size());
    std::vector<Eigen::MatrixXd> transformationA(pArrA.size());
    for (int i = 0; i < pArrA.size(); i++)
      std::tie(canoBasisA[i], transformationA[i]) =
        u4::phys_to_cano(pArrA[i], pArrA[i].SS, pArrA[i].TT, gArrA);

    // find transformation for physical states belong to the 3rd irrep
    std::vector<std::vector<int>> canoBasisC(pArrC.size());
    std::vector<Eigen::MatrixXd> transformationC(pArrC.size());
    for (int i = 0; i < pArrC.size(); i++)
      std::tie(canoBasisC[i], transformationC[i]) =
        u4::phys_to_cano(pArrC[i], pArrC[i].SS, pArrC[i].TT, gArrC);

    std::vector<std::tuple<int, int, int>> tArr;
    std::vector<Eigen::MatrixXd> coeffs;

    int i, j, k;
    for (j = 0; j < pArrB.size(); j++)
    {
      std::vector<std::vector<std::vector<int>>> canoBasisB(pArrB[j].SS + 1, 
        std::vector<std::vector<int>>(pArrB[j].TT + 1));
      std::vector<std::vector<Eigen::MatrixXd>> transformationB(pArrB[j].SS + 1, 
        std::vector<Eigen::MatrixXd>(pArrB[j].TT + 1));      

      for (int mSx2 = pArrB[j].SS; mSx2 >= -pArrB[j].SS; mSx2 -= 2)
      {
        int indexS = (pArrB[j].SS - mSx2)/2;

        if (mSx2 == pArrB[j].SS)
        {
          std::tie(canoBasisB[0][0], transformationB[0][0]) = 
            u4::phys_to_cano(pArrB[j], pArrB[j].SS, pArrB[j].TT, gArrB);
        }
              
        else if (mSx2 < pArrB[j].SS)
        {
          canoBasisB[indexS][0] =
            u4::expand_to_gelfand(mSx2, pArrB[j].TT, gArrB);

          transformationB[indexS][0] =
            u4::lowerMS(pArrB[j].zeta, pArrB[j].SS, mSx2, gArrB,
            transformationB[indexS-1][0], canoBasisB[indexS-1][0], 
            canoBasisB[indexS][0]);
        }

        for (int mTx2 = pArrB[j].TT - 2; mTx2 >= -pArrB[j].TT; mTx2 -= 2)
        {
          int indexT = (pArrB[j].TT - mTx2)/2;

          canoBasisB[indexS][indexT] =
            u4::expand_to_gelfand(mSx2, mTx2, gArrB);

          transformationB[indexS][indexT] =
            u4::lowerMT(pArrB[j].zeta, pArrB[j].TT, mTx2, gArrB,
            transformationB[indexS][indexT-1], canoBasisB[indexS][indexT-1], 
            canoBasisB[indexS][indexT]);
        }
      }  

      for (i = 0; i < pArrA.size(); i++)
      {
        for (k = 0; k < pArrC.size(); k++)
        {
          if (std::fabs(pArrA[i].SS - pArrB[j].SS) <= pArrC[k].SS 
              && (pArrA[i].SS + pArrB[j].SS) >= pArrC[k].SS && 
              std::fabs(pArrA[i].TT - pArrB[j].TT) <= pArrC[k].TT 
              && (pArrA[i].TT + pArrB[j].TT) >= pArrC[k].TT)
          {
            tArr.emplace_back(i,j,k);

            Eigen::MatrixXd coeffs_temp(pArrA[i].zeta * pArrB[j].zeta 
                                            * pArrC[k].zeta, etamax);
            coeffs_temp.setZero();

            int MSx2 = pArrC[k].SS - pArrA[i].SS,
                MTx2 = pArrC[k].TT - pArrA[i].TT;

            double cgcS = cgcs.get({pArrA[i].SS, pArrB[j].SS, pArrC[k].SS,
                                       pArrA[i].SS, MSx2, pArrC[k].SS}),
                   cgcT = cgcs.get({pArrA[i].TT, pArrB[j].TT, pArrC[k].TT,
                                       pArrA[i].TT, MTx2, pArrC[k].TT});
            
            int indexS = (pArrB[j].SS - MSx2)/2,
                indexT = (pArrB[j].TT - MTx2)/2;

            for (int iA = 0; iA < canoBasisA[i].size(); iA++)
            {
              int gA = canoBasisA[i][iA];
              for (int iB = 0; iB < canoBasisB[indexS][indexT].size(); iB++)
              {
                int gB = canoBasisB[indexS][indexT][iB];
                for (int iC = 0; iC < canoBasisC[k].size(); iC++)
                {
                  int gC = canoBasisC[k][iC];

                  //u3::irrep u3A(gArrA[gA].n13(), gArrA[gA].n23(), gArrA[gA].n33()),
                  //          u3B(gArrB[gB].n13(), gArrB[gB].n23(), gArrB[gB].n33()),
                  //          u3C(gArrC[gC].n13(), gArrC[gC].n23(), gArrC[gC].n33());
                  //auto [exist, accU4] = canoCoeffs.access(u3A,u3B,u3C);

                  auto [exist, accU4] = wigU4U3db.get(
                    pArrA[0].irr.n14(), pArrA[0].irr.n24(), 
                    pArrA[0].irr.n34(), pArrA[0].irr.n44(),
                    gArrA[gA].n13(), gArrA[gA].n23(), gArrA[gA].n33(),
                    pArrB[0].irr.n14(), pArrB[0].irr.n24(), 
                    pArrB[0].irr.n34(), pArrB[0].irr.n44(),
                    gArrB[gB].n13(), gArrB[gB].n23(), gArrB[gB].n33(),
                    pArrC[0].irr.n14(), pArrC[0].irr.n24(), 
                    pArrC[0].irr.n34(), pArrC[0].irr.n44(),
                    gArrC[gC].n13(), gArrC[gC].n23(), gArrC[gC].n33());

                  int jj1 = gArrA[gA].Lambdax2(), mm1 = gArrA[gA].MLambdax2(),
                      jj2 = gArrB[gB].Lambdax2(), mm2 = gArrB[gB].MLambdax2(),
                      jj3 = gArrC[gC].Lambdax2(), mm3 = gArrC[gC].MLambdax2();

                  if (exist == true && (mm1 + mm2) == mm3 && 
                    std::fabs(jj1 - jj2) <= jj3 && (jj1 + jj2) >= jj3)
                  {
                    double cgc = cgcs.get({jj1, jj2, jj3, mm1, mm2, mm3});

                    auto [allowU3, accU3] = wigU3U2db.get(
                      gArrA[gA].n13(), gArrA[gA].n23(), gArrA[gA].n33(), 
                      gArrA[gA].n12(), gArrA[gA].n22(), 
                      gArrB[gB].n13(), gArrB[gB].n23(), gArrB[gB].n33(),                          
                      gArrB[gB].n12(), gArrB[gB].n22(),
                      gArrC[gC].n13(), gArrC[gC].n23(), gArrC[gC].n33(), 
                      gArrC[gC].n12(), gArrC[gC].n22());
                    if (allowU3 == false)
                      continue;

                    for (int e = 0; e < etamax; e++)   
                      for (int z1 = 0; z1 < pArrA[i].zeta; z1++)
                        for (int z2 = 0; z2 < pArrB[j].zeta; z2++)
                          for (int z3 = 0; z3 < pArrC[k].zeta; z3++)
                          {
                            double y = 0.0;
                            for (int r = 0; r < accU4.rows(); r++)
                              y += accU4(r,e) * accU3(r);
                            
                            coeffs_temp((z1 * pArrB[j].zeta + z2) * pArrC[k].zeta + z3, e)
                             += y * cgc * transformationA[i](iA, z1)
                             * transformationB[indexS][indexT](iB, z2)
                             * transformationC[k](iC, z3);
                          }
                  }
                }
              }
            }

            coeffs_temp /= (cgcS * cgcT);
            for (int row = 0; row < coeffs_temp.rows(); row++)
              for (int col = 0; col < coeffs_temp.cols(); col++)
                if (std::fabs(coeffs_temp(row,col)) <= 1e-15)
                  coeffs_temp(row,col) = 0.0;

            coeffs.push_back(coeffs_temp);
          }
        }
      }
    }
    return std::make_pair(tArr, coeffs);
  }
*/

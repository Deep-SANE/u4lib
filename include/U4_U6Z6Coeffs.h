// protection against multiple header file inclusion
#ifndef U4LIB_U4_U6Z6COEFFS_H
#define U4LIB_U4_U6Z6COEFFS_H

#include <cmath>
#include <iostream>
#include <vector>

// #define TIME_U4U6Z6
#ifdef TIME_U4U6Z6
#include <chrono>
#endif

#include <Eigen/Dense>

#include "U4basis.h"
#include "U4U3xU4U3.h"

#include <so3Coeffs/SO3_CGC.h>
#include <so3Coeffs/SO3_Racah.h>

#include <U3_U6Z6Coeffs.h>

namespace u4
{

  // input by users: u4a, u4b, u4abc, u4c, u4ab, u4bc, all normalized
  inline std::vector<double>
  U6(const u4::irrep& u4A, const std::vector<u4::gelfand>& gArrA,
     const u4::irrep& u4B, const std::vector<u4::gelfand>& gArrB,
     const u4::irrep& u4ABC, const std::vector<u4::gelfand>& gArrABC,
     const u4::irrep& u4C, const std::vector<u4::gelfand>& gArrC,
     const u4::irrep& u4AB, const std::vector<u4::gelfand>& gArrAB,
     int etaABmax, int etaAB_Cmax, const u4::irrep& u4BC,
     const std::vector<u4::gelfand>& gArrBC, int etaBCmax, int etaA_BCmax)
  {
#ifdef TIME_U4U6Z6
    static std::chrono::duration<double> u4u6_time(0);
    static double u4u6_last_print = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
#endif

    auto& wigU4U3db = u4::wigU4U3db();

    Eigen::MatrixXd A(etaA_BCmax, etaA_BCmax);
    Eigen::MatrixXd B(etaA_BCmax, etaABmax * etaAB_Cmax * etaBCmax);
    B.setZero();

    int r = 0;
    bool done = false;
    for (int n13A = u4A.n14(); n13A >= u4A.n24() && !done; n13A--)
    {
      for (int n23A = u4A.n24(); n23A >= u4A.n34() && !done; n23A--)
      {
        for (int n33A = u4A.n34(); n33A >= u4A.n44() && !done; n33A--)
        {
          auto [existA_BC, accA_BC] = wigU4U3db.get(
              u4A.n14(), u4A.n24(), u4A.n34(), u4A.n44(), n13A, n23A, n33A,
              u4BC.n14(), u4BC.n24(), u4BC.n34(), u4BC.n44(), u4BC.n14(),
              u4BC.n24(), u4BC.n34(), u4ABC.n14(), u4ABC.n24(), u4ABC.n34(),
              u4ABC.n44(), u4ABC.n14(), u4ABC.n24(), u4ABC.n34());
          if (!existA_BC) continue;
          
          // std::cout << "accA_BC = " << accA_BC << std::endl;
          u3::irrep u3a(n13A - n33A, n23A - n33A, 0),
            u3bc(u4BC.n14() - u4BC.n34(), u4BC.n24() - u4BC.n34(), 0),
            u3abc(u4ABC.n14() - u4ABC.n34(), u4ABC.n24() - u4ABC.n34(), 0);
          // u3a.normalize();
          // u3bc.normalize();
          // u3abc.normalize();

          u3::gelfands gArr3a = u3::gelfandBasis(u3a),
                       gArr3bc = u3::gelfandBasis(u3bc),
                       gArr3abc = u3::gelfandBasis(u3abc);

          for (int rA_BC = 0; rA_BC < accA_BC.rows(); rA_BC++)
          {
            A.row(r) = accA_BC.row(rA_BC);

            for (int n13B = u4B.n14(); n13B >= u4B.n24(); n13B--)
            {
              for (int n23B = u4B.n24(); n23B >= u4B.n34(); n23B--)
              {
                for (int n33B = u4B.n34(); n33B >= u4B.n44(); n33B--)
                {
                  for (int n13AB = u4AB.n14(); n13AB >= u4AB.n24(); n13AB--)
                  {
                    for (int n23AB = u4AB.n24(); n23AB >= u4AB.n34(); n23AB--)
                    {
                      for (int n33AB = u4AB.n34(); n33AB >= u4AB.n44(); n33AB--)
                      {
                        auto [existAB, accAB] = wigU4U3db.get(
                          u4A.n14(), u4A.n24(), u4A.n34(), u4A.n44(), 
                          n13A, n23A, n33A, 
                          u4B.n14(), u4B.n24(), u4B.n34(), u4B.n44(), 
                          n13B, n23B, n33B, u4AB.n14(),
                          u4AB.n24(), u4AB.n34(), u4AB.n44(), 
                          n13AB, n23AB, n33AB);
                        if (!existAB) continue;
                          
                        // std::cout << "accAB = " << accAB << std::endl;
                        u3::irrep u3b(n13B - n33B, n23B - n33B, 0),
                              u3ab(n13AB - n33AB, n23AB - n33AB, 0);
                        // u3b.normalize();
                        // u3ab.normalize();

                        u3::gelfands gArr3b = u3::gelfandBasis(u3b),
                                     gArr3ab = u3::gelfandBasis(u3ab);

                        for (int n13C = u4C.n14(); n13C >= u4C.n24(); n13C--)
                        {
                          for (int n23C = u4C.n24(); n23C >= u4C.n34(); n23C--)
                          {
                            for (int n33C = u4C.n34(); n33C >= u4C.n44(); n33C--)
                            {
                              auto [existAB_C, accAB_C] = wigU4U3db.get(
                                u4AB.n14(), u4AB.n24(), u4AB.n34(), u4AB.n44(),
                                n13AB, n23AB, n33AB,
                                u4C.n14(), u4C.n24(), u4C.n34(), u4C.n44(), 
                                n13C, n23C, n33C, 
                                u4ABC.n14(), u4ABC.n24(), u4ABC.n34(), u4ABC.n44(),
                                u4ABC.n14(), u4ABC.n24(), u4ABC.n34());
                              if (!existAB_C) continue;

                              auto [existBC, accBC] = wigU4U3db.get(
                                u4B.n14(), u4B.n24(), u4B.n34(), u4B.n44(),
                                n13B, n23B, n33B, 
                                u4C.n14(), u4C.n24(), u4C.n34(), u4C.n44(),
                                n13C, n23C, n33C, 
                                u4BC.n14(), u4BC.n24(), u4BC.n34(), u4BC.n44(), 
                                u4BC.n14(), u4BC.n24(), u4BC.n34());
                              if (!existBC) continue;
                                  
                              u3::irrep u3c(n13C - n33C, n23C - n33C, 0);
                              // u3c.normalize();
                              u3::gelfands gArr3c = u3::gelfandBasis(u3c);

                              std::vector<double> U6_coeffs = u3::U6(
                                u3a, gArr3a, u3b, gArr3b, u3abc,
                                gArr3abc, u3c, gArr3c, u3ab, gArr3ab,
                                accAB.rows(), accAB_C.rows(), u3bc,
                                gArr3bc, accBC.rows(), accA_BC.rows());

                              for (int eAB = 0; eAB < etaABmax; eAB++)
                                for (int eAB_C = 0; eAB_C < etaAB_Cmax; eAB_C++)
                                  for (int eBC = 0; eBC < etaBCmax; eBC++)
                                    for (int rAB = 0; rAB < accAB.rows(); rAB++)
                                      for (int rAB_C = 0; rAB_C < accAB_C.rows(); rAB_C++)
                                        for (int rBC = 0; rBC < accBC.rows(); rBC++)
                                          B(r, (eAB * etaAB_Cmax + eAB_C) * etaBCmax + eBC) 
                                          += accAB(rAB, eAB) *
                                             accAB_C(rAB_C, eAB_C) *
                                             accBC(rBC, eBC) * 
                                             U6_coeffs[((rAB * accAB_C.rows() +
                                                         rAB_C) * accBC.rows() +
                                                         rBC) * accA_BC.rows() +
                                                         rA_BC];
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }

            r++;

            if (r == etaA_BCmax)
            {
              done = true;
              break;
            }
          }
        }
      }
    }

    // std::cout << A << std::endl;
    // std::cout << B << std::endl;

    Eigen::MatrixXd X = A.householderQr().solve(B);

    std::vector<double> U(etaABmax * etaAB_Cmax * etaBCmax * etaA_BCmax);

    for (int eAB = 0; eAB < etaABmax; eAB++)
      for (int eAB_C = 0; eAB_C < etaAB_Cmax; eAB_C++)
        for (int eBC = 0; eBC < etaBCmax; eBC++)
          for (int eA_BC = 0; eA_BC < etaA_BCmax; eA_BC++)
          {
            int col = (eAB * etaAB_Cmax + eAB_C) * etaBCmax + eBC;
            if (std::fabs(X(eA_BC, col)) < 1e-14)
              X(eA_BC, col) = 0.0;

            U[((eAB * etaAB_Cmax + eAB_C) * etaBCmax + eBC) * etaA_BCmax +
              eA_BC] = X(eA_BC, col);
          }

#ifdef TIME_U4U6Z6
    auto end = std::chrono::high_resolution_clock::now();
    u4u6_time += end - start;  // duration in seconds

    // Print occasionally 
    if (u4u6_time.count() - u4u6_last_print >= 1.0) 
    {
        std::cout << "Accumulated U4U6 time: " << u4u6_time.count() << " s\n";
        u4u6_last_print = u4u6_time.count();
    }
#endif

    return U;
  }

#define U4LIB_ENABLE_U4U6_CACHE
  class U4U6_db
  {
    using key_type = std::array<uint8_t, 18>;
    using value_type = std::vector<double>;

#ifdef U4LIB_ENABLE_U4U6_CACHE
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
    U4U6_db() = default;
    U4U6_db(const U4U6_db&) = delete;
    U4U6_db& operator=(const U4U6_db&) = delete;

    static value_type compute(
      const u4::irrep& u4A, const u4::irrep& u4B, 
      const u4::irrep& u4ABC, const u4::irrep& u4C,
      const u4::irrep& u4AB, int etaABmax, int etaAB_Cmax,
      const u4::irrep& u4BC, int etaBCmax, int etaA_BCmax)
    {
      std::vector<u4::gelfand> gArrA = u4::gelfandBasis(u4A),
                               gArrB = u4::gelfandBasis(u4B),
                               gArrC = u4::gelfandBasis(u4C),
                               gArrAB = u4::gelfandBasis(u4AB),
                               gArrBC = u4::gelfandBasis(u4BC),
                               gArrABC = u4::gelfandBasis(u4ABC);

      return U6(u4A,gArrA,u4B,gArrB,u4ABC,gArrABC,u4C,gArrC,
                u4AB,gArrAB,etaABmax,etaAB_Cmax,
                u4BC,gArrBC,etaBCmax,etaA_BCmax);
    }

  public:
    value_type get(
      int n14a, int n24a, int n34a, int n44a, 
      int n14b, int n24b, int n34b, int n44b, 
      int n14abc, int n24abc, int n34abc, int n44abc, 
      int n14c, int n24c, int n34c, int n44c, 
      int n14ab, int n24ab, int n34ab, int n44ab, 
      int n14bc, int n24bc, int n34bc, int n44bc)
    {
      u4::irrep u4A(n14a - n44a, n24a - n44a, n34a - n44a, 0), 
                u4B(n14b - n44b, n24b - n44b, n34b - n44b, 0), 
                u4C(n14c - n44c, n24c - n44c, n34c - n44c, 0), 
                u4AB(n14ab - n44ab, n24ab - n44ab, n34ab - n44ab, 0), 
                u4BC(n14bc - n44bc, n24bc - n44bc, n34bc - n44bc, 0), 
                u4ABC(n14abc - n44abc, n24abc - n44abc, n34abc - n44abc, 0);

      int etaABmax = Eta(u4A,u4B,u4AB), etaAB_Cmax = Eta(u4AB,u4C,u4ABC),
          etaBCmax = Eta(u4B,u4C,u4BC), etaA_BCmax = Eta(u4A,u4BC,u4ABC);

      if (etaABmax == 0 || etaAB_Cmax == 0 || etaBCmax == 0 || etaA_BCmax == 0)
        return {};
      
#ifdef U4LIB_ENABLE_U4U6_CACHE
      std::array<qcx_utils::UInt8, 18> safe_key = {
        u4A.n14(), u4A.n24(), u4A.n34(),
        u4B.n14(), u4B.n24(), u4B.n34(),
        u4ABC.n14(), u4ABC.n24(), u4ABC.n34(),
        u4C.n14(), u4C.n24(), u4C.n34(),
        u4AB.n14(), u4AB.n24(), u4AB.n34(),
        u4BC.n14(), u4BC.n24(), u4BC.n34()};

      key_type key = {safe_key[0], safe_key[1], safe_key[2],
                      safe_key[3], safe_key[4], safe_key[5],
                      safe_key[6], safe_key[7], safe_key[8],
                      safe_key[9], safe_key[10], safe_key[11],
                      safe_key[12], safe_key[13], safe_key[14],
                      safe_key[15], safe_key[16], safe_key[17]};

      auto producer = [&u4A, &u4B, &u4ABC, &u4C, 
                       &u4AB, &etaABmax, &etaAB_Cmax,
                       &u4BC, &etaBCmax, &etaA_BCmax](key_type)
      { 
        return compute(u4A, u4B, u4ABC, u4C, u4AB, etaABmax, etaAB_Cmax, 
                                             u4BC, etaBCmax, etaA_BCmax);
      };
        
      auto accessor = [](const auto& element) { *p_result_ = element.second; };

      cache_.visit(key, producer, accessor);
#else
      *p_result_ = compute(u4A, u4B, u4ABC, u4C, u4AB, etaABmax, etaAB_Cmax, 
                                                 u4BC, etaBCmax, etaA_BCmax);
#endif

      return *p_result_;
    }

    friend U4U6_db& u4u6db();
  };

  inline U4U6_db& u4u6db()
  {
    static U4U6_db u4u6db;

    return u4u6db;
  }

  // input by users: u4b, u4a, u4abc, u4c, u4ab, u4bc, all normalized
  inline std::vector<double>
  Z6(const u4::irrep& u4B, const std::vector<u4::gelfand>& gArrB,
     const u4::irrep& u4A, const std::vector<u4::gelfand>& gArrA,
     const u4::irrep& u4ABC, const std::vector<u4::gelfand>& gArrABC,
     const u4::irrep& u4C, const std::vector<u4::gelfand>& gArrC,
     const u4::irrep& u4AB, const std::vector<u4::gelfand>& gArrAB,
     int etaABmax, int etaAB_Cmax, const u4::irrep& u4AC,
     const std::vector<u4::gelfand>& gArrAC, int etaACmax, int etaAC_Bmax)
  {
#ifdef TIME_U4U6Z6
    static std::chrono::duration<double> u4z6_time(0);
    static double u4z6_last_print = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
#endif

    auto& wigU4U3db = u4::wigU4U3db();

    Eigen::MatrixXd A(etaAC_Bmax, etaAC_Bmax);
    Eigen::MatrixXd B(etaAC_Bmax, etaABmax * etaAB_Cmax * etaACmax);
    B.setZero();

    int r = 0;
    bool done = false;
    for (int n13B = u4B.n14(); n13B >= u4B.n24() && !done; n13B--)
    {
      for (int n23B = u4B.n24(); n23B >= u4B.n34() && !done; n23B--)
      {
        for (int n33B = u4B.n34(); n33B >= u4B.n44() && !done; n33B--)
        {
          auto [existAC_B, accAC_B] = wigU4U3db.get(
              u4AC.n14(), u4AC.n24(), u4AC.n34(), u4AC.n44(), u4AC.n14(),
              u4AC.n24(), u4AC.n34(), u4B.n14(), u4B.n24(), u4B.n34(),
              u4B.n44(), n13B, n23B, n33B, u4ABC.n14(), u4ABC.n24(),
              u4ABC.n34(), u4ABC.n44(), u4ABC.n14(), u4ABC.n24(), u4ABC.n34());
          if (!existAC_B) continue;
          
          // std::cout << "accAC_B = " << accAC_B << std::endl;
          u3::irrep u3b(n13B - n33B, n23B - n33B, 0),
            u3ac(u4AC.n14() - u4AC.n34(), u4AC.n24() - u4AC.n34(), 0),
            u3abc(u4ABC.n14() - u4ABC.n34(), u4ABC.n24() - u4ABC.n34(), 0);
          // u3b.normalize();
          // u3ac.normalize();
          // u3abc.normalize();

          u3::gelfands gArr3b = u3::gelfandBasis(u3b),
                       gArr3ac = u3::gelfandBasis(u3ac),
                       gArr3abc = u3::gelfandBasis(u3abc);

          for (int rAC_B = 0; rAC_B < accAC_B.rows(); rAC_B++)
          {
            A.row(r) = accAC_B.row(rAC_B);

            for (int n13A = u4A.n14(); n13A >= u4A.n24(); n13A--)
            {
              for (int n23A = u4A.n24(); n23A >= u4A.n34(); n23A--)
              {
                for (int n33A = u4A.n34(); n33A >= u4A.n44(); n33A--)
                {
                  for (int n13AB = u4AB.n14(); n13AB >= u4AB.n24(); n13AB--)
                  {
                    for (int n23AB = u4AB.n24(); n23AB >= u4AB.n34(); n23AB--)
                    {
                      for (int n33AB = u4AB.n34(); n33AB >= u4AB.n44(); n33AB--)
                      {
                        auto [existAB, accAB] = wigU4U3db.get(
                          u4A.n14(), u4A.n24(), u4A.n34(), u4A.n44(), 
                          n13A, n23A, n33A, 
                          u4B.n14(), u4B.n24(), u4B.n34(), u4B.n44(), 
                          n13B, n23B, n33B, 
                          u4AB.n14(), u4AB.n24(), u4AB.n34(), u4AB.n44(), 
                          n13AB, n23AB, n33AB);
                        if (!existAB) continue;
                          
                        // std::cout << "accAB = " << accAB << std::endl;
                        u3::irrep u3a(n13A - n33A, n23A - n33A, 0),
                            u3ab(n13AB - n33AB, n23AB - n33AB, 0);
                        // u3a.normalize();
                        // u3ab.normalize();

                        u3::gelfands gArr3a = u3::gelfandBasis(u3a),
                                     gArr3ab = u3::gelfandBasis(u3ab);

                        for (int n13C = u4C.n14(); n13C >= u4C.n24(); n13C--)
                        {
                          for (int n23C = u4C.n24(); n23C >= u4C.n34(); n23C--)
                          {
                            for (int n33C = u4C.n34(); n33C >= u4C.n44(); n33C--)
                            {
                              auto [existAB_C, accAB_C] = wigU4U3db.get(
                                u4AB.n14(), u4AB.n24(), u4AB.n34(), u4AB.n44(),
                                n13AB, n23AB, n33AB,
                                u4C.n14(), u4C.n24(), u4C.n34(), u4C.n44(),
                                n13C, n23C, n33C, 
                                u4ABC.n14(), u4ABC.n24(), u4ABC.n34(), u4ABC.n44(),
                                u4ABC.n14(), u4ABC.n24(), u4ABC.n34());
                              if (!existAB_C) continue;

                              auto [existAC, accAC] = wigU4U3db.get(
                                u4A.n14(), u4A.n24(), u4A.n34(), u4A.n44(),
                                n13A, n23A, n33A, 
                                u4C.n14(), u4C.n24(), u4C.n34(), u4C.n44(),
                                n13C, n23C, n33C, 
                                u4AC.n14(), u4AC.n24(), u4AC.n34(), u4AC.n44(),
                                u4AC.n14(), u4AC.n24(), u4AC.n34());
                              if (!existAC) continue;
                                  
                              u3::irrep u3c(n13C - n33C, n23C - n33C, 0);
                              // u3c.normalize();
                              u3::gelfands gArr3c = u3::gelfandBasis(u3c);

                              std::vector<double> Z6_coeffs = u3::Z6(
                                u3b, gArr3b, u3a, gArr3a, u3abc,
                                gArr3abc, u3c, gArr3c, u3ab, gArr3ab,
                                accAB.rows(), accAB_C.rows(), u3ac,
                                gArr3ac, accAC.rows(), accAC_B.rows());

                              for (int eAB = 0; eAB < etaABmax; eAB++)
                                for (int eAB_C = 0; eAB_C < etaAB_Cmax; eAB_C++)
                                  for (int eAC = 0; eAC < etaACmax; eAC++)
                                    for (int rAB = 0; rAB < accAB.rows(); rAB++)
                                      for (int rAB_C = 0; rAB_C < accAB_C.rows(); rAB_C++)
                                        for (int rAC = 0; rAC < accAC.rows(); rAC++)
                                          B(r, (eAB * etaAB_Cmax + eAB_C) * etaACmax + eAC)
                                          += accAB(rAB, eAB) *
                                             accAB_C(rAB_C, eAB_C) *
                                             accAC(rAC, eAC) *
                                             Z6_coeffs [((rAB * accAB_C.rows() +
                                                          rAB_C) * accAC.rows() +
                                                          rAC) * accAC_B.rows() +
                                                          rAC_B];                            
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }

            r++;

            if (r == etaAC_Bmax)
            {
              done = true;
              break;
            }
          }    
        }
      }
    }

    // std::cout << A << std::endl;
    // std::cout << B << std::endl;

    Eigen::MatrixXd X = A.householderQr().solve(B);

    std::vector<double> Z(etaABmax * etaAB_Cmax * etaACmax * etaAC_Bmax);

    for (int eAB = 0; eAB < etaABmax; eAB++)
      for (int eAB_C = 0; eAB_C < etaAB_Cmax; eAB_C++)
        for (int eAC = 0; eAC < etaACmax; eAC++)
          for (int eAC_B = 0; eAC_B < etaAC_Bmax; eAC_B++)
          {
            int col = (eAB * etaAB_Cmax + eAB_C) * etaACmax + eAC;
            if (std::fabs(X(eAC_B, col)) < 1e-14)
              X(eAC_B, col) = 0.0;

            Z[((eAB * etaAB_Cmax + eAB_C) * etaACmax + eAC) * etaAC_Bmax +
              eAC_B] = X(eAC_B, col);
          }

#ifdef TIME_U4U6Z6
    auto end = std::chrono::high_resolution_clock::now();
    u4z6_time += end - start;  // duration in seconds

    // Print occasionally 
    if (u4z6_time.count() - u4z6_last_print >= 1.0) 
    {
        std::cout << "Accumulated U4Z6 time: " << u4z6_time.count() << " s\n";
        u4z6_last_print = u4z6_time.count();
    }
#endif

    return Z;
  }

#define U4LIB_ENABLE_U4Z6_CACHE
  class U4Z6_db
  {
    using key_type = std::array<uint8_t, 18>;
    using value_type = std::vector<double>;

#ifdef U4LIB_ENABLE_U4Z6_CACHE
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
    U4Z6_db() = default;
    U4Z6_db(const U4Z6_db&) = delete;
    U4Z6_db& operator=(const U4Z6_db&) = delete;

    static value_type compute(
      const u4::irrep& u4B, const u4::irrep& u4A, 
      const u4::irrep& u4ABC, const u4::irrep& u4C,
      const u4::irrep& u4AB, int etaABmax, int etaAB_Cmax,
      const u4::irrep& u4AC, int etaACmax, int etaAC_Bmax)
    {
      std::vector<u4::gelfand> gArrA = u4::gelfandBasis(u4A),
                               gArrB = u4::gelfandBasis(u4B),
                               gArrC = u4::gelfandBasis(u4C),
                               gArrAB = u4::gelfandBasis(u4AB),
                               gArrAC = u4::gelfandBasis(u4AC),
                               gArrABC = u4::gelfandBasis(u4ABC);

      return Z6(u4B,gArrB,u4A,gArrA,u4ABC,gArrABC,u4C,gArrC,
                u4AB,gArrAB,etaABmax,etaAB_Cmax,
                u4AC,gArrAC,etaACmax,etaAC_Bmax);
    }

  public:
    value_type get(
      int n14b, int n24b, int n34b, int n44b,
      int n14a, int n24a, int n34a, int n44a, 
      int n14abc, int n24abc, int n34abc, int n44abc, 
      int n14c, int n24c, int n34c, int n44c, 
      int n14ab, int n24ab, int n34ab, int n44ab, 
      int n14ac, int n24ac, int n34ac, int n44ac)
    {
      u4::irrep u4A(n14a - n44a, n24a - n44a, n34a - n44a, 0), 
                u4B(n14b - n44b, n24b - n44b, n34b - n44b, 0), 
                u4C(n14c - n44c, n24c - n44c, n34c - n44c, 0), 
                u4AB(n14ab - n44ab, n24ab - n44ab, n34ab - n44ab, 0), 
                u4AC(n14ac - n44ac, n24ac - n44ac, n34ac - n44ac, 0), 
                u4ABC(n14abc - n44abc, n24abc - n44abc, n34abc - n44abc, 0);

      int etaABmax = Eta(u4A,u4B,u4AB), etaAB_Cmax = Eta(u4AB,u4C,u4ABC),
          etaACmax = Eta(u4A,u4C,u4AC), etaAC_Bmax = Eta(u4AC,u4B,u4ABC);

      if (etaABmax == 0 || etaAB_Cmax == 0 || etaACmax == 0 || etaAC_Bmax == 0)
        return {};

#ifdef U4LIB_ENABLE_U4Z6_CACHE
      std::array<qcx_utils::UInt8, 18> safe_key = {
        u4B.n14(), u4B.n24(), u4B.n34(),
        u4A.n14(), u4A.n24(), u4A.n34(),
        u4ABC.n14(), u4ABC.n24(), u4ABC.n34(),
        u4C.n14(), u4C.n24(), u4C.n34(),
        u4AB.n14(), u4AB.n24(), u4AB.n34(),
        u4AC.n14(), u4AC.n24(), u4AC.n34()};

      key_type key = {safe_key[0], safe_key[1], safe_key[2],
                      safe_key[3], safe_key[4], safe_key[5],
                      safe_key[6], safe_key[7], safe_key[8],
                      safe_key[9], safe_key[10], safe_key[11],
                      safe_key[12], safe_key[13], safe_key[14],
                      safe_key[15], safe_key[16], safe_key[17]};

      auto producer = [&u4B, &u4A, &u4ABC, &u4C, 
                       &u4AB, &etaABmax, &etaAB_Cmax,
                       &u4AC, &etaACmax, &etaAC_Bmax](key_type)
      { 
        return compute(u4B, u4A, u4ABC, u4C, u4AB, etaABmax, etaAB_Cmax,
                                             u4AC, etaACmax, etaAC_Bmax);
      };

      auto accessor = [](const auto& element) { *p_result_ = element.second; };

      cache_.visit(key, producer, accessor);
#else
      *p_result_ = compute(u4B, u4A, u4ABC, u4C, u4AB, etaABmax, etaAB_Cmax,
                                                 u4AC, etaACmax, etaAC_Bmax);
#endif

      return *p_result_;
    }

    friend U4Z6_db& u4z6db();
  };

  inline U4Z6_db& u4z6db()
  {
    static U4Z6_db u4z6db;

    return u4z6db;
  }

} // namespace u4
#endif

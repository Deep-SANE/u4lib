#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>

#include <u3lib.h>

#include <u4lib.h>
#include <U4decomposition.h>
#include <U4_U9Coeffs.h>
#include <U4basis.h>

#include <utils.h> 

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  std::cout
      << "This program compute U9-recoupling coefficients of four U(4) irreps"
      << std::endl;

  int n14a, n24a, n34a, n44a;
  std::cout << "Please enter the I. irrep [n14a,n24a,n34a,n44a]: ";
  std::cin >> n14a >> n24a >> n34a >> n44a;
  u4::irrep u4a(n14a, n24a, n34a, n44a);
  if (n44a != 0)
  {
    u4a.normalize();
    std::cout << "Your irrep is normalized to " << u4a << std::endl;
  }

  int n14b, n24b, n34b, n44b;
  std::cout << "Please enter the II. irrep [n14b,n24b,n34b,n44b]: ";
  std::cin >> n14b >> n24b >> n34b >> n44b;
  u4::irrep u4b(n14b, n24b, n34b, n44b);
  if (n44b != 0)
  {
    u4b.normalize();
    std::cout << "Your irrep is normalized to " << u4b << std::endl;
  }

  int n14c, n24c, n34c, n44c;
  std::cout << "Please enter the III. irrep [n14c,n24c,n34c,n44c]: ";
  std::cin >> n14c >> n24c >> n34c >> n44c;
  u4::irrep u4c(n14c, n24c, n34c, n44c);
  if (n44c != 0)
  {
    u4c.normalize();
    std::cout << "Your irrep is normalized to " << u4c << std::endl;
  }

  int n14d, n24d, n34d, n44d;
  std::cout << "Please enter the IV. irrep [n14d,n24d,n34d,n44d]: ";
  std::cin >> n14d >> n24d >> n34d >> n44d;
  u4::irrep u4d(n14d, n24d, n34d, n44d);
  if (n44d != 0)
  {
    u4d.normalize();
    std::cout << "Your irrep is normalized to " << u4d << std::endl;
  }

  // couple the I. and II. irreps
  auto [u4ab, etamaxsab] = u4::decomposition(u4a, u4b);

  std::cout << "The normalized coupled irreps of the I and II are: "
            << std::endl;
  for (int i = 0; i < u4ab.size(); i++)
  {
    std::cout << "(" << i << "): " << u4ab[i] << "\t eta = 1 ... " 
              << etamaxsab[i] << std::endl;
  }

  // couple the I. and III. irreps
  auto [u4ac, etamaxsac] = u4::decomposition(u4a, u4c);

  std::cout << "The normalized coupled irreps of the I and III are: "
            << std::endl;
  for (int i = 0; i < u4ac.size(); i++)
  {
    std::cout << "(" << i << "): " << u4ac[i] << "\t eta = 1 ... " 
              << etamaxsac[i] << std::endl;
  }

  // couple the II. and IV. irreps
  auto [u4bd, etamaxsbd] = u4::decomposition(u4b, u4d);

  std::cout << "The normalized coupled irreps of the II and IV are: "
            << std::endl;
  for (int i = 0; i < u4bd.size(); i++)
  {
    std::cout << "(" << i << "): " << u4bd[i] << "\t eta = 1 ... " 
              << etamaxsbd[i] << std::endl;
  }

  // couple the III. and IV. irreps
  auto [u4cd, etamaxscd] = u4::decomposition(u4c, u4d);

  std::cout << "The normalized coupled irreps of the III and IV are: "
            << std::endl;
  for (int i = 0; i < u4cd.size(); i++)
  {
    std::cout << "(" << i << "): " << u4cd[i] << "\t eta = 1 ... " 
              << etamaxscd[i] << std::endl;
  }

  int i_ab, i_ac, i_bd, i_cd, i_ab_cd, i_ac_bd;
  std::cout << "Please enter the index of the I-II coupled irrep you are "
               "interested: ";
  std::cin >> i_ab;
  std::cout << "Please enter the index of the I-III coupled irrep you are "
               "interested: ";
  std::cin >> i_ac;
  std::cout << "Please enter the index of the II-IV coupled irrep you are "
               "interested: ";
  std::cin >> i_bd;
  std::cout << "Please enter the index of the III-IV coupled irrep you are "
               "interested: ";
  std::cin >> i_cd;

  // couple the 4 irreps in the scheme (I-II)-(III-IV)
  auto [u4ab_cd, etamaxsab_cd] = u4::decomposition(u4ab[i_ab], u4cd[i_cd]);

  

  std::cout << "The normalized coupled irreps of the I, II, III and IV are: "
            << std::endl;
  for (int i = 0; i < u4ab_cd.size(); i++)
  {
    std::cout << "(" << i << "): " << u4ab_cd[i] << "\t eta = 1 ... "
              << etamaxsab_cd[i] << std::endl;
  }
  std::cout << "Please enter the index of the I-II-III-IV coupled irrep you "
               "are interested: ";
  std::cin >> i_ab_cd;

  // couple the 4 irreps in scheme (I-III)-(II-IV)
  auto etamaxac_bd = u4::Eta(u4ac[i_ac], u4bd[i_bd], u4ab_cd[i_ab_cd]);

  if (etamaxac_bd > 0)
  {
    // Start time measurement
    auto t0 = std::chrono::high_resolution_clock::now();

    u4::gelfands gArrA = u4::gelfandBasis(u4a);
    u4::gelfands gArrB = u4::gelfandBasis(u4b);
    u4::gelfands gArrC = u4::gelfandBasis(u4c);
    u4::gelfands gArrD = u4::gelfandBasis(u4d);
    u4::gelfands gArrAB = u4::gelfandBasis(u4ab[i_ab]);
    u4::gelfands gArrCD = u4::gelfandBasis(u4cd[i_cd]);
    u4::gelfands gArrAC = u4::gelfandBasis(u4ac[i_ac]);
    u4::gelfands gArrBD = u4::gelfandBasis(u4bd[i_bd]);
    u4::gelfands gArrABCD = u4::gelfandBasis(u4ab_cd[i_ab_cd]);

    auto coeffs =
      u4::U9(u4a, gArrA, u4b, gArrB, u4c, gArrC, u4d, gArrD, u4ab[i_ab],
             gArrAB, u4cd[i_cd], gArrCD, u4ac[i_ac], gArrAC, u4bd[i_bd],
             gArrBD, u4ab_cd[i_ab_cd], gArrABCD, etamaxsab[i_ab],
             etamaxscd[i_cd], etamaxsab_cd[i_ab_cd], etamaxsac[i_ac],
             etamaxsbd[i_bd], etamaxac_bd);

    // End time measurement
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << u4a << "\t" << u4b << "\t" << u4ab[i_ab] << "\tetaab"
              << std::endl;
    std::cout << u4c << "\t" << u4d << "\t" << u4cd[i_cd] << "\tetacd"
              << std::endl;
    std::cout << u4ac[i_ac] << "\t" << u4bd[i_bd] << "\t" << u4ab_cd[i_ab_cd]
              << "\tetaac_bd" << std::endl;
    std::cout << "etaac\tetabd\tetaab_cd\t" << std::endl;

    std::cout << coeffs << std::endl;

    // Calculate the duration
    std::chrono::duration<double> duration = t1 - t0;

    // Output the time in seconds
    std::cout << "Elapsed time: " << duration.count() << " seconds or "
              << duration.count() / 60.0 << " minutes" << std::endl;

    // -----------------------------------------------------------------------
    // Multi-threading demonstration:
    // Each thread calls u4::initialize_thread() before database access,
    // evaluates coefficients inside, prints them out along with thread index,
    // and calls u4::finalize_thread().
    // -----------------------------------------------------------------------
    std::cout << "\n[Multi-threading demonstration]" << std::endl;
#pragma omp parallel num_threads(4)
    {
      u4::initialize_thread();
      int tid = omp_get_thread_num();

      const auto blk = u4::u4u9db().get_U4U9(
          u4a, u4b, u4c, u4d, u4ab[i_ab], u4cd[i_cd], u4ac[i_ac], u4bd[i_bd],
          u4ab_cd[i_ab_cd], etamaxsac[i_ac], etamaxsbd[i_bd],
          etamaxsab_cd[i_ab_cd], etamaxsab[i_ab], etamaxscd[i_cd], etamaxac_bd);

#pragma omp critical
      {
        std::cout << "[Thread " << tid << "] Evaluated U9 block with size " << blk.size()
                  << ", (0,0,0,0,0,0) = " << (blk.size() > 0 ? blk(0,0,0,0,0,0) : 0.0)
                  << std::endl;
      }

      u4::finalize_thread();
    }
  }
  else
  {
    std::cout << "Recoupling does not exist" << std::endl;
  }

  u4::finalize();
}

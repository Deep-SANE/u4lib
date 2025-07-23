#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <U4decomposition.h>
#include <U4_U6Z6Coeffs.h>
#include <U4basis.h>

#ifdef U4LIB_USE_SLEPC
#include <slepcsys.h> 
#endif 

int main(int argc, char* argv[])
{  
#ifdef U4LIB_USE_SLEPC 
  SlepcInitialize(&argc, &argv, NULL, NULL);     
#endif

  std::cout
      << "This program compute Z6-recoupling coefficients of three U(4) irreps"
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

  int i_ab, i_ac, i_ab_c;
  std::cout << "Please enter the index of the I-II coupled irrep you are "
               "interested: ";
  std::cin >> i_ab;
  std::cout << "Please enter the index of the I-III coupled irrep you are "
               "interested: ";
  std::cin >> i_ac;

  // couple the three irreps in scheme (I-II)-III
  auto [u4ab_c, etamaxsab_c] = u4::decomposition(u4ab[i_ab], u4c);

  std::cout << "The normalized coupled irreps of the I, II and III are: "
            << std::endl;
  for (int i = 0; i < u4ab_c.size(); i++)
  {
    std::cout << "(" << i << "): " << u4ab_c[i] << "\t eta = 1 ... "
              << etamaxsab_c[i] << std::endl;
  }
  std::cout << "Please enter the indices of the I-II-III coupled irrep you are "
               "interested: ";
  std::cin >> i_ab_c;

  // couple the three irreps in scheme (I-III)-II
  auto etamaxac_b = u4::Eta(u4ac[i_ac], u4b, u4ab_c[i_ab_c]);

  if (etamaxac_b > 0)
  {
    // Start time measurement
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<u4::gelfand> gArrA = u4::gelfandBasis(u4a);
    std::vector<u4::gelfand> gArrB = u4::gelfandBasis(u4b);
    std::vector<u4::gelfand> gArrC = u4::gelfandBasis(u4c);
    std::vector<u4::gelfand> gArrAB = u4::gelfandBasis(u4ab[i_ab]);
    std::vector<u4::gelfand> gArrAC = u4::gelfandBasis(u4ac[i_ac]);
    std::vector<u4::gelfand> gArrABC = u4::gelfandBasis(u4ab_c[i_ab_c]);

    qcx_utils::SO3_CGC cgcs;
    qcx_utils::SO3_Racah racahs;
    u3::WigU3U2_db wigU3U2db(cgcs);
    u4::WigU4U3_db wigU4U3db(cgcs,wigU3U2db);

    std::vector<double> coeffs = u4::Z6(
      u4b, gArrB, u4a, gArrA, u4ab_c[i_ab_c], gArrABC, u4c, gArrC, u4ab[i_ab], 
      gArrAB, etamaxsab[i_ab], etamaxsab_c[i_ab_c], u4ac[i_ac], gArrAC, 
      etamaxsac[i_ac], etamaxac_b, cgcs, racahs, wigU3U2db, wigU4U3db);

    // End time measurement
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Z([ni4b],   [ni4a],   [ni4],   [ni4c] ; [ni4ab],eab,eab_c, "
                 "[ni4ac],eac,eac_b)" << std::endl;
    std::cout << "  " << u4b << "," << u4a << "," << u4ab_c[i_ab_c] << ","
              << u4c << ";" << u4ab[i_ab] << ",\t    " << u4ac[i_ac]
              << std::endl;

    for (int eab = 0; eab < etamaxsab[i_ab]; eab++)
    {
      for (int eab_c = 0; eab_c < etamaxsab_c[i_ab_c]; eab_c++)
      {
        for (int eac = 0; eac < etamaxsac[i_ac]; eac++)
        {
          for (int eac_b = 0; eac_b < etamaxac_b; eac_b++)
          {
            std::cout << "\t\t\t\t\t\t   " << eab + 1 << "   " << eab_c + 1
                      << "\t      " << eac + 1 << "  " << eac_b + 1 << "\t -> "
                      << coeffs[((eab * etamaxsab_c[i_ab_c] + eab_c) *
                                     etamaxsac[i_ac] +
                                 eac) *
                                    etamaxac_b + eac_b]
                      << std::endl;
          }
        }
      }
    }

    // Calculate the duration
    std::chrono::duration<double> duration = t1 - t0;

    // Output the time in seconds
    std::cout << "Elapsed time: " << duration.count() << " seconds or "
              << duration.count() / 60.0 << " minutes" << std::endl;
  }
  else
  {
    std::cout << "Recoupling does not exist" << std::endl;
  }

#ifdef U4LIB_USE_SLEPC 
  SlepcFinalize();
#endif

  return 0;
}

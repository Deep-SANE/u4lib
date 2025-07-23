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
      << "This program compute U6-recoupling coefficients of three U(4) irreps"
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

  // couple the II. and III. irreps  
  auto [u4bc, etamaxsbc] = u4::decomposition(u4b, u4c);

  std::cout << "The normalized coupled irreps of the II and III are: "
            << std::endl;
  for (int i = 0; i < u4bc.size(); i++)
  {
    std::cout << "(" << i << "): " << u4bc[i] << "\t eta = 1 ... " 
              << etamaxsbc[i] << std::endl;
  }

  int i_ab, i_bc, i_ab_c;
  std::cout << "Please enter the index of the I-II coupled irrep you are "
               "interested: ";
  std::cin >> i_ab;
  std::cout << "Please enter the index of the II-III coupled irrep you are "
               "interested: ";
  std::cin >> i_bc;

  // couple the three irreps in scheme (I-II)-III
  auto [u4ab_c, etamaxsab_c] = u4::decomposition(u4ab[i_ab], u4c);

  std::cout << "The normalized coupled irreps of the I, II and III are: "
            << std::endl;
  for (int i = 0; i < u4ab_c.size(); i++)
  {
    std::cout << "(" << i << "): " << u4ab_c[i] << "\t eta = 1 ... "
              << etamaxsab_c[i] << std::endl;
  }
  std::cout << "Please enter the index of the I-II-III coupled irrep you are "
               "interested: ";
  std::cin >> i_ab_c;

  // couple the three irreps in scheme I-(II-III)
  auto etamaxa_bc = u4::Eta(u4a, u4bc[i_bc], u4ab_c[i_ab_c]);

  if (etamaxa_bc > 0)
  {
    // Start time measurement
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<u4::gelfand> gArrA = u4::gelfandBasis(u4a);
    std::vector<u4::gelfand> gArrB = u4::gelfandBasis(u4b);
    std::vector<u4::gelfand> gArrC = u4::gelfandBasis(u4c);
    std::vector<u4::gelfand> gArrAB = u4::gelfandBasis(u4ab[i_ab]);
    std::vector<u4::gelfand> gArrBC = u4::gelfandBasis(u4bc[i_bc]);
    std::vector<u4::gelfand> gArrABC = u4::gelfandBasis(u4ab_c[i_ab_c]);

    qcx_utils::SO3_CGC cgcs;
    qcx_utils::SO3_Racah racahs;
    u3::WigU3U2_db wigU3U2db(cgcs);
    u4::WigU4U3_db wigU4U3db(cgcs,wigU3U2db);

    std::vector<double> coeffs = u4::U6(
        u4a, gArrA, u4b, gArrB, u4ab_c[i_ab_c], gArrABC, u4c, gArrC, u4ab[i_ab], 
        gArrAB, etamaxsab[i_ab], etamaxsab_c[i_ab_c], u4bc[i_bc], gArrBC,
        etamaxsbc[i_bc], etamaxa_bc, cgcs, racahs, wigU3U2db, wigU4U3db);

    // End time measurement
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "U([ni4a],   [ni4b],   [ni4],    [ni4c] ;  [ni4ab],eab,eab_c, "
                 "[ni4bc],ebc,ea_bc)" << std::endl;
    std::cout << "  " << u4a << "," << u4b << "," << u4ab_c[i_ab_c] << ","
              << u4c << ";" << u4ab[i_ab] << ",\t    " << u4bc[i_bc]
              << std::endl;

    for (int eab = 0; eab < etamaxsab[i_ab]; eab++)
    {
      for (int eab_c = 0; eab_c < etamaxsab_c[i_ab_c]; eab_c++)
      {
        for (int ebc = 0; ebc < etamaxsbc[i_bc]; ebc++)
        {
          for (int ea_bc = 0; ea_bc < etamaxa_bc; ea_bc++)
          {
            std::cout << "\t\t\t\t\t\t   " << eab + 1 << "   " << eab_c + 1
                      << "\t      " << ebc + 1 << "  " << ea_bc + 1 << "\t -> "
                      << coeffs[((eab * etamaxsab_c[i_ab_c] + eab_c) *
                                     etamaxsbc[i_bc] +
                                 ebc) *
                                    etamaxa_bc + ea_bc]
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

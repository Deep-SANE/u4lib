#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <U4basis.h>
#include <U4decomposition.h>
#include <U4U3xU4U3.h>

int main()
{

  int n14a, n24a, n34a, n44a;
  std::cout << "Please enter the first irrep [n14a,n24a,n34a,n44a]: ";
  std::cin >> n14a >> n24a >> n34a >> n44a;
  u4::irrep u4a(n14a, n24a, n34a, n44a);
  if (n44a != 0)
  {
    u4a.normalize();
    std::cout << "Your irrep is normalized to " << u4a << std::endl;
  }

  int n14b, n24b, n34b, n44b;
  std::cout << "Please enter the second irrep [n14b,n24b,n34b,n44b]: ";
  std::cin >> n14b >> n24b >> n34b >> n44b;
  u4::irrep u4b(n14b, n24b, n34b, n44b);
  if (n44b != 0)
  {
    u4b.normalize();
    std::cout << "Your irrep is normalized to " << u4b << std::endl;
  }

  auto [u4cs, etamaxs] = u4::decomposition(u4a, u4b);

  std::cout << "The normalized coupled irreps are: " << std::endl;
  for (int i = 0; i < u4cs.size(); i++)
  {
    std::cout << "(" << i << "): [" << u4cs[i].n14() - u4cs[i].n44() << ","
              << u4cs[i].n24() - u4cs[i].n44() << "," << u4cs[i].n34() - u4cs[i].n44()
              << ",0] \t eta = 1 ... " << etamaxs[i] << std::endl;
  }

  int index;
  std::cout
      << "Please enter the index of the coupled irrep you are interested: ";
  std::cin >> index;

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  std::vector<u4::gelfand> gArrA = u4::gelfandBasis(u4a),
                           gArrB = u4::gelfandBasis(u4b),
                           gArrC = u4::gelfandBasis(u4cs[index]);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coeffs;
  u4::HighestWeightReducedCoeffs(gArrA, gArrB, gArrC, coeffs); 

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

/*
  std::cout << "< Ga; Gb | Gc > \t\t eta = 1 ... " << etamaxs[index]
            << std::endl;
  for (int j = 0; j < gArr1.size(); j++)
  {
    gArr3[j].normalize();
    std::cout << "< " << gArr1[j] << " ; " << gArr2[j] << " | " << gArr3[j]
              << " >" << "\t" << coeffs.row(j) << std::endl;
  }
*/
  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;
  return 0;
}
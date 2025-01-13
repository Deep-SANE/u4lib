#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <decomposition.h>
#include <U4basis.h>
#include <U4SU2SU2xU4SU2SU2.h>

int main()
{
  /*
  int n14a, n24a, n34a, n44a, n14b, n24b, n34b, n44b;

  std::cout << "Please enter a U(4) irrep [n14a,n24a,n34a,n44a]: ";
  std::cin >> n14a >> n24a >> n34a >> n44a;
  u4::irrep u4a(n14a,n24a,n34a,n44a);
  if (n44a!=0){
          u4a.normalize();
          std::cout << "Your irrep is normalized to " << u4a << std::endl;
  }

  std::cout << "Please enter another U(4) irrep [n14b,n24b,n34b,n44b]: ";
  std::cin >> n14b >> n24b >> n34b >> n44b;
  u4::irrep u4b(n14b,n24b,n34b,n44b);
  if (n44b!=0){
          u4b.normalize();
          std::cout << "Your irrep is normalized to " << u4b << std::endl;
  }
  */

  int P1a, P2a, P3a, P1b, P2b, P3b;
  std::cout << "Please enter SU(4) quantum numbers (P1a,P2a,P3a): ";
  std::cin >> P1a >> P2a >> P3a;
  u4::irrep u4a(P1a, P2a, P3a);

  std::cout << "Please enter SU(4) quantum numbers (P1b,P2b,P3b): ";
  std::cin >> P1b >> P2b >> P3b;
  u4::irrep u4b(P1b, P2b, P3b);

  std::vector<int> etamaxs;
  std::vector<u4::irrep> u4cs = u4::decomposition(u4a, u4b, etamaxs);

  std::cout << "The coupled irreps (P1,P2,P3) are" << std::endl;
  for (int i = 0; i < u4cs.size(); i++)
  {
    std::cout << "(" << i << "): (" << u4cs[i].P1() << "," << u4cs[i].P2()
              << "," << u4cs[i].P3() << ") \t eta = 1 ... " << etamaxs[i]
              << std::endl;
  }

  int index;
  std::cout << "Please enter the index of the irrep you are interested in: ";
  std::cin >> index;

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  std::vector<u4::physicalvector> physvecA;
  std::vector<u4::physicalvector> physvecB;
  std::vector<u4::physicalvector> physvecC;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coeffs;

  coeffs = u4::WigU4SU2SU2(u4a, u4b, u4cs[index], physvecA, physvecB, physvecC);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  for (int j = 0; j < physvecA.size(); j++)
  {
    physvecC[j].toprow.normalize();
    std::cout << "< " << physvecA[j] << " ; " << physvecB[j] << " || "
              << physvecC[j] << " >" << "\t" << coeffs.row(j) << std::endl;
  }

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  return 0;
}
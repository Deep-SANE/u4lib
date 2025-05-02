#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include <U4basis.h>
#include <U4SU2SU2_to_U4U3U2U1.h> 

#ifdef U4LIB_USE_SLEPC
#include <slepcsys.h>
#endif
  
int main(int argc, char* argv[])
{
#ifdef U4LIB_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  int P1, P2, P3;
  std::cout << "Please enter SU(4) quantum numbers (P1,P2,P3): ";
  std::cin >> P1 >> P2 >> P3;

  u4::irrep u4top(P1, P2, P3); 

  // building the gelfand basis 
  std::vector<u4::gelfand> gArr = u4::gelfandBasis(u4top);

  // building the physical basis excluding those which have zero multiplicity
  std::vector<u4::physicalvector> pArr = u4::physicalBasis(u4top);

  std::cout << "The physical basis states are: " << std::endl;
  for (int i = 0; i < pArr.size(); i++)
  {
    std::cout << "(" << i << "): |(" << pArr[i].toprow.P1() << ","
              << pArr[i].toprow.P2() << "," << pArr[i].toprow.P3()
              << "), zeta_max=" << pArr[i].zeta << ", SS=" 
              << pArr[i].SS << ", TT=" << pArr[i].TT << ">" << std::endl;
  }

  int index;
  std::cout << "Please enter the index of the state that you are interested: ";
  std::cin >> index;

  int MSx2, MTx2;
  std::cout << "Please enter projections 2MS <= 2S and 2MT <= 2T: ";
  std::cin >> MSx2 >> MTx2;

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  // store Gelfand states associated with the interested physical state
  std::vector<int> canoBasis;

  // get the transformation
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> transformation =
      u4::phys_to_cano(pArr[index], MSx2, MTx2, canoBasis, gArr);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now(); 

  std::cout
      << "The Gelfand states and the expansion coefficients for zeta = 1,...,"
      << pArr[index].zeta << " are:" << std::endl;
 
  for (int i = 0; i < canoBasis.size(); i++)
    std::cout << "|" << gArr[canoBasis[i]] << "> \t" << transformation.row(i) 
              << std::endl; 

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

#ifdef U4LIB_USE_SLEPC
  SlepcFinalize();
#endif  

  return 0;
}

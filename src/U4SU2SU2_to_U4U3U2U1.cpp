#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <U4basis.h>
#include <U4SU2SU2_to_U4U3U2U1.h>

int main()
{

  /*
  int n14,n24,n34,n44;
  std::cout << "Please enter U(4) quantum numbers [n14,n24,n34,n44]: ";
  std::cin >> n14 >> n24 >> n34 >> n44;
  u4::irrep u4top(n14,n24,n34,n44);
  if (n44!=0){
      u4top.normalize();
      std::cout << "Your irrep is normalized to " << u4top << std::endl;
  }
  */

  int P1, P2, P3;
  std::cout << "Please enter SU(4) quantum numbers (P1,P2,P3): ";
  std::cin >> P1 >> P2 >> P3;
  u4::irrep u4top(P1, P2, P3);

  // building the physical basis excluding those which have zero multiplicity
  std::vector<u4::physicalvector> physbasis = u4::physicalbasis(u4top);

  std::cout << "The physical basis states are: " << std::endl;
  for (int i = 0; i < physbasis.size(); i++)
  {
    std::cout << "(" << i << "): |(" << physbasis[i].toprow.P1() << ","
              << physbasis[i].toprow.P2() << "," << physbasis[i].toprow.P3()
              << "), zeta_max=" << physbasis[i].zeta
              << ", SS=" << physbasis[i].SS << ", TT=" << physbasis[i].TT << ">"
              << std::endl;
  }

  int index;
  std::cout << "Please enter the index of the state that you are interested: ";
  std::cin >> index;

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  // Construct the canonical basis associated with the physical basis state that
  // we are looking at
  std::vector<u4::gelfand> canobasis = u4::canonicalbasis(physbasis[index]);

  // Building the projection matrix
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> STproj =
      STprojection(physbasis[index], canobasis);
  // and solving its null space
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> nullspace =
      phys_to_cano(physbasis[index], STproj);
  // after solving the nullspace, the dimension of the nullspace is the
  // multiplicity of (S,T) and the multiplicity within the physical basis state
  // labeling is modified

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  std::cout
      << "The Gelfand states and the expansion coefficients for zeta = 1,...,"
      << physbasis[index].zeta << " are:" << std::endl;

  for (int i = 0; i < canobasis.size(); i++)
    std::cout << "|" << canobasis[i] << "> \t" << nullspace.row(i) << std::endl;

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  return 0;
}

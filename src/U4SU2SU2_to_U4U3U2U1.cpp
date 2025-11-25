#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include <U4basis.h>
#include <U4SU2SU2_to_U4U3U2U1.h> 

#ifdef USE_SLEPC
#include <slepcsys.h>
#endif
  
int main(int argc, char* argv[])
{
#ifdef USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif

  int P1, P2, P3;
  std::cout << "Please enter SU(4) quantum numbers (P1,P2,P3): ";
  std::cin >> P1 >> P2 >> P3;

  u4::irrep x(P1, P2, P3); 

  // building the gelfand basis 
  std::vector<u4::gelfand> gArr = u4::gelfandBasis(x);

  // building the physical basis excluding those which have zero multiplicity
  std::vector<u4::physicalvector> pArr = u4::physicalBasis(x);

  std::cout << "The physical basis states are: " << std::endl;
  for (int i = 0; i < pArr.size(); i++)
  {
    std::cout << "(" << i << "): |(" << pArr[i].P1() << ","
              << pArr[i].P2() << "," << pArr[i].P3()
              << "), zeta_max=" << pArr[i].zeta << ", SS=" 
              << pArr[i].SS << ", TT=" << pArr[i].TT << ">" << std::endl;
  }

  int index;
  std::cout << "Please enter the index of the state that you are interested: ";
  std::cin >> index;
  const auto& p = pArr[index];

  int mode;
  std::cout << "Enter 0 for specific projections MS & MT, 1 for all: ";
  std::cin >> mode;

  if (mode == 0)
  {
    int MMS, MMT;
    std::cout << "Please enter projections 2MS <= 2S and 2MT <= 2T: ";
    std::cin >> MMS >> MMT;

    // Start time measurement
    auto t0 = std::chrono::high_resolution_clock::now();

    // get the transformation
    auto [canoBasis, transformation] = u4::phys_to_cano(p, MMS, MMT, gArr);

    // End time measurement
    auto t1 = std::chrono::high_resolution_clock::now(); 

    std::cout
      << "The Gelfand states and the expansion coefficients for zeta = 1,...,"
      << p.zeta << " are:" << std::endl;
 
    for (int i = 0; i < canoBasis.size(); i++)
      std::cout << "|" << gArr[canoBasis[i]] << "> \t" << transformation.row(i) 
              << std::endl; 

    // Calculate the duration
    std::chrono::duration<double> duration = t1 - t0;

    // Output the time in seconds
    std::cout << "Elapsed time: " << duration.count() << " seconds or "
              << duration.count() / 60.0 << " minutes" << std::endl;
  }
  else if (mode == 1)
  {
    // Start time measurement
    auto t0 = std::chrono::high_resolution_clock::now();

    // get the transformation
    auto [canoBasis, transformation] = u4::phys_to_cano_allM(p, gArr);

    // End time measurement
    auto t1 = std::chrono::high_resolution_clock::now();

    for (int mms = p.SS; mms >= -p.SS; mms -= 2)
    {
      int indexS = (p.SS - mms) / 2;

      for (int mmt = p.TT; mmt >= -p.TT; mmt -= 2)
      {
        int indexT = (p.TT - mmt) / 2;

        const auto& cano = canoBasis[indexS][indexT];
        const auto& trans = transformation[indexS][indexT];

        std::cout
          << "Gelfand states and expansion coefficients for MMS = " 
          << mms << " and MMT = " << mmt 
          << " for zeta = 1,...," << p.zeta << " are:" << std::endl;

        for (int i = 0; i < cano.size(); i++)
          std::cout << "|" << gArr[cano[i]] << "> \t" << trans.row(i)
                    << std::endl;
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
    std::cout << "Wrong mode" << std::endl;
  }

#ifdef USE_SLEPC
  SlepcFinalize();
#endif  

  return 0;
}

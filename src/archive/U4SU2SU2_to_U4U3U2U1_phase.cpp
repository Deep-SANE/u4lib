#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include <u3lib.h>

#include <u4lib.h>
#include <U4basis.h>
#include <U4SU2SU2_to_U4U3U2U1.h>

#include <qcx_utils.h>
  
int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  int P1, P2, P3;
  std::cout << "Please enter SU(4) quantum numbers (P1,P2,P3): ";
  std::cin >> P1 >> P2 >> P3;

  u4::irrep x(P1, P2, P3), x_conj = x.conjugate(); 

  // building the gelfand basis 
  std::vector<u4::gelfand> gArr = u4::gelfandBasis(x),
                           gArr_conj = u4::gelfandBasis(x_conj);

  // building the physical basis excluding those which have zero multiplicity
  std::vector<u4::physicalvector> pArr = u4::physicalBasis(x);

  std::cout << "The physical basis states are: " << std::endl;
  for (int i = 0; i < pArr.size(); i++)
  {
    std::cout << "(" << i << "): |(" << pArr[i].P1() << ","
              << pArr[i].P2() << "," << pArr[i].P3()
              << "), zeta_max=" << pArr[i].zeta() << ", SS=" 
              << pArr[i].SS() << ", TT=" << pArr[i].TT() << ">" << std::endl;
  }

  int index;
  std::cout << "Please enter the index of the state that you are interested: ";
  std::cin >> index;
  const auto& p = pArr[index];

  int MMS, MMT;
  std::cout << "Please enter projections 2MS <= 2S and 2MT <= 2T: ";
  std::cin >> MMS >> MMT;

  u4::physicalvector p_conj(x_conj, p.SS(), p.TT());
  double p_phase = ((2 * (P1 + P2 + P3) - p.SS() - p.TT() + MMS + MMT) % 4 == 0) 
                 ? 1.0 : -1.0;

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  // get the transformation
  auto [canoBasis, transformation] = u4::phys_to_cano(p, MMS, MMT, gArr);
  auto [canoBasis_conj, transformation_conj] = 
                          u4::phys_to_cano(p_conj, -MMS, -MMT, gArr_conj);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now(); 

  std::cout
      << "The Gelfand states and the expansion coefficients for zeta = 1,...,"
      << p.zeta() << " are:" << std::endl;
 
  for (int i = 0; i < canoBasis.size(); i++)
  {
    const auto& g = gArr[canoBasis[i]];
    auto g_conj = g.conjugate();
    
    std::cout << "|" << g << "> \t" << transformation.row(i) << "\t"; 

    for (int j = 0; j < canoBasis_conj.size(); j++)
    {
      if (g_conj == gArr_conj[canoBasis_conj[j]])
      {
        auto g_phase = gArr_conj[canoBasis_conj[j]].conjugate_phase();
        
        auto diff = p_phase * g_phase * transformation.row(i)
                  - transformation_conj.row(j);

        if (diff.norm() > 1e-12) std::cout << "\tViolate";
        else std::cout << "\tGood";

        break;
      }
    }
    
    std::cout << std::endl;
  }

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl;

  u4::finalize();
}

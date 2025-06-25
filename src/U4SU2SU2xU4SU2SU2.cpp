#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <U4decomposition.h>
#include <U4basis.h>
#include <U4SU2SU2xU4SU2SU2.h>

#ifdef U4LIB_USE_SLEPC
#include <slepcsys.h>
#endif 

int main(int argc, char* argv[])
{
#ifdef U4LIB_USE_SLEPC
  SlepcInitialize(&argc, &argv, NULL, NULL);
#endif
  
  int P1a, P2a, P3a, P1b, P2b, P3b;
  std::cout << "Please enter SU(4) quantum numbers (P1a,P2a,P3a): ";
  std::cin >> P1a >> P2a >> P3a;

  std::cout << "Please enter SU(4) quantum numbers (P1b,P2b,P3b): ";
  std::cin >> P1b >> P2b >> P3b;

  u4::irrep u4A(P1a, P2a, P3a), u4B(P1b, P2b, P3b);

  auto [u4Cs, etamaxs] = u4::decomposition(u4A, u4B);

  std::cout << "The coupled irreps (P1,P2,P3) are" << std::endl;
  for (int i = 0; i < u4Cs.size(); i++)
  {
    std::cout << "(" << i << "): (" << u4Cs[i].P1() << "," << u4Cs[i].P2()
              << "," << u4Cs[i].P3() << ") \t eta = 1 ... " << etamaxs[i]
              << std::endl;
  }

  int index;
  std::cout << "Please enter the index of the irrep you are interested in: ";
  std::cin >> index;
  u4::irrep u4C = u4Cs[index];
  int etamax = etamaxs[index];

  // Build the physical bases
  std::vector<u4::physicalvector> pArrA, pArrB, pArrC;

  // Input spin and isospin
  int ss1, tt1, ss2,  tt2, ss3, tt3;
  std::vector<int> SS1, TT1, SS2, TT2, SS3, TT3;
  std::cout << "Please enter twice spin and isospin values ss >= 0, tt >= 0,"
               "(enter -1 to stop, -2 to include all)"
            << std::endl;
  // first states
  std::cout << "ss1: ";
  while (true)
  {
    std::cin >> ss1;
    if (ss1 >= 0)
      SS1.push_back(ss1);

    // Stop condition
    if (ss1 == -1 || ss1 == -2)
      break;
  }
  std::cout << "tt1: ";
  while (true)
  {
    std::cin >> tt1;
    if (tt1 >= 0)
      TT1.push_back(tt1);

    // Stop condition
    if (tt1 == -1 || tt1 == -2)
      break;
  }
  if (ss1 == -2 && tt1 == -2)
    pArrA = u4::physicalBasis(u4A);
  else
  {
    for (int i = 0; i < SS1.size(); i++)
      for (int j = 0; j < TT1.size(); j++)
      {
        u4::physicalvector pA(u4A, SS1[i], TT1[j]);
        if (pA.zeta != 0)
          pArrA.push_back(pA);
      }
  }
  // second states
  std::cout << "ss2: ";
  while (true)
  {
    std::cin >> ss2;
    if (ss2 >= 0)
      SS2.push_back(ss2);

    // Stop condition
    if (ss2 == -1 || ss2 == -2)
      break;
  }
  std::cout << "tt2: ";
  while (true)
  {
    std::cin >> tt2;
    if (tt2 >= 0)
      TT2.push_back(tt2);

    // Stop condition
    if (tt2 == -1 || tt2 == -2)
      break;
  }
  if (ss2 == -2 && tt2 == -2)
    pArrB = u4::physicalBasis(u4B);
  else
  {
    for (int i = 0; i < SS2.size(); i++)
      for (int j = 0; j < TT2.size(); j++)
      {
        u4::physicalvector pB(u4B, SS2[i], TT2[j]);
        if (pB.zeta != 0)
          pArrB.push_back(pB);
      }
  }
  // third states
  std::cout << "ss3: ";
  while (true)
  {
    std::cin >> ss3;
    if (ss3 >= 0)
      SS3.push_back(ss3);

    // Stop condition
    if (ss3 == -1 || ss3 == -2)
      break;
  }
  std::cout << "tt3: ";
  while (true)
  {
    std::cin >> tt3;
    if (tt3 >= 0)
      TT3.push_back(tt3);

    // Stop condition
    if (tt3 == -1 || tt3 == -2)
      break;
  }
  if (ss3 == -2 && tt3 == -2)
    pArrC = u4::physicalBasis(u4C);
  else
  {
    for (int i = 0; i < SS3.size(); i++)
      for (int j = 0; j < TT3.size(); j++)
      {
        u4::physicalvector pC(u4C, SS3[i], TT3[j]);
        if (pC.zeta != 0)
          pArrC.push_back(pC);
      }
  }
 
  if (pArrA.empty() || pArrB.empty() || pArrC.empty())
  {
    std::cout << "Your spin and isospin values are invalid" << std::endl;
    return 1;
  } 

  // Start time measurement
  auto t0 = std::chrono::high_resolution_clock::now();

  std::vector<std::tuple<int, int, int>> tArr;
  u34utils::SO3_CGC cgcs;
  u3::WigU3U2_db wigU3U2db(cgcs);
  u4::WigU4U3_db wigU4U3db(cgcs, wigU3U2db);

  auto coeffs = u4::WigU4SU2SU2(etamax, pArrA, pArrB, pArrC, tArr, 
                                      cgcs, wigU3U2db, wigU4U3db);

  // End time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  // Print out
  for (int t = 0; t < tArr.size(); t++)
  {
    int i = std::get<0>(tArr[t]), j = std::get<1>(tArr[t]),
        k = std::get<2>(tArr[t]);

    std::cout << "Reduced coupling coefficients: < (" << P1a << "," << P2a << "," 
              << P3a << ") z1 " << pArrA[i].SS << " " << pArrA[i].TT 
              << " ; (" << P1b << "," << P2b << "," << P3b << ") z2 " 
              << pArrB[j].SS << " " << pArrB[j].TT << " || "
              << "(" << u4C.P1() << "," << u4C.P2() << "," << u4C.P3() 
              << ") z3 " << pArrC[k].SS << " " << pArrC[k].TT << " > "
              << "for eta = 1 ... " << etamax << std::endl;
    std::cout << "z1\tz2\tz3" << std::endl;

    Eigen::MatrixXd coeffs_temp = coeffs[t];

    for (int z1 = 0; z1 < pArrA[i].zeta; z1++)
      for (int z2 = 0; z2 < pArrB[j].zeta; z2++)
        for (int z3 = 0; z3 < pArrC[k].zeta; z3++)
          std::cout << "" << z1 + 1 << "\t" << z2 + 1 << "\t" << z3 + 1 << "\t"
            << coeffs_temp.row((z1 * pArrB[j].zeta + z2) * pArrC[k].zeta + z3)
            << std::endl;
  }

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
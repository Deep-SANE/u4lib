#include <chrono>
#include <iostream>
#include <omp.h>

#include <u3lib.h>

#include <u4lib.h>
#include <U4basis.h>
#include <U4generatorsRME.h>

#include <utils.h> 

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

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
  u4::irrep ir(P1, P2, P3);

  // building the physical basis excluding those which have zero multiplicity
  std::vector<u4::physicalvector> physbasis = u4::physicalBasis(ir);

  std::cout << "The physical basis states are: " << std::endl;
  for (int i = 0; i < physbasis.size(); i++)
  {
    std::cout << "(" << i << "): |(" << physbasis[i].P1() << ","
              << physbasis[i].P2() << "," << physbasis[i].P3()
              << "), zeta_max=" << physbasis[i].zeta()
              << ", SS=" << physbasis[i].SS() << ", TT=" << physbasis[i].TT() << ">"
              << std::endl;
  }

  std::cout
      << "This program computes reduced matrix elements of SU(4) generator: "
         "<(P1,P2,P3) zeta' SS' TT'|| T || (P1,P2,P3) zeta SS TT>"
      << std::endl;

  int option;
  std::cout << "Enter 0 to choose S or 1 to choose T or 2 to choose U: ";
  std::cin >> option;

  int i_bra;
  std::cout << "Enter the index of the bra that you are interested: ";
  std::cin >> i_bra;

  int i_ket;
  std::cout << "Enter the index of the ket that you are interested: ";
  std::cin >> i_ket;

  std::vector<int> bra_zeta, ket_zeta;
  Eigen::MatrixXd RME;

  if (option == 0)
    RME = u4::RME_S(physbasis[i_bra], physbasis[i_ket], bra_zeta, ket_zeta);
  else if (option == 1)
    RME = u4::RME_T(physbasis[i_bra], physbasis[i_ket], bra_zeta, ket_zeta);
  else if (option == 2)
    RME = u4::RME_U(physbasis[i_bra], physbasis[i_ket], bra_zeta, ket_zeta);

  std::cout << "\tzeta = ";
  for (int i = 0; i < ket_zeta.size(); i++)
    std::cout << ket_zeta[i] << "\t";
  std::cout << std::endl;
  std::cout << "zeta'" << std::endl;
  for (int i = 0; i < bra_zeta.size(); i++)
    std::cout << bra_zeta[i] << "\t\t" << RME.row(i) << std::endl;

  // -------------------------------------------------------------------------
  // Multi-threading demonstration:
  // Each thread calls u4::initialize_thread() before computing generator RMEs,
  // prints results along with its thread index, and calls u4::finalize_thread().
  // -------------------------------------------------------------------------
  std::cout << "\n[Multi-threading demonstration]" << std::endl;
#pragma omp parallel num_threads(4)
  {
    u4::initialize_thread();
    int tid = omp_get_thread_num();

    std::vector<int> b_zet, k_zet;
    Eigen::MatrixXd rme_t;
    if (option == 0)
      rme_t = u4::RME_S(physbasis[i_bra], physbasis[i_ket], b_zet, k_zet);
    else if (option == 1)
      rme_t = u4::RME_T(physbasis[i_bra], physbasis[i_ket], b_zet, k_zet);
    else
      rme_t = u4::RME_U(physbasis[i_bra], physbasis[i_ket], b_zet, k_zet);

#pragma omp critical
    {
      std::cout << "[Thread " << tid << "] Generator RME shape: (" << rme_t.rows()
                << "," << rme_t.cols() << "); norm = " << rme_t.norm() << std::endl;
    }

    u4::finalize_thread();
  }

  u4::finalize();
}

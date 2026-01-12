#include <U4decomposition.h>
#include <U4basis.h>
#include <U4SU2SU2xU4SU2SU2.h>

#include <qcx_utils.h> 


/* This program tests the following conjugation relation
   
   < (P1,P1',P1") z1 S1 T1; (1,0,0) 1/2 1/2 || (P3,P3',P3") z3 S3 T3 >_etamax=1
   = (-)^(phi - S1 - T1 - (phi" - S3 - T3) + 3/2)
   * sqrt(dim(P3,P3',P3") * dim(S1) * dim(T1) / dim(P1,P1',P1") / dim(S3) / dim(T3))
   * < (P3,P3',P3") z3 S3 T3; (0,0,1) 1/2 1/2 || (P1,P1',P1") z1 S1 T1 >_rhomax=1

   where phi(P,P',P") = (3P + 4P' + 3P")/2
*/

Eigen::MatrixXd diff(int z1, const Eigen::MatrixXd& m1, 
  					         int z3, const Eigen::MatrixXd& m3)
{
  Eigen::MatrixXd d(m1.rows(), m1.cols());

  for (size_t i = 0; i < z1; i++)
  {
  	for (size_t j = 0; j < z3; j++)
  	{
  	  auto m1_row = i * z3 + j;
  	  auto m3_row = j * z1 + i;

  	  d(m1_row,0) = m1(m1_row,0) - m3(m3_row,0);
  	}
  }

  return d;
}

int main(int argc, char* argv[])
{
  qcx_utils::initialize(&argc, &argv);

  auto& wigU4SU2SU2db = u4::wigU4SU2SU2db();

  int P1a, P2a, P3a;
  std::cout << "Please enter SU(4) quantum numbers (P1a,P2a,P3a): ";
  std::cin >> P1a >> P2a >> P3a;

  u4::irrep u4A(P1a, P2a, P3a), u4B(1, 0, 0);

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
  int P1c = u4C.P1(), P2c = u4C.P2(), P3c = u4C.P3(), etamax = etamaxs[index];

  std::vector<u4::physicalvector> pArrA = u4::physicalBasis(u4A), 
  	pArrB = u4::physicalBasis(u4B), pArrC = u4::physicalBasis(u4C);

  double dimA = u4A.dim(), dimC = u4C.dim();

  for (const auto& pA : pArrA)
  {
  	for (const auto& pB : pArrB)
  	{
  	  for (const auto& pC : pArrC)
  	  {
  	  	auto [flag, cgcU4] = wigU4SU2SU2db.get(
                          P1a, P2a, P3a, pA.SS(), pA.TT(),
  	  		                1, 0, 0, 1, 1, 
                          P1c, P2c, P3c, pC.SS(), pC.TT());

  	  	if (flag)
  	  	{
  	  	  auto [flag_conj, cgcU4_conj] = wigU4SU2SU2db.get(
                                P1c, P2c, P3c, pC.SS(), pC.TT(),
                                0, 0, 1, 1, 1, 
                                P1a, P2a, P3a, pA.SS(), pA.TT());
          assert(flag_conj);

  	  	  int phix2 = (3 * P1a + 4 * P2a + 3 * P3a) - pA.SS() - pA.TT()
                    - (3 * P1c + 4 * P2c + 3 * P3c) + pC.SS() + pC.TT() + 3;
  	  	  double phase = (phix2 % 4 == 0) ? 1.0 : -1.0;
  	  	  double factor = 
  	  	  	std::sqrt((pA.SS() + 1.0) * (pA.TT() + 1.0) * dimC 
                    / (pC.SS() + 1.0) / (pC.TT() + 1.0) / dimA);

          auto d = diff(pA.zeta(), cgcU4, pC.zeta(), phase * factor * cgcU4_conj);

  	  	  std::cout << "\nSS1 = " << pA.SS() << ", TT1 = " << pA.TT() 
                    << "; SS3 = " << pC.SS() << ", TT3 = " << pC.TT()
  	  	  			    << "\nCoupling:\n" << cgcU4 
  	  	  			    << "\nConjugate coupling:\n" << cgcU4_conj 
  	  	  			    << "\nDifference with factor: " << ((d.norm() > 1e-10) ? "Violated" : "")
                    //<< "\nDifference with factor: " << ((d.norm() > 1e-6) ? d : Eigen::MatrixXd(0,0))
  	  	  			    << std::endl;
  	  	}
  	  }
  	}
  }

  qcx_utils::finalize();
}

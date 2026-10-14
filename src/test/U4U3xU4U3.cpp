#include <cassert>

#include <utils.h>

#include <u3lib.h>

#include <u4lib.h>
#include <U4U3xU4U3.h>

int main(int argc, char* argv[])
{
  u4::initialize(&argc, &argv);

  u4::irrep u4a(2, 1, 1, 0);
  u4::irrep u4b(2, 1, 1, 0);
  u4::irrep u4c(2, 1, 1, 0);

  auto etamax = u4::Eta(u4a, u4b, u4c);

  auto gArrA = u4::gelfandBasis(u4a);
  auto gArrB = u4::gelfandBasis(u4b);
  auto gArrC = u4::gelfandBasis(u4c);

  auto coeffs =
      u4::WigU4U3(u4a, u4b, u4c, etamax, -1, gArrA, gArrB, gArrC);

  assert(coeffs.size() == 17);

  auto coeff = coeffs.access({1, 1, 1}, {1, 1, 1}, {1, 1, 1});
  assert(coeff.size() > 0);
  std::cout << "Single thread:\n" << coeff << std::endl;

std::cout << "Multiple threads:" << std::endl;
#pragma omp parallel num_threads(4)
  {
    u4::initialize_thread();

    auto coeff_t = u4::wigU4U3db().get(2, 1, 1, 0, 1, 1, 1,
                                                  2, 1, 1, 0, 1, 1, 1,
                                                  2, 1, 1, 0, 1, 1, 1);
    assert(coeff_t.size() > 0);
    assert(coeff_t.isApprox(coeff, 1e-10));
    std::cout << coeff_t << std::endl;

    u4::finalize_thread();
  }

  u4::finalize();
}

#include <cassert>

#include <qcx_utils.h>

#include <u3lib.h>

#include <u4lib.h>
#include <U4U3xU4U3.h>

int main(int argc, char* argv[])
{
  qcx_utils::initialize(&argc, &argv);
  u3::initialize();
  u4::initialize();

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

  auto [allow, coeff] = coeffs.access({1, 1, 1}, {1, 1, 1}, {1, 1, 1});
  assert(allow);
  std::cout << "Single thread:\n" << coeff << std::endl;

std::cout << "Multiple threads:" << std::endl;
#pragma omp parallel num_threads(4)
  {
    qcx_utils::initialize_thread();
    u3::initialize_thread();
    u4::initialize_thread();

    auto [allow_t, coeff_t] = u4::wigU4U3db().get(2, 1, 1, 0, 1, 1, 1,
                                                  2, 1, 1, 0, 1, 1, 1,
                                                  2, 1, 1, 0, 1, 1, 1);
    assert(allow_t);
    std::cout << coeff_t << std::endl;

    u4::finalize_thread();
    u3::finalize_thread();
    qcx_utils::finalize_thread();
  }

  u4::finalize();
  u3::finalize();
  qcx_utils::finalize();
}

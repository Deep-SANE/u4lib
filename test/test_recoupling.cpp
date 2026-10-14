#include "recoupling_cases.h"
int main(int argc, char** argv)
{
  u4::initialize(&argc, &argv);
  test::Suite suite;
  try
  {
    test::Reference ref("recoupling");
    test::recoupling(suite, ref);
    suite.run("recoupling: all documented entries exercised",
              [&] { ref.finish(); });
  }
  catch (const std::exception& e)
  {
    ++suite.failed;
    std::cout << "[FAIL] " << e.what() << '\n';
  }
  u4::finalize();
  return suite.result();
}

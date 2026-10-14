#include "physical_cases.h"
int main(int argc, char** argv)
{
  u4::initialize(&argc, &argv);
  test::Suite suite;
  try
  {
    test::Reference ref("physical");
    test::physical(suite, ref);
    suite.run("physical: all documented entries exercised",
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

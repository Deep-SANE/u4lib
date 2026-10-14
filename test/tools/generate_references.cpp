#include "../canonical_cases.h"
#include "../conjugation_cases.h"
#include "../physical_cases.h"
#include "../recoupling_cases.h"
int main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cerr << "Usage: generate_references EXISTING_OUTPUT_DIRECTORY\n";
    return 2;
  }
  std::string directory = argv[1];
  u4::initialize(&argc, &argv);
  test::Suite suite;
  try
  {
    {
      test::Reference ref("canonical", directory, true);
      test::canonical(suite, ref);
      ref.finish();
    }
    {
      test::Reference ref("physical", directory, true);
      test::physical(suite, ref);
      ref.finish();
    }
    {
      test::Reference ref("conjugation", directory, true);
      test::conjugation(suite, ref);
      ref.finish();
    }
    {
      test::Reference ref("recoupling", directory, true);
      test::recoupling(suite, ref);
      ref.finish();
    }
  }
  catch (const std::exception& e)
  {
    ++suite.failed;
    std::cerr << e.what() << '\n';
  }
  u4::finalize();
  return suite.result();
}

#include "test_helpers.h"
int main(int argc, char** argv)
{
  u4::initialize(&argc, &argv);
  test::Suite suite;
  for (int a = 0; a <= 5; ++a)
    for (int b = 0; b <= 5; ++b)
      for (int c = 0; c <= 5; ++c)
      {
        u4::irrep ir(a, b, c);
        suite.run(
            "Basis " + test::irname(ir) +
                ": dimension, branching, Gelfand labels and conjugation",
            [&]
            {
              auto gs = u4::gelfandBasis(ir);
              long dim = long(a + 1) * (b + 1) * (c + 1) * (a + b + 2) *
                         (b + c + 2) * (a + b + c + 3) / 12;
              test::require(gs.size() == size_t(dim) && ir.dim() == dim,
                            "SU(4) Weyl dimension");
              long physical_dim = 0;
              for (auto p : u4::physicalBasis(ir))
                physical_dim += long(p.SS() + 1) * (p.TT() + 1) * p.zeta();
              test::require(physical_dim == dim,
                            "Spin-isospin branching dimension");
              std::array<int, 6> previous = {999, 999, 999, 999, 999, 999};
              for (auto g : gs)
              {
                test::require(g.check(), "Gelfand interlacing");
                std::array<int, 6> current = {g.n13(), g.n23(), g.n33(),
                                              g.n12(), g.n22(), g.n11()};
                test::require(current < previous, "Gelfand descending order");
                previous = current;
                test::require(g.conjugate().conjugate() == g,
                              "Gelfand double conjugation labels");
                test::near(g.conjugate_phase() *
                               g.conjugate().conjugate_phase(),
                           test::sign(a + c), "SU(4) conjugation squared sign");
              }
              for (int i : {0, int(gs.size() / 2), int(gs.size() - 1)})
                test::require(u4::findGelfand(gs[i], gs) == size_t(i),
                              "Gelfand lookup");
              u4::irrep shifted(ir.n14() + 1, ir.n24() + 1, ir.n34() + 1, 1);
              shifted.normalize();
              test::require(shifted == ir, "Determinant shift normalization");
              test::require(u4::Zeta(ir, -1, 0) == 0 &&
                                u4::Zeta(ir, 999, 0) == 0,
                            "Forbidden spin-isospin channels");
            });
      }
  suite.run("Fundamental decomposition 4x4=10+6 and 4x4bar=1+15",
            [&]
            {
              for (auto b : std::vector<u4::irrep>{{1, 0, 0}, {0, 0, 1}})
              {
                auto [out, m] = u4::decomposition({1, 0, 0}, b);
                int dimension = 0;
                test::require(out.size() == 2,
                              "Two fundamental product channels");
                for (int i = 0; i < int(out.size()); ++i)
                {
                  test::require(m[i] == 1, "Multiplicity one");
                  dimension += m[i] * out[i].dim();
                }
                test::require(dimension == 16, "Tensor product dimension");
              }
            });
  u4::finalize();
  return suite.result();
}

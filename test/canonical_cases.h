#pragma once
#include "test_helpers.h"
namespace test
{
  inline void canonical(Suite& suite, Reference& ref)
  {
    for (const auto& x : couplings())
      suite.run(
          "Canonical " + label(x) +
              ": complete reduced blocks against documented values and "
              "normalization",
          [&]
          {
            int eta = u4::Eta(x.a, x.b, x.c);
            require(eta > 0, "Allowed coupling");
            auto table = u4::wigU4U3db().get_coeffs(x.a, x.b, x.c);
            require(bool(table), "Canonical coefficient table");
            int blocks = 0;
            for (auto c : subreps(x.c))
            {
              Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(eta, eta);
              for (auto a : subreps(x.a))
                for (auto b : subreps(x.b))
                {
                  auto v = table->find_block(a, b, c);
                  if (v.size() == 0)
                    continue;
                  ++blocks;
                  require(v.rows() == u3::Rho(a, b, c) && v.cols() == eta,
                          "rho/eta block dimensions");
                  gram += v.transpose() * v;
                  for (int r = 0; r < v.rows(); ++r)
                    for (int e = 0; e < eta; ++e)
                      ref.check(key(label(x), subname(a), subname(b),
                                    subname(c), r, e),
                                v(r, e));
                }
              near((gram - Eigen::MatrixXd::Identity(eta, eta)).norm(), 0,
                   "Reduced canonical orthogonality", 1e-9, 0);
            }
            require(blocks == table->size(), "All canonical blocks exercised");
            auto ga = u4::gelfandBasis(x.a), gb = u4::gelfandBasis(x.b),
                 gc = u4::gelfandBasis(x.c);
            std::set<int> selected = {0, int(gc.size() / 2),
                                      int(gc.size() - 1)};
            for (int k : selected)
            {
              auto ids = u4::expand_to_uncoupled(gc[k], ga, gb);
              Eigen::MatrixXd v(ids.size(), eta);
              for (int i = 0; i < int(ids.size()); ++i)
              {
                auto a = ga[ids[i].first], b = gb[ids[i].second];
                v.row(i) = full_cg(a, b, gc[k], *table, eta);
                if (x.a.zero())
                  near(v(i, 0), b == gc[k] ? 1 : 0, "Analytic scalar identity");
                if (x.c.zero())
                  near(v(i, 0),
                       a.conjugate() == b
                           ? a.conjugate_phase() / std::sqrt(double(ga.size()))
                           : 0,
                       "Analytic singlet");
              }
              orthogonal(v, "Reconstructed full canonical CG orthogonality");
            }
          });
    suite.run("Tutorial canonical block [2110]x[3210]->[3210]: rho and eta "
              "multiplicities",
              [&]
              {
                auto table = u4::wigU4U3db().get_coeffs(
                    {2, 1, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0});
                auto block = table->find_block({2, 1, 0}, {3, 2, 0}, {3, 2, 0});
                require(block.rows() > 1 && block.cols() > 1,
                        "Tutorial block has rho and eta multiplicities");
                // Its complete values were checked above under the explicit
                // coupling labels.
              });
  }
} // namespace test

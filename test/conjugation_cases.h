#pragma once
#include "test_helpers.h"
namespace test
{
  inline auto gtkey(const u4::gelfand& g)
  {
    return std::array<int, 6>{g.n13(), g.n23(), g.n33(),
                              g.n12(), g.n22(), g.n11()};
  }
  inline void conjugation(Suite& suite, Reference& ref)
  {
    for (const auto& ir : physical_irreps())
      suite.run(
          "Conjugation " + irname(ir) +
              ": documented Gelfand signs and physical action, every "
              "S/T/MS/MT, both lowering operators",
          [&]
          {
            auto bar = ir.conjugate();
            auto gs = u4::gelfandBasis(ir), bs = u4::gelfandBasis(bar);
            std::map<std::array<int, 6>, int> lookup;
            for (int i = 0; i < int(bs.size()); ++i)
              lookup[gtkey(bs[i])] = i;
            std::vector<int> permutation(gs.size());
            for (int i = 0; i < int(gs.size()); ++i)
            {
              auto g = gs[i];
              permutation[i] = lookup.at(gtkey(g.conjugate()));
              near(g.conjugate_phase() * g.conjugate().conjugate_phase(),
                   sign(ir.P1() + ir.P3()), "Gelfand C squared");
              ref.check(key(irname(ir), "GT", int(g.n13()), int(g.n23()),
                            int(g.n33()), int(g.n12()), int(g.n22()),
                            int(g.n11())),
                        g.conjugate_phase());
            }
            for (auto p : u4::physicalBasis(ir))
            {
              auto [ids, vs] = u4::phys_to_cano_allM(p, gs);
              auto [bids, bvs] = u4::phys_to_cano_allM(
                  u4::physicalvector(bar, p.SS(), p.TT()), bs);
              Eigen::MatrixXd D;
              for (int s = 0; s <= p.SS(); ++s)
                for (int t = 0; t <= p.TT(); ++t)
                {
                  int ms = p.SS() - 2 * s, mt = p.TT() - 2 * t;
                  auto f = full(ids[s][t], vs[s][t], gs.size());
                  auto bf = full(bids[p.SS() - s][p.TT() - t],
                                 bvs[p.SS() - s][p.TT() - t], bs.size());
                  auto [singleids, singlev] = u4::phys_to_cano(p, ms, mt, gs);
                  near((f - full(singleids, singlev, gs.size())).norm(), 0,
                       "Single/all-M agreement", 1e-9, 0);
                  orthogonal(f, "Physical basis orthogonality");
                  if (s > 0)
                  {
                    auto [ii, v] =
                        u4::lowerMS(p, ms, mt, gs, ids[s - 1][t], vs[s - 1][t]);
                    near((f - full(ii, v, gs.size())).norm(), 0,
                         "Explicit S_- lowering", 1e-9, 0);
                  }
                  if (t > 0)
                  {
                    auto [ii, v] =
                        u4::lowerMT(p, ms, mt, gs, ids[s][t - 1], vs[s][t - 1]);
                    near((f - full(ii, v, gs.size())).norm(), 0,
                         "Explicit T_- lowering", 1e-9, 0);
                  }
                  Eigen::MatrixXd mapped =
                      Eigen::MatrixXd::Zero(bs.size(), f.cols());
                  for (int i = 0; i < int(gs.size()); ++i)
                    mapped.row(permutation[i]) =
                        gs[i].conjugate_phase() * f.row(i);
                  if (ir.P1() != ir.P3())
                  {
                    // C_GT^2=(-1)^(P1+P3), so the reverse irrep direction
                    // carries that sign.
                    double phase =
                        sign(ir.P1() + ir.P2() + ir.P3() -
                             (p.SS() + p.TT() - ms - mt) / 2 +
                             (ir.P1() < ir.P3() ? ir.P1() + ir.P3() : 0));
                    near((mapped - phase * bf).norm(), 0,
                         "Distinct conjugate irreps", 1e-9, 0);
                    if (s == 0 && t == 0)
                      ref.check(key(irname(ir), "phase", p.SS(), p.TT()),
                                phase);
                  }
                  else
                  {
                    double rotation = sign(s + t);
                    Eigen::MatrixXd d = rotation * bf.transpose() * mapped;
                    if (s == 0 && t == 0)
                    {
                      D = d;
                      for (int a = 0; a < d.rows(); ++a)
                        for (int b = 0; b < d.cols(); ++b)
                          ref.check(key(irname(ir), "D", p.SS(), p.TT(), a, b),
                                    d(a, b));
                    }
                    near((d - D).norm(), 0,
                         "Conjugation matrix projection independence", 1e-9, 0);
                    near((mapped - rotation * bf * D).norm(), 0,
                         "Self-conjugate reconstruction", 1e-9, 0);
                    near((D - D.transpose()).norm(), 0,
                         "Self-conjugate symmetry", 1e-9, 0);
                    near((D * D - Eigen::MatrixXd::Identity(D.rows(), D.cols()))
                             .norm(),
                         0, "Self-conjugate C squared", 1e-9, 0);
                  }
                }
            }
          });
  }
} // namespace test

#pragma once
#include "test_helpers.h"
namespace test
{
  inline void physical(Suite& suite, Reference& ref)
  {
    for (const auto& x : couplings())
      suite.run(
          "Physical " + label(x) +
              ": all spin-isospin blocks against documented values, "
              "orthogonality and canonical contraction",
          [&]
          {
            auto pa = u4::physicalBasis(x.a), pb = u4::physicalBasis(x.b),
                 pc = u4::physicalBasis(x.c);
            int eta = u4::Eta(x.a, x.b, x.c);
            auto [indices, coeffs] = u4::WigU4SU2SU2(eta, pa, pb, pc);
            require(!coeffs.empty(), "Nonempty physical blocks");
            for (int k = 0; k < int(pc.size()); ++k)
            {
              int n = pc[k].zeta() * eta;
              Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(n, n);
              for (int t = 0; t < int(indices.size()); ++t)
              {
                auto [i, j, z] = indices[t];
                if (z != k)
                  continue;
                const auto& v = coeffs[t];
                require(v.size() == size_t(pa[i].zeta() * pb[j].zeta() * n),
                        "Physical block dimensions");
                for (int a = 0; a < pa[i].zeta(); ++a)
                  for (int b = 0; b < pb[j].zeta(); ++b)
                  {
                    Eigen::VectorXd row(n);
                    for (int c = 0; c < pc[k].zeta(); ++c)
                      for (int e = 0; e < eta; ++e)
                      {
                        double value = v(a, b, c, e);
                        row[c * eta + e] = value;
                        ref.check(key(label(x), pa[i].SS(), pa[i].TT(),
                                      pb[j].SS(), pb[j].TT(), pc[k].SS(),
                                      pc[k].TT(), a, b, c, e),
                                  value);
                        if (x.a.zero())
                          near(value, b == c ? 1 : 0,
                               "Scalar physical identity");
                      }
                    gram += row * row.transpose();
                  }
              }
              near((gram - Eigen::MatrixXd::Identity(n, n)).norm(), 0,
                   "Physical reduced orthogonality", 1e-9, 0);
            }
            // Reconstruct one complete zeta/eta block at low magnetic
            // projections.
            auto [i, j, k] = indices[indices.size() / 2];
            auto ga = u4::gelfandBasis(x.a), gb = u4::gelfandBasis(x.b),
                 gc = u4::gelfandBasis(x.c);
            int mcS = pc[k].SS() % 2, mcT = pc[k].TT() % 2;
            auto choose = [](int ja, int jb, int jc, int mc)
            {
              int ma = 0;
              double cg = 0;
              for (int m = -ja; m <= ja; m += 2)
              {
                int mb = mc - m;
                if (std::abs(mb) > jb || (jb - mb) % 2)
                  continue;
                double candidate =
                    utils::cgcs().get({ja, jb, jc, m, mb, mc});
                if (std::abs(candidate) > std::abs(cg))
                {
                  cg = candidate;
                  ma = m;
                }
              }
              require(std::abs(cg) > 1e-12, "Nonzero spin/isospin CG");
              return std::make_pair(ma, cg);
            };
            auto [maS, cgS] = choose(pa[i].SS(), pb[j].SS(), pc[k].SS(), mcS);
            auto [maT, cgT] = choose(pa[i].TT(), pb[j].TT(), pc[k].TT(), mcT);
            auto [ia, ta] = u4::phys_to_cano(pa[i], maS, maT, ga);
            auto [ib, tb] = u4::phys_to_cano(pb[j], mcS - maS, mcT - maT, gb);
            auto [ic, tc] = u4::phys_to_cano(pc[k], mcS, mcT, gc);
            auto table = u4::wigU4U3db().get_coeffs(x.a, x.b, x.c);
            const auto& actual = coeffs[indices.size() / 2];
            std::vector<double> contracted(actual.size(), 0);
            std::map<int, int> ra, rb;
            for (int a = 0; a < int(ia.size()); ++a)
              ra[ia[a]] = a;
            for (int b = 0; b < int(ib.size()); ++b)
              rb[ib[b]] = b;
            for (int c = 0; c < int(ic.size()); ++c)
              for (auto [a, b] : u4::expand_to_uncoupled(gc[ic[c]], ga, gb))
              {
                if (!ra.count(a) || !rb.count(b))
                  continue;
                auto cg = full_cg(ga[a], gb[b], gc[ic[c]], *table, eta);
                for (int za = 0; za < ta.cols(); ++za)
                  for (int zb = 0; zb < tb.cols(); ++zb)
                    for (int zc = 0; zc < tc.cols(); ++zc)
                      for (int e = 0; e < eta; ++e)
                        contracted[((za * tb.cols() + zb) * tc.cols() + zc) *
                                       eta +
                                   e] +=
                            ta(ra[a], za) * tb(rb[b], zb) * tc(c, zc) * cg[e];
              }
            for (int e = 0; e < int(actual.size()); ++e)
              near(contracted[e], cgS * cgT * actual[e],
                   "Independent canonical contraction", 1e-10, 1e-10);
          });
    suite.run(
        "Tutorial physical block (111) (SS,TT)=(2,2) x (111) (4,2) -> (121) "
        "(4,2): zeta and eta multiplicities",
        [&]
        {
          const auto v = u4::wigU4SU2SU2db().get_U4SU2SU2(
              1, 1, 1, 2, 2, 1, 1, 1, 4, 2, 1, 2, 1, 4, 2);
          require(!v.empty() && v.etamax() > 1, "Tutorial outer multiplicity");
          require(v.zetaAmax() > 1 || v.zetaBmax() > 1 || v.zetaCmax() > 1,
                  "Tutorial branching multiplicity");
          // Complete values, including this block, are documented above.
        });
  }
} // namespace test

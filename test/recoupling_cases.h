#pragma once
#include "test_helpers.h"
namespace test
{
  struct TripleChannel
  {
    u4::irrep intermediate;
    int pair_rho, total_rho;
  };

  // Complete multiplicity-resolved basis for (A x B) x C -> total.
  inline auto triple_channels(const u4::irrep& a, const u4::irrep& b,
                              const u4::irrep& c, const u4::irrep& total)
  {
    std::vector<TripleChannel> channels;
    auto [irreps, multiplicities] = u4::decomposition(a, b);
    for (int i = 0; i < int(irreps.size()); ++i)
      for (int r = 0; r < multiplicities[i]; ++r)
        for (int t = 0; t < u4::Eta(irreps[i], c, total); ++t)
        {
          require(irreps[i].P1() <= 5 && irreps[i].P2() <= 5 &&
                      irreps[i].P3() <= 5,
                  "Intermediate irrep exceeds test bound");
          channels.push_back({irreps[i], r, t});
        }
    return channels;
  }

  inline void multiplicity_u6z6(Suite& suite, Reference& ref, u4::irrep a,
                                u4::irrep b, u4::irrep c, u4::irrep total,
                                int expected_dimension)
  {
    auto left = triple_channels(a, b, c, total);
    // U6 uses A x (B x C); multiplicity counts agree with (B x C) x A,
    // but coefficient retrieval below preserves the actual coupling order.
    auto rightU = triple_channels(b, c, a, total);
    auto rightZ = triple_channels(a, c, b, total);
    for (bool isZ : {false, true})
    {
      std::string kind = isZ ? "Z6" : "U6";
      suite.run(
          kind + " " + irname(a) + "x" + irname(b) + "x" + irname(c) + " -> " +
              irname(total) +
              ": complete multiplicity matrix against documented values and "
              "orthogonality",
          [&]
          {
            const auto& right = isZ ? rightZ : rightU;
            require((expected_dimension == 0 ||
                     left.size() == size_t(expected_dimension)) &&
                        right.size() == left.size(),
                    "Complete recoupling basis dimension");
            ref.check(key(kind + "dimension", irname(a), irname(b), irname(c),
                          irname(total)),
                      double(left.size()));
            Eigen::MatrixXd matrix(left.size(), right.size());
            bool pair_multiplicity = false, total_multiplicity = false;
            for (int i = 0; i < int(left.size()); ++i)
              for (int j = 0; j < int(right.size()); ++j)
              {
                const auto& l = left[i];
                const auto& r = right[j];
                int rAB = u4::Eta(a, b, l.intermediate),
                    rAB_C = u4::Eta(l.intermediate, c, total);
                int rPair = isZ ? u4::Eta(a, c, r.intermediate)
                                : u4::Eta(b, c, r.intermediate);
                int rTotal = isZ ? u4::Eta(r.intermediate, b, total)
                                 : u4::Eta(a, r.intermediate, total);
                pair_multiplicity |= rAB > 1 || rPair > 1;
                total_multiplicity |= rAB_C > 1 || rTotal > 1;
                double value;
                if (isZ)
                {
                  const auto block = u4::u4z6db().get_U4Z6(
                      b, a, total, c, l.intermediate, rAB, rAB_C,
                      r.intermediate, rPair, rTotal);
                  require(block.size() == size_t(rAB * rAB_C * rPair * rTotal),
                          "Z6 block dimensions");
                  value =
                      block(l.pair_rho, l.total_rho, r.pair_rho, r.total_rho);
                }
                else
                {
                  const auto block = u4::u4u6db().get_U4U6(
                      a, b, total, c, l.intermediate, rAB, rAB_C,
                      r.intermediate, rPair, rTotal);
                  require(block.size() == size_t(rAB * rAB_C * rPair * rTotal),
                          "U6 block dimensions");
                  value =
                      block(l.pair_rho, l.total_rho, r.pair_rho, r.total_rho);
                }
                matrix(i, j) = value;
                ref.check(key(kind + "multi", irname(a), irname(b), irname(c),
                              irname(total), irname(l.intermediate), l.pair_rho,
                              l.total_rho, irname(r.intermediate), r.pair_rho,
                              r.total_rho),
                          value);
              }
            require(total_multiplicity,
                    "Expected nontrivial final coupling multiplicity");
            if (a == b && b == c)
              require(pair_multiplicity,
                      "Expected nontrivial pair multiplicity");
            orthogonal(matrix, kind + " complete column orthogonality");
            orthogonal(matrix.transpose(),
                       kind + " complete row orthogonality");
          });
    }
  }

  struct FourChannel
  {
    u4::irrep first, second;
    int first_rho, second_rho, total_rho;
  };
  inline auto four_channels(const u4::irrep& a, const u4::irrep& b,
                            const u4::irrep& c, const u4::irrep& d,
                            const u4::irrep& total)
  {
    std::vector<FourChannel> channels;
    auto [ab, rab] = u4::decomposition(a, b);
    auto [cd, rcd] = u4::decomposition(c, d);
    for (int i = 0; i < int(ab.size()); ++i)
      for (int j = 0; j < int(cd.size()); ++j)
        for (int p = 0; p < rab[i]; ++p)
          for (int q = 0; q < rcd[j]; ++q)
            for (int r = 0; r < u4::Eta(ab[i], cd[j], total); ++r)
            {
              require(ab[i].P1() <= 5 && ab[i].P2() <= 5 && ab[i].P3() <= 5 &&
                          cd[j].P1() <= 5 && cd[j].P2() <= 5 && cd[j].P3() <= 5,
                      "Intermediate irrep exceeds test bound");
              channels.push_back({ab[i], cd[j], p, q, r});
            }
    return channels;
  }

  inline void multiplicity_u9(Suite& suite, Reference& ref, u4::irrep total,
                              int expected_dimension)
  {
    u4::irrep f(1, 0, 1);
    suite.run(
        "U9 four (1,0,1) irreps -> " + irname(total) +
            ": complete multiplicity matrix against documented values and "
            "orthogonality",
        [&]
        {
          auto channels = four_channels(f, f, f, f, total);
          require(expected_dimension == 0 ||
                      channels.size() == size_t(expected_dimension),
                  "Complete U9 basis dimension");
          ref.check(key("U9dimension", irname(f), irname(total)),
                    double(channels.size()));
          Eigen::MatrixXd matrix(channels.size(), channels.size());
          bool pair_multiplicity = false, total_multiplicity = false;
          for (int i = 0; i < int(channels.size()); ++i)
            for (int j = 0; j < int(channels.size()); ++j)
            {
              const auto& l = channels[i];
              const auto& r = channels[j];
              int rab = u4::Eta(f, f, l.first), rcd = u4::Eta(f, f, l.second),
                  rl = u4::Eta(l.first, l.second, total);
              int rac = u4::Eta(f, f, r.first), rbd = u4::Eta(f, f, r.second),
                  rr = u4::Eta(r.first, r.second, total);
              pair_multiplicity |= rab > 1 && rcd > 1 && rac > 1 && rbd > 1;
              total_multiplicity |= rl > 1 && rr > 1;
              const auto block = u4::u4u9db().get_U4U9(
                  f, f, f, f, l.first, l.second, r.first, r.second, total, rac,
                  rbd, rl, rab, rcd, rr);
              require(block.size() == size_t(rab * rcd * rl * rac * rbd * rr),
                      "U9 block dimensions");
              // The structured API's axis order is AC, BD, (AB CD), AB, CD, (AC
              // BD).
              double value = block(r.first_rho, r.second_rho, l.total_rho,
                                   l.first_rho, l.second_rho, r.total_rho);
              matrix(i, j) = value;
              ref.check(key("U9multi", irname(f), irname(total),
                            irname(l.first), irname(l.second), l.first_rho,
                            l.second_rho, l.total_rho, irname(r.first),
                            irname(r.second), r.first_rho, r.second_rho,
                            r.total_rho),
                        value);
            }
          require(pair_multiplicity,
                  "Expected multiplicity on all four pair-coupling axes");
          if (total == f)
            require(total_multiplicity,
                    "Expected multiplicity on both final-coupling axes");
          orthogonal(matrix, "U9 complete column orthogonality");
          orthogonal(matrix.transpose(), "U9 complete row orthogonality");
        });
  }

  inline void recoupling(Suite& suite, Reference& ref)
  {
    suite.run(
        "Scalar U6/Z6/U9 against documented values and analytic identities",
        [&]
        {
          u4::irrep z(0, 0, 0);
          auto u = u4::u4u6db().get(z, z, z, z, z, 1, 1, z, 1, 1);
          auto v = u4::u4z6db().get(z, z, z, z, z, 1, 1, z, 1, 1);
          auto w =
              u4::u4u9db().get(z, z, z, z, z, z, z, z, z, 1, 1, 1, 1, 1, 1);
          require(u.size() == 1 && v.size() == 1 && w.size() == 1,
                  "Scalar recoupling dimensions");
          ref.check("scalar:U6:", u[0]);
          ref.check("scalar:Z6:", v[0]);
          ref.check("scalar:U9:", w[0]);
          near(u[0], 1, "Scalar U6");
          near(v[0], 1, "Scalar Z6");
          near(w[0], 1, "Scalar U9");
        });
    // The tutorial uses adjoints [2110], AB=total=[3100], BC=[2110], AC=[2200].
    multiplicity_u6z6(suite, ref, {1, 0, 1}, {1, 0, 1}, {1, 0, 1}, {2, 1, 0},
                      0);
    multiplicity_u6z6(suite, ref, {1, 0, 1}, {1, 0, 1}, {1, 0, 1}, {1, 0, 1},
                      0);
    multiplicity_u6z6(suite, ref, {1, 1, 1}, {1, 0, 1}, {1, 0, 1}, {1, 1, 1},
                      0);
    // Full matrices include the tutorial's all-adjoint intermediate block.
    multiplicity_u9(suite, ref, {0, 0, 0}, 0);
    multiplicity_u9(suite, ref, {1, 0, 1}, 0);
  }
} // namespace test

#pragma once
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <u4lib.h>

namespace test
{
  inline void require(bool ok, const std::string& what)
  {
    if (!ok)
      throw std::runtime_error(what);
  }
  inline void near(double actual, double expected, const std::string& what,
                   double atol = 1e-10, double rtol = 1e-8)
  {
    require(std::isfinite(actual) && std::isfinite(expected) &&
                std::abs(actual - expected) <= atol + rtol * std::abs(expected),
            what);
  }
  inline double sign(int n) { return n % 2 ? -1.0 : 1.0; }
  inline std::string irname(const u4::irrep& ir)
  {
    return "(" + std::to_string(ir.P1()) + "," + std::to_string(ir.P2()) + "," +
           std::to_string(ir.P3()) + ")";
  }
  template <class... T>
  std::string key(T... args)
  {
    std::ostringstream out;
    ((out << args << ':'), ...);
    return out.str();
  }
  // Generation is a separate executable, never enabled by the test runner.
  class Reference
  {
    std::map<std::string, double> expected;
    std::ofstream output;
    std::string filename;

  public:
    Reference(const std::string& name,
              const std::string& directory = "reference", bool generate = false)
        : filename(directory + "/" + name + ".dat")
    {
      if (generate)
      {
        output.open(filename);
        require(bool(output), "Cannot write " + filename);
        output << "# Signed RREF-canonicalized reference values; labels and "
                  "provenance: reference/README.md\n"
               << std::setprecision(17);
      }
      else
      {
        std::ifstream in(filename);
        require(bool(in), "Cannot read " + filename);
        std::string line;
        while (std::getline(in, line))
        {
          if (line.empty() || line[0] == '#')
            continue;
          std::istringstream row(line);
          std::string label, extra;
          double value;
          require(bool(row >> label >> value) && !(row >> extra) &&
                      std::isfinite(value),
                  "Malformed reference in " + filename);
          require(expected.emplace(label, value).second,
                  "Duplicate reference " + label);
        }
        require(!expected.empty(), "Empty reference " + filename);
      }
    }
    void check(const std::string& label, double value)
    {
      require(std::isfinite(value), "Nonfinite result: " + label);
      if (output.is_open())
      {
        output << label << ' ' << value << '\n';
        return;
      }
      auto it = expected.find(label);
      std::ostringstream msg;
      msg << "Documented-value mismatch: " << label << " actual=" << value << " expected=" << it->second << " diff=" << std::abs(value - it->second);
      near(value, it->second, msg.str());
      expected.erase(it);
    }
    void finish()
    {
      require(expected.empty(), "Unused documented values in " + filename);
      if (output.is_open())
      {
        output.flush();
        require(bool(output), "Write failed: " + filename);
      }
    }
  };
  struct Suite
  {
    int passed = 0, failed = 0;
    template <class F>
    void run(const std::string& label, F work)
    {
      std::cout << "[TEST] " << label << std::endl;
      try
      {
        work();
        ++passed;
        std::cout << "[PASS] " << label << std::endl;
      }
      catch (const std::exception& e)
      {
        ++failed;
        std::cout << "[FAIL] " << label << ": " << e.what() << std::endl;
      }
    }
    int result() const
    {
      std::cout << "Summary: " << passed << " passed, " << failed
                << " failed\n";
      return failed ? 1 : 0;
    }
  };
  inline Eigen::MatrixXd full(const std::vector<int>& ids,
                              const Eigen::MatrixXd& v, int dim)
  {
    Eigen::MatrixXd f = Eigen::MatrixXd::Zero(dim, v.cols());
    for (int i = 0; i < int(ids.size()); ++i)
      f.row(ids[i]) = v.row(i);
    return f;
  }
  inline void orthogonal(const Eigen::MatrixXd& v, const std::string& label)
  {
    near((v.transpose() * v - Eigen::MatrixXd::Identity(v.cols(), v.cols()))
             .norm(),
         0, label, 1e-9, 0);
  }
  struct Coupling
  {
    u4::irrep a, b, c;
  };
  inline std::vector<Coupling> couplings()
  {
    return {
        {{0, 0, 0}, {1, 2, 1}, {1, 2, 1}}, {{1, 0, 0}, {1, 0, 0}, {2, 0, 0}},
        {{1, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {{1, 0, 0}, {0, 0, 1}, {0, 0, 0}},
        {{1, 0, 0}, {0, 0, 1}, {1, 0, 1}}, {{1, 0, 1}, {1, 1, 1}, {1, 1, 1}},
        {{1, 1, 1}, {1, 1, 1}, {1, 2, 1}}, {{1, 0, 1}, {1, 0, 1}, {1, 0, 1}},
        {{2, 1, 0}, {1, 0, 1}, {2, 1, 0}}, {{5, 0, 0}, {1, 0, 1}, {5, 0, 0}},
        {{0, 0, 5}, {1, 0, 1}, {0, 0, 5}}, {{0, 5, 0}, {1, 0, 1}, {0, 5, 0}}};
  }
  inline std::string label(const Coupling& x)
  {
    return irname(x.a) + "x" + irname(x.b) + "->" + irname(x.c);
  }
  inline std::vector<u4::irrep> physical_irreps()
  {
    return {{0, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 1},
            {1, 1, 1}, {1, 2, 1}, {2, 1, 0}, {0, 1, 2}, {2, 0, 2},
            {5, 0, 0}, {0, 0, 5}, {0, 5, 0}, {3, 1, 1}, {1, 1, 3}};
  }
  inline u3::irrep sub(const u4::gelfand& g)
  {
    return {g.n13(), g.n23(), g.n33()};
  }
  inline std::string subname(const u3::irrep& g)
  {
    return "[" + std::to_string(g.n13()) + "," + std::to_string(g.n23()) + "," +
           std::to_string(g.n33()) + "]";
  }
  inline std::vector<u3::irrep> subreps(const u4::irrep& ir)
  {
    std::vector<u3::irrep> result;
    for (int a = ir.n14(); a >= ir.n24(); --a)
      for (int b = ir.n24(); b >= ir.n34(); --b)
        for (int c = ir.n34(); c >= ir.n44(); --c)
          result.emplace_back(a, b, c);
    return result;
  }
  // Restore both lower-chain factors, retaining every U(3) rho component.
  inline Eigen::RowVectorXd full_cg(const u4::gelfand& a, const u4::gelfand& b,
                                    const u4::gelfand& c,
                                    const u4::WigU4U3_coeffs& table, int eta)
  {
    Eigen::RowVectorXd result = Eigen::RowVectorXd::Zero(eta);
    auto block = table.find_block(sub(a), sub(b), sub(c));
    if (block.size() == 0)
      return result;
    auto reduced = u3::wigU3U2db().get(
        a.n13(), a.n23(), a.n33(), a.n12(), a.n22(), b.n13(), b.n23(), b.n33(),
        b.n12(), b.n22(), c.n13(), c.n23(), c.n33(), c.n12(), c.n22());
    if (reduced.size() == 0)
      return result;
    double cg =
        utils::cgcs().get({a.Lambdax2(), b.Lambdax2(), c.Lambdax2(),
                               a.MLambdax2(), b.MLambdax2(), c.MLambdax2()});
    for (int e = 0; e < eta; ++e)
      for (int r = 0; r < block.rows(); ++r)
        result[e] += cg * reduced[r] * block(r, e);
    return result;
  }
} // namespace test

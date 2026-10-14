#ifndef U4LIB_IRREP_H
#define U4LIB_IRREP_H

#include <cassert>
#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

#include <convert.h>
#include <readwrite.h>

#include <U4gelfand.h>

#ifdef HAVE_FMT
#include <fmt/format.h>
#endif

namespace su4
{
  class irrep
  {
  protected:
    // type used to represent labels:
    using rep_type = u4::uqn_t; 

    // SU(4) labels:
    rep_type P1_, P2_, P3_;

  public:
    // default constructor; creates (0,0):
    irrep() : P1_(0), P2_(0), P3_(0) { }

    irrep(utils::UInt8 P1, utils::UInt8 P2, utils::UInt8 P3)
        : P1_(P1), P2_(P2), P3_(P3) // conversion check is implicit
    { }

    // comparison:
    auto operator<=>(const irrep&) const = default;

    // SU(4) getters:
    int P1() const { return static_cast<int>(P1_); }
    int P2() const { return static_cast<int>(P2_); }
    int P3() const { return static_cast<int>(P3_); }   

    // U(4) getters:
    int n14() const { return P1() + P2() + P3(); }
    int n24() const { return P2() + P3(); }
    int n34() const { return P3(); }
    int n44() const { return 0; }

    // dimension
    int dim() const
    {
      // return (1 + n14() - n24()) * (2 + n14() - n34()) * (3 + n14() - n44()) *
      //        (1 + n24() - n34()) * (2 + n24() - n44()) * (1 + n34() - n44()) / 12;
      return (1 + P1()) * (2 + P1() + P2()) * (3 + P1() + P2() + P3()) *
             (1 + P2()) * (2 + P2() + P3()) * (1 + P3()) / 12;
    }

    // check for (0,0,0):
    bool zero() const
    {
      return (P1() == 0) && (P2() == 0) && (P3() == 0);
    }

    // second-order Casimir
    double C2() const
    {
      /* O(6) quantum numbers
      double p1 = (n14_ + n24_ - n34_ - n44_) / 2.0,
             p2 = (n14_ - n24_ + n34_ - n44_) / 2.0,
             p3 = (n14_ - n24_ - n34_ + n44_) / 2.0; */

      /*
      double p1 = (P1() + 2 * P2() + P3()) / 2.0,
             p2 = (P1() + P3()) / 2.0,
             p3 = (P1() - P3()) / 2.0;

      return p1 * (p1 + 4.0) + p2 * (p2 + 2.0) + p3 * p3; 
      */

      return 1.5 * P1() * P1() + 2 * P2() * P2() + 1.5 * P3() * P3() +
             2 * P1() * P2() + 2 * P2() * P3() + P1() * P3() +
             6 * P1() + 8 * P2() + 6 * P3();
    }

    // third-order Casimir
    double C3() const
    {
      return 1.5 * (P1() - P3()) * (P1() * P1() + P3() * P3() + 
                                    2 * (P1() * P2() + P2() * P3() + P1() * P3()) +
                                    2 * (3 * P1() + 2 * P2() + 3 * P3() + 4));
    }

    // forth-order Casimir
    double C4() const
    {
      double P1_to2 = P1() * P1(), P1_to3 = P1_to2 * P1(), P1_to4 =  P1_to3 * P1(),
             P2_to2 = P2() * P2(), P2_to3 = P2_to2 * P2(), P2_to4 =  P2_to3 * P2(),
             P3_to2 = P3() * P3(), P3_to3 = P3_to2 * P3(), P3_to4 =  P3_to3 * P3();

      return (21.0 * P1_to4 + 16.0 * P2_to4 + 21.0 * P3_to4 +
              4 * P1_to3 * (14 * P2() + 7 * P3() + 42) +
              4 * P3_to3 * (14 * P2() + 7 * P1() + 42) +
              32 * P2_to3 * (4 + P1() + P3()) + 
              72 * P2_to2 * (P1_to2 + P3_to2) +
              30 * P1_to2 * P3_to2 + 
              24 * P1() * P2() * P3() * (3 * P1() + 2 * P2() + 3 * P3()) +
              288 * P2_to2 * (P1() + P3()) +
              24 * P1_to2 * (16 * P2() + 9 * P3()) + 
              24 * P3_to2 * (16 * P2() + 9 * P1()) +
              384 * P1() * P2() * P3() + 576 * (P1_to2 + P3_to2) + 512 * P2_to2 +
              896 * P2() * (P1() + P3()) + 640 * P1() * P3() +
              960 * (P1() + P3()) + 1024 * P2()) / 64.0;
    }

    // conjugate irrep
    su4::irrep conjugate() const { return su4::irrep(P3(), P2(), P1()); }
    double conjugate_phase_power() const
    {
      // return (3.0 * n14_ + n24_ - n34_ - 3.0 * n44_) / 2.0;
      return (3.0 * P1() + 4.0 * P2() + 3.0 * P3()) / 2.0;
    }

    // write to binary file:
    void writeBinary(std::ostream& s) const
    {
      utils::writeBinary(s, P1_);
      utils::writeBinary(s, P2_);
      utils::writeBinary(s, P3_);
    }

    // read from binary file:
    void readBinary(std::istream& s)
    {
      utils::readBinary(s, P1_);
      utils::readBinary(s, P2_);
      utils::readBinary(s, P3_);
    }
  };
  using irreps = std::vector<su4::irrep>;

  // output stream operator:
  inline std::ostream& operator<<(std::ostream& s, const irrep& su4)
  {
    return s << "(" << su4.P1() << "," << su4.P2() << "," << su4.P3() << ")";
  }
} // namespace su4

namespace u4
{
   // Class that represents U(4) irrep.
  class irrep : public su4::irrep
  {
    rep_type n44_;

  public:
    // default constructor; creates [0,0,0,0]:
    irrep() : n44_(0) { }

    irrep(utils::UInt8 n1, utils::UInt8 n2, 
          utils::UInt8 n3, utils::UInt8 n4)
        : su4::irrep(n1 - n2, n2 - n3, n3 - n4), n44_(n4)
    { 
      assert(n1 >= n2);
      assert(n2 >= n3);
      assert(n3 >= n4);
    }

    irrep(utils::UInt8 P1, utils::UInt8 P2, utils::UInt8 P3) 
      : su4::irrep(P1, P2, P3), n44_(0) { }

    irrep(const su4::irrep& su4) : su4::irrep(su4), n44_(0) { }

    // SU(4)->U(4) conversion with the total number of quanta provided:
    irrep(const su4::irrep& su4, int A) : irrep(su4)
    {
      assert(A >= 0);
      int n44 = A - su4.P1() - 2 * su4.P2() - 3 * su4.P3();
      assert(n44 % 4 == 0);
      n44 /= 4;
      n44_ = n44_ + static_cast<rep_type>(n44);
      assert(quanta() == A);
    }

    // comparison:
    auto operator<=>(const irrep&) const = default;

    // U(4) getters:
    int n44() const { return static_cast<int>(n44_); }
    int n14() const { return su4::irrep::n14() + n44(); }
    int n24() const { return su4::irrep::n24() + n44(); }
    int n34() const { return su4::irrep::n34() + n44(); }
    
    // total quanta
    int quanta() const { return n14() + n24() + n34() + n44(); }
    int C1() const { return quanta(); }

    /* dimension
    int dim() const
    {
      return (1 + n14_ - n24_) * (2 + n14_ - n34_) * (3 + n14_ - n44_) *
             (1 + n24_ - n34_) * (2 + n24_ - n44_) * (1 + n34_ - n44_) / 12;
    }*/

    // normalize the irrep:
    void normalize() { n44_ = 0; }

    // check for [0,0,0,0]:
    bool zero() const { return su4::irrep::zero() && (n44_ == 0); }

    // conjugate irrep
    u4::irrep conjugate() const { return u4::irrep(P3(), P2(), P1()); }

    // write to binary file:
    void writeBinary(std::ostream& s) const
    {
      su4::irrep::writeBinary(s);
      utils::writeBinary(s, n44_);
    }

    // read from binary file:
    void readBinary(std::istream& s)
    {
      su4::irrep::readBinary(s);
      utils::readBinary(s, n44_);
    }
  };
  using irreps = std::vector<u4::irrep>;

  // output stream operator:
  inline std::ostream& operator<<(std::ostream& s, const irrep& u4)
  {
    return s << "[" << u4.n14() << "," << u4.n24() << "," << u4.n34() << ","
             << u4.n44() << "]";
  }

} // namespace u4

#ifdef HAVE_FMT

// formatter for su4::irrep:

template <>
class fmt::formatter<su4::irrep>
{
public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template <typename Context>
  constexpr auto format(const su4::irrep& i, Context& ctx) const
  {
    return fmt::format_to(ctx.out(), "({},{},{})", i.P1(), i.P2(), i.P3());
  }
};

// formatter for u4::irrep:

template <>
class fmt::formatter<u4::irrep>
{
public:
  constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

  template <typename Context>
  constexpr auto format(const u4::irrep& i, Context& ctx) const
  {
    return fmt::format_to(ctx.out(), "[{},{},{},{}]", i.n14(), i.n24(), i.n34(),
                          i.n44());
  }
};

#endif

#endif

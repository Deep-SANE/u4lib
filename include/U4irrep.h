#ifndef U4LIB_IRREP_H
#define U4LIB_IRREP_H

#include <cassert>
#include <cstdint>
#include <istream>
#include <ostream>

#include <qcx_convert.h>
#include <qcx_readwrite.h>

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

    irrep(qcx_utils::UInt8 P1, qcx_utils::UInt8 P2, qcx_utils::UInt8 P3)
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

      double p1 = (P1() + 2 * P2() + P3()) / 2.0,
             p2 = (P1() + P3()) / 2.0,
             p3 = (P1() - P3()) / 2.0;

      return p1 * (p1 + 4.0) + p2 * (p2 + 2.0) + p3 * p3; 
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
      qcx_utils::writeBinary(s, P1_);
      qcx_utils::writeBinary(s, P2_);
      qcx_utils::writeBinary(s, P3_);
    }

    // read from binary file:
    void readBinary(std::istream& s)
    {
      qcx_utils::readBinary(s, P1_);
      qcx_utils::readBinary(s, P2_);
      qcx_utils::readBinary(s, P3_);
    }
  };

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

    irrep(qcx_utils::UInt8 n1, qcx_utils::UInt8 n2, 
          qcx_utils::UInt8 n3, qcx_utils::UInt8 n4)
        : su4::irrep(n1 - n2, n2 - n3, n3 - n4), n44_(n4)
    { 
      assert(n1 >= n2);
      assert(n2 >= n3);
      assert(n3 >= n4);
    }

    irrep(qcx_utils::UInt8 P1, qcx_utils::UInt8 P2, qcx_utils::UInt8 P3) 
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
      qcx_utils::writeBinary(s, n44_);
    }

    // read from binary file:
    void readBinary(std::istream& s)
    {
      su4::irrep::readBinary(s);
      qcx_utils::readBinary(s, n44_);
    }
  };

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

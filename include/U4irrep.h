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

namespace u4
{
   // Class that represents U(4) irrep.
  class irrep 
  {
    using rep_type = uint16_t;

    rep_type n14_, n24_, n34_, n44_;

  public:
    // default constructor; creates [0,0,0,0]:
    irrep() : n14_(0), n24_(0), n34_(0), n44_(0) { }

    irrep(int n1, int n2, int n3, int n4)
        : n14_(qcx_utils::itou16(n1)),
          n24_(qcx_utils::itou16(n2)),
          n34_(qcx_utils::itou16(n3)),
          n44_(qcx_utils::itou16(n4))
    { 
      assert(n14_ >= n24_);
      assert(n24_ >= n34_);
      assert(n34_ >= n44_);
    }

    // construction from Draayer's SU(4) (P1,P2,P3):
    irrep(int P1, int P2, int P3) : irrep(P1 + P2 + P3, P2 + P3, P3, 0) { }

    // comparison:
    auto operator<=>(const irrep&) const = default;

    // U(4) getters:

    int n14() const { return n14_; }
    int n24() const { return n24_; }
    int n34() const { return n34_; }
    int n44() const { return n44_; }

    // SU(4) getters:

    int P1() const { return n14_ - n24_; }
    int P2() const { return n24_ - n34_; }
    int P3() const { return n34_ - n44_; }

    // total quanta
    int quanta() const { return n14_ + n24_ + n34_ + n44_; }

    // dimension
    int dim() const
    {
      return (1 + n14_ - n24_) * (2 + n14_ - n34_) * (3 + n14_ - n44_) *
             (1 + n24_ - n34_) * (2 + n24_ - n44_) * (1 + n34_ - n44_) / 12;
    }

    // normalize the irrep:
    void normalize()
    {
      n14_ -= n44_;
      n24_ -= n44_;
      n34_ -= n44_;
      n44_ = 0;
    }

    // check for [0,0,0,0]:
    bool zero() const
    {
      return (n14_ == 0) && (n24_ == 0) && (n34_ == 0) && (n44_ == 0);
    }

    // write to binary file:
    void writeBinary(std::ostream& s) const
    {
      qcx_utils::writeBinary(s, n14_);
      qcx_utils::writeBinary(s, n24_);
      qcx_utils::writeBinary(s, n34_);
      qcx_utils::writeBinary(s, n44_);
    }

    // read from binary file:
    void readBinary(std::istream& s)
    {
      qcx_utils::readBinary(s, n14_);
      qcx_utils::readBinary(s, n24_);
      qcx_utils::readBinary(s, n34_);
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

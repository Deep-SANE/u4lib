#ifndef U4LIB_INTERNAL_IRREP_UINT16_T_REP_H
#define U4LIB_INTERNAL_IRREP_UINT16_T_REP_H

#include <cassert>
#include <cstdint>
#include <limits>
#include <tuple>

namespace u4
{
  namespace internal
  {
    // helper function for safe-asserted conversion from `int` to `uint16_t`:
    inline uint16_t itou16(int i)
    {
      static const int max =
          static_cast<int>(std::numeric_limits<uint16_t>::max());

      assert(i <= max);

      return static_cast<uint16_t>(i);
    }

    class irrep_uint16_t_rep
    {
      uint16_t n14_, n24_, n34_, n44_;

      // helper function for comparison operators:
      auto tie() const { return std::tie(n14_, n24_, n34_, n44_); }

    public:
      irrep_uint16_t_rep() : n14_(0), n24_(0), n34_(0), n44_(0) { }

      irrep_uint16_t_rep(int n14, int n24, int n34, int n44)
          : n14_(itou16(n14)),
            n24_(itou16(n24)),
            n34_(itou16(n34)),
            n44_(itou16(n44))
      { }

      int n14() const { return static_cast<int>(n14_); }
      int n24() const { return static_cast<int>(n24_); }
      int n34() const { return static_cast<int>(n34_); }
      int n44() const { return static_cast<int>(n44_); }

      void n14(int n14) { n14_ = itou16(n14); }
      void n24(int n24) { n24_ = itou16(n24); }
      void n34(int n34) { n34_ = itou16(n34); }
      void n44(int n44) { n44_ = itou16(n44); }

      bool operator==(const irrep_uint16_t_rep& other) const
      {
        return tie() == other.tie();
      }

      bool operator!=(const irrep_uint16_t_rep& other) const
      {
        return !(*this == other);
      }
    };

  } // namespace internal
} // namespace u4

#endif

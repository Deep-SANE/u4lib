#ifndef U4LIB_INTERNAL_IRREP_INT_REP_H
#define U4LIB_INTERNAL_IRREP_INT_REP_H

#include <tuple>

namespace u4
{
  namespace internal
  {
    class irrep_int_rep
    {
      int n14_, n24_, n34_, n44_;

      // helper function for comparison operators:
      auto tie() const { return std::tie(n14_, n24_, n34_, n44_); }

    public:
      irrep_int_rep() : n14_(0), n24_(0), n34_(0), n44_(0) { }

      irrep_int_rep(int n14, int n24, int n34, int n44) : n14_(n14), n24_(n24), n34_(n34), n44_(n44)
      { }

      int n14() const { return n14_; }
      int n24() const { return n24_; }
      int n34() const { return n34_; }
      int n44() const { return n44_; }

      void n14(int n14) { n14_ = n14; }
      void n24(int n24) { n24_ = n24; }
      void n34(int n34) { n34_ = n34; }
      void n44(int n44) { n44_ = n44; }

      bool operator==(const irrep_int_rep& other) const
      {
        return tie() == other.tie();
      }

      bool operator!=(const irrep_int_rep& other) const
      {
        return !(*this == other);
      }
    };

  } // namespace internal
} // namespace u4

#endif

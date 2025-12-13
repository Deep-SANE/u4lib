#ifndef U4LIB_INTERNAL_GELFAND_INT_REP_H
#define U4LIB_INTERNAL_GELFAND_INT_REP_H

#include <tuple>

#include <qcx_Int.h>

namespace u4
{
  // Define alias datatypes
  using uqn_t = qcx_utils::UInt8;  // unsigned quantum numbers: 0 to 255
  using qn_t = qcx_utils::Int8;    // signed quantum numbers: -128 to 127
  using mult_t = qcx_utils::UInt8; // multiplicities: 0 to 255

  namespace internal
  {
    class gelfand_int_rep
    {
      uqn_t n14_, n24_, n34_, n44_, n13_, n23_, n33_, n12_, n22_, n11_;

      // helper function for comparison operators:
      auto tie() const
      {
        return std::tie(n14_, n24_, n34_, n44_, n13_, n23_, n33_, n12_, n22_,
                        n11_);
      }

    public:
      gelfand_int_rep()
          : n14_(0),
            n24_(0),
            n34_(0),
            n44_(0),
            n13_(0),
            n23_(0),
            n33_(0),
            n12_(0),
            n22_(0),
            n11_(0)
      { }

      gelfand_int_rep(uqn_t n14, uqn_t n24, uqn_t n34, uqn_t n44, uqn_t n13, 
                      uqn_t n23, uqn_t n33, uqn_t n12, uqn_t n22, uqn_t n11)
          : n14_(n14),
            n24_(n24),
            n34_(n34),
            n44_(n44),
            n13_(n13),
            n23_(n23),
            n33_(n33),
            n12_(n12),
            n22_(n22),
            n11_(n11)
      { }

      int n14() const { return static_cast<int>(n14_); }
      int n24() const { return static_cast<int>(n24_); }
      int n34() const { return static_cast<int>(n34_); }
      int n44() const { return static_cast<int>(n44_); }
      int n13() const { return static_cast<int>(n13_); }
      int n23() const { return static_cast<int>(n23_); }
      int n33() const { return static_cast<int>(n33_); }
      int n12() const { return static_cast<int>(n12_); }
      int n22() const { return static_cast<int>(n22_); }
      int n11() const { return static_cast<int>(n11_); }

      void n14(uqn_t n14) { n14_ = n14; }
      void n24(uqn_t n24) { n24_ = n24; }
      void n34(uqn_t n34) { n34_ = n34; }
      void n44(uqn_t n44) { n44_ = n44; }
      void n13(uqn_t n13) { n13_ = n13; }
      void n23(uqn_t n23) { n23_ = n23; }
      void n33(uqn_t n33) { n33_ = n33; }
      void n12(uqn_t n12) { n12_ = n12; }
      void n22(uqn_t n22) { n22_ = n22; }
      void n11(uqn_t n11) { n11_ = n11; }

      bool operator==(const gelfand_int_rep& other) const
      {
        return tie() == other.tie();
      }

      bool operator!=(const gelfand_int_rep& other) const
      {
        return !(*this == other);
      }
    };

  } // namespace internal
} // namespace u4

#endif

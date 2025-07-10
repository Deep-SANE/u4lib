#ifndef U4LIB_INTERNAL_IRREP_GELFAND_H
#define U4LIB_INTERNAL_IRREP_GELFAND_H

#include <ostream>

#include "int_rep.h"
#include "uint16_t_rep.h"

namespace u4
{
  namespace internal
  {
    /*
     * A class for U(4) irrep [n14,n24,n34,n44]
     *
     * The template argument specifies the internal representation.
     */
    template <typename Rep>
    class irrep : Rep
    {
    public:
      using Rep::n14, Rep::n24, Rep::n34, Rep::n44;

      irrep() = default;

      irrep(int n14, int n24, int n34, int n44) : Rep(n14, n24, n34, n44) { }

      // construction from Draayer's SU(4) (P1,P2,P3)
      irrep(int P1, int P2, int P3) : Rep(P1 + P2 + P3, P2 + P3, P3, 0) { }

      // comparison operators:
      bool operator==(const irrep& other) const
      {
        return static_cast<const Rep&>(*this) == static_cast<const Rep&>(other);
      }

      bool operator!=(const irrep& other) const { return !(*this == other); }

      // P1, P2, P3
      int P1() const { return n14() - n24(); }
      int P2() const { return n24() - n34(); }
      int P3() const { return n34() - n44(); }

      // total quanta
      int quanta() const { return n14() + n24() + n34() + n44(); }

      // dimension
      int dim() const
      {
        return (1 + n14() - n24()) * (2 + n14() - n34()) * (3 + n14() - n44()) *
               (1 + n24() - n34()) * (2 + n24() - n44()) * (1 + n34() - n44()) / 12;
      }

      // check order of quantum numbers
      bool check() { return (n14() >= n24() && n24() >= n34() && n34() >= n44()); }

      // normalize the irrep
      void normalize()
      {
        n14(n14() - n44());
        n24(n24() - n44());
        n34(n34() - n44());
        n44(0);
      }
    };

    // print u4irrep
    template <typename Rep>
    inline std::ostream& operator<<(std::ostream& s, const irrep<Rep>& u4)
    {
      return s << "[" << u4.n14() << "," << u4.n24() << "," << u4.n34() << "," << u4.n44() << "]";
    }

    // type aliases:
    using irrep_int_rep_t = irrep<irrep_int_rep>;
    using irrep_uint16_t_rep_t = irrep<irrep_uint16_t_rep>;

  } // namespace internal
} // namespace u4

#endif

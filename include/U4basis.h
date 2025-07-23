// protection against multiple header file inclusion
#ifndef U4LIB_U4BASIS_H
#define U4LIB_U4BASIS_H

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "U4gelfand.h"
#include "U4irrep.h"

namespace u4
{
  // compare two gelfand patterns of the same irrep
  inline bool E11(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && i.n23() == f.n23() && i.n33() == f.n33() &&
           i.n12() == f.n12() && i.n22() == f.n22() && i.n11() == f.n11();
  }

  // define action of the raising generators, that of the lowering generators
  // is simply swapping the initial and final states
  inline bool E12(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && i.n23() == f.n23() && i.n33() == f.n33() &&
           i.n12() == f.n12() && i.n22() == f.n22() && (i.n11() + 1) == f.n11();
  }

  inline bool E23_12(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && i.n23() == f.n23() && i.n33() == f.n33() &&
           (i.n12() + 1) == f.n12() && i.n22() == f.n22() && i.n11() == f.n11();
  }

  inline bool E23_22(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && i.n23() == f.n23() && i.n33() == f.n33() &&
           i.n12() == f.n12() && (i.n22() + 1) == f.n22() && i.n11() == f.n11();
  }

  inline bool E13_12(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && i.n23() == f.n23() && i.n33() == f.n33() &&
           (i.n12() + 1) == f.n12() && i.n22() == f.n22() &&
           (i.n11() + 1) == f.n11();
  }

  inline bool E13_22(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && i.n23() == f.n23() && i.n33() == f.n33() &&
           i.n12() == f.n12() && (i.n22() + 1) == f.n22() &&
           (i.n11() + 1) == f.n11();
  }

  inline bool E34_13(const u4::gelfand& i, const u4::gelfand& f)
  {
    return (i.n13() + 1) == f.n13() && i.n23() == f.n23() && i.n33() == f.n33() 
           && i.n12() == f.n12() && i.n22() == f.n22() && i.n11() == f.n11();
  }

  inline bool E34_23(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && (i.n23() + 1) == f.n23() && i.n33() == f.n33() 
           && i.n12() == f.n12() && i.n22() == f.n22() && i.n11() == f.n11();
  }

  inline bool E34_33(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && i.n23() == f.n23() && (i.n33() + 1) == f.n33() 
           && i.n12() == f.n12() && i.n22() == f.n22() && i.n11() == f.n11();
  }

  inline bool E24_13_12(const u4::gelfand& i, const u4::gelfand& f)
  {
    return (i.n13() + 1) == f.n13() && i.n23() == f.n23() && i.n33() == f.n33() 
           && (i.n12() + 1) == f.n12() && i.n22() == f.n22() && i.n11() == f.n11();
  }

  inline bool E24_13_22(const u4::gelfand& i, const u4::gelfand& f)
  {
    return (i.n13() + 1) == f.n13() && i.n23() == f.n23() && i.n33() == f.n33() 
           && i.n12() == f.n12() && (i.n22() + 1) == f.n22() && i.n11() == f.n11();
  }

  inline bool E24_23_12(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && (i.n23() + 1) == f.n23() && i.n33() == f.n33() 
           && (i.n12() + 1) == f.n12() && i.n22() == f.n22() && i.n11() == f.n11();
  }

  inline bool E24_23_22(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && (i.n23() + 1) == f.n23() && i.n33() == f.n33() 
           && i.n12() == f.n12() && (i.n22() + 1) == f.n22() && i.n11() == f.n11();
  }

  inline bool E24_33_12(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && i.n23() == f.n23() && (i.n33() + 1) == f.n33() 
           && (i.n12() + 1) == f.n12() && i.n22() == f.n22() && i.n11() == f.n11();
  }

  inline bool E24_33_22(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && i.n23() == f.n23() && (i.n33() + 1) == f.n33() 
           && i.n12() == f.n12() && (i.n22() + 1) == f.n22() && i.n11() == f.n11();
  }

  inline bool E14_13_12(const u4::gelfand& i, const u4::gelfand& f)
  {
    return (i.n13() + 1) == f.n13() && i.n23() == f.n23() && i.n33() == f.n33() 
           && (i.n12() + 1) == f.n12() && i.n22() == f.n22() 
           && (i.n11() + 1) == f.n11();
  }

  inline bool E14_13_22(const u4::gelfand& i, const u4::gelfand& f)
  {
    return (i.n13() + 1) == f.n13() && i.n23() == f.n23() && i.n33() == f.n33() 
           && i.n12() == f.n12() && (i.n22() + 1) == f.n22() 
           && (i.n11() + 1) == f.n11();
  }

  inline bool E14_23_12(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && (i.n23() + 1) == f.n23() && i.n33() == f.n33() 
           && (i.n12() + 1) == f.n12() && i.n22() == f.n22() 
           && (i.n11() + 1) == f.n11();
  }

  inline bool E14_23_22(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && (i.n23() + 1) == f.n23() && i.n33() == f.n33() 
           && i.n12() == f.n12() && (i.n22() + 1) == f.n22() 
           && (i.n11() + 1) == f.n11();
  }

  inline bool E14_33_12(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && i.n23() == f.n23() && (i.n33() + 1) == f.n33() 
           && (i.n12() + 1) == f.n12() && i.n22() == f.n22() 
           && (i.n11() + 1) == f.n11();
  }

  inline bool E14_33_22(const u4::gelfand& i, const u4::gelfand& f)
  {
    return i.n13() == f.n13() && i.n23() == f.n23() && (i.n33() + 1) == f.n33() 
           && i.n12() == f.n12() && (i.n22() + 1) == f.n22() 
           && (i.n11() + 1) == f.n11();
  }

  // find all Gelfand states associated with an irrep
  inline std::vector<u4::gelfand> gelfandBasis(const u4::irrep& x)
  {
    std::vector<u4::gelfand> gArr;

    for (int n13 = x.n14(); n13 >= x.n24(); n13--)
      for (int n23 = x.n24(); n23 >= x.n34(); n23--)
        for (int n33 = x.n34(); n33 >= x.n44(); n33--)
          for (int n12 = n13; n12 >= n23; n12--)
            for (int n22 = n23; n22 >= n33; n22--)
              for (int n11 = n12; n11 >= n22; n11--)
                gArr.emplace_back(x.n14(), x.n24(), x.n34(), x.n44(), n13, n23, n33, n12, n22, n11);

    return gArr;
  }

  // find the location (index) of a Gelfand state within the Gelfand basis above
  inline std::size_t findGelfand(const u4::gelfand& g,
                          const std::vector<u4::gelfand>& gArr)
  {
    std::size_t i;

    for (i = 0; i < gArr.size(); i++)
      if (u4::E11(g, gArr[i]))
        break;

    return i;
  }
  

  /*------------------------------------------------------
  The highest weight physical basis has the following form
  | n14,n24,n34,n44  \
  |     zeta S T     /
  where
  gamma_min <= S,T <= p1 = (n14 + n24 - n34 - n44)/2
  gamma_min <= S+T <= p1+p2 = n14 - n44
  Note that we store 2S and 2T as SS and TT, resspectively, to ensure integer
  type
  */

  /*This part of the code compute the multiplicity of (S,T) in the reduction of
  U(4) irrep using Racah's formula in Rev. Mod. Phys. in 1949 For an U(4) irrep
  [n14,n24,n34,n44], the multiplicity of (S,T) is computed as follows:
  multi(S,T) = Omega(n14-n34,n24-n44) - Omega(n14-n44+1,n24-n34-1) -
  Omega(n14-n24-1,n34-n44-1) where the function Omega(x,y) vanishes unless S,T
  <= (x+y)/2. When Omega(x,y) is not vanishing and x >= y, Omega(x,y) =
  Omega(y,x) = phi(y+2-|T-S|) - phi(y+1-T-S) + phi(T+S-x-1) -
  phi(T+S-|T-S|-x+y+1)/2 where the phi function is defined as follows phi(x) =
  [x^2/4], x > 0 = 0, otherwise
  */
  // phi function
  inline int phi(int x)
  {
    if (x > 0)
      return x * x / 4;
    else
      return 0;
  }

  // Omega function
  inline int Omega(int SS, int TT, int x, int y)
  {
    if (x >= y && TT <= (x + y) && SS <= (x + y))
    {
      return phi(y + 2 - abs(TT - SS) / 2) - phi(y + 1 - (TT + SS) / 2) +
             phi((TT + SS) / 2 - x - 1) -
             phi((TT + SS) / 2 - abs(TT - SS) / 2 - x + y + 1) / 2;
    }
    else if (x < y && TT <= (x + y) && SS <= (x + y))
    {
      return Omega(SS, TT, y, x);
    }
    else
    {
      return 0;
    }
  }

  // Multiplicity of (S,T)
  inline int Zeta(const u4::irrep& u4, int SS, int TT)
  {
    return Omega(SS, TT, u4.n14() - u4.n34(), u4.n24() - u4.n44()) -
           Omega(SS, TT, u4.n14() - u4.n44() + 1, u4.n24() - u4.n34() - 1) -
           Omega(SS, TT, u4.n14() - u4.n24() - 1, u4.n34() - u4.n44() - 1);
  }

  // alternative called 'mult':
  inline int mult(const u4::irrep& u4, int SS, int TT)
  {
    return u4::Zeta(u4, SS, TT);
  }

  struct physicalvector
  {
    u4::irrep toprow;
    int SS, TT;
    int zeta;

    // Default constructor
    physicalvector() : zeta(0), SS(0), TT(0) { }

    // initialization
    physicalvector(u4::irrep u4, int ss, int tt)
    {
      toprow = u4;
      zeta = Zeta(u4, ss, tt);
      SS = ss;
      TT = tt;
    }

    // assignment operator
    physicalvector& operator=(const physicalvector& physvec)
    {
      toprow = physvec.toprow;
      zeta = physvec.zeta;
      SS = physvec.SS;
      TT = physvec.TT;
      return *this;
    }
  };

  // print out a physical basis vector
  inline std::ostream& operator<<(std::ostream& s, physicalvector& physvec)
  {
    return s << "(" << physvec.toprow.P1() << "," << physvec.toprow.P2() << ","
             << physvec.toprow.P3() << "), zeta=" << physvec.zeta
             << ", SS=" << physvec.SS << ", TT=" << physvec.TT;
  }

  // find all physical states associated with an irrep
  inline std::vector<u4::physicalvector> physicalBasis(const u4::irrep& u4)
  {
    std::vector<u4::physicalvector> basis;

    int ss, tt, sum; // sum = ss + tt

    // Upper limit for S,T
    int ss_max = u4.n14() + u4.n24() - u4.n34() - u4.n44(); // it is also tt_max
    int sum_max = 2 * (u4.n14() - u4.n44());

    // Lower limit for S,T
    int ss_min = ((u4.n14() + u4.n24() - u4.n34() - u4.n44()) % 2 == 0) ? 0 : 1;

    for (ss = ss_max; ss >= ss_min; ss -= 2)
    {
      for (tt = ss_max; tt >= ss_min; tt -= 2)
      {
        if ((ss + tt) >= ss_min && (ss + tt) <= sum_max)
        {
          u4::physicalvector physvec(u4, ss, tt);
          if (physvec.zeta == 0)
            continue;
          else
            basis.push_back(physvec);
        }
      }
    }
    return basis;
  }


} // namespace u4

#endif

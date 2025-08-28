#ifndef U4LIB_DECOMPOSITION_H
#define U4LIB_DECOMPOSITION_H

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

#include "U4basis.h"

namespace u4
{

  inline std::pair<std::vector<u4::irrep>, std::vector<int>>
  decomposition(const u4::irrep& u41, const u4::irrep& u42)
  {
    std::vector<u4::irrep> u4cs;
    std::vector<int> etamaxs;

    u4::irrep u4a, u4b;

    if (u41.dim() <= u42.dim())
    {
      u4a = u41;
      u4b = u42;
    }
    else
    {
      u4a = u42;
      u4b = u41;
    }

    for (int n13 = u4a.n14(); n13 >= u4a.n24(); n13--)
    {
      for (int n23 = u4a.n24(); n23 >= u4a.n34(); n23--)
      {
        for (int n33 = u4a.n34(); n33 >= u4a.n44(); n33--)
        {
          for (int n12 = n13; n12 >= n23; n12--)
          {
            for (int n22 = n23; n22 >= n33; n22--)
            {
              for (int n11 = n12; n11 >= n22; n11--)
              {
                int b14 = u4a.n14() - n13, b24 = u4a.n24() - n23,
                    b34 = u4a.n34() - n33, b44 = u4a.n44(), b13 = n13 - n12,
                    b23 = n23 - n22, b33 = n33, b12 = n12 - n11, b22 = n22,
                    b11 = n11;

                assert(b14 >= 0);
                assert(b24 >= 0);
                assert(b34 >= 0);
                assert(b44 >= 0);
                assert(b13 >= 0);
                assert(b23 >= 0);
                assert(b33 >= 0);
                assert(b12 >= 0);
                assert(b22 >= 0);
                assert(b11 >= 0);

                // u4::irrep u4c = u4b;
                int n14 = u4b.n14();
                int n24 = u4b.n24();
                int n34 = u4b.n34();
                int n44 = u4b.n44();

                // u4c.n44(u4c.n44() + b14);
                n44 += b14;
                if (n44 > n34)
                  continue;

                // u4c.n34(u4c.n34() + b13);
                n34 += b13;
                if (n34 > n24)
                  continue;

                // u4c.n24(u4c.n24() + b12);
                n24 += b12;
                if (n24 > n14)
                  continue;

                // u4c.n14(u4c.n14() + b11);
                n14 += b11;

                // u4c.n44(u4c.n44() + b24);
                n44 += b24;
                if (n44 > n34)
                  continue;

                // u4c.n34(u4c.n34() + b23);
                n34 += b23;
                if (n34 > n24)
                  continue;

                // u4c.n24(u4c.n24() + b22);
                n24 += b22;
                if (n24 > n14)
                  continue;

                // u4c.n44(u4c.n44() + b34);
                n44 += b34;
                if (n44 > n34)
                  continue;

                // u4c.n34(u4c.n34() + b33);
                n34 += b33;
                if (n34 > n24)
                  continue;

                // u4c.n44(u4c.n44() + b44);
                n44 += b44;
                if (n44 > n34)
                  continue;

                u4::irrep u4c(n14, n24, n34, n44);
                u4c.normalize();

                auto iter = std::find(u4cs.begin(), u4cs.end(), u4c);
                if (iter == u4cs.end())
                {
                  u4cs.push_back(u4c);
                  etamaxs.push_back(1);
                }
                else
                {
                  auto index = std::distance(u4cs.begin(), iter);
                  etamaxs[index]++;
                }
              }
            }
          }
        }
      }
    }

    return std::make_pair(std::move(u4cs), std::move(etamaxs));
  }

  // compute the outer multiplicity directly
  // input irreps are normalized
  inline int Eta(const u4::irrep& u4a, const u4::irrep& u4b,
                 const u4::irrep& u4c)
  {
    //std::cout << u4a << u4b << u4c << std::endl;
    int x = u4a.quanta() + u4b.quanta() - u4c.quanta();

    if(x % 4 != 0)
      return 0;
    
    x /= 4;

    if ((u4c.n44() + x) < 0)
      x = - (u4c.n44() + x);
    /*
    u4::irrep u4c_temp = u4c;
    u4c_temp.n14(u4c_temp.n14() + x);
    u4c_temp.n24(u4c_temp.n24() + x);
    u4c_temp.n34(u4c_temp.n34() + x);
    u4c_temp.n44(u4c_temp.n44() + x);
    */
    u4::irrep u4c_temp(u4c.n14() + x, u4c.n24() + x, u4c.n34() + x,
                       u4c.n44() + x);

    int eta = 0, n11 = u4c_temp.n14() - u4b.n14();

    for (int n13 = u4a.n14(); n13 >= u4a.n24(); n13--)
    {
      for (int n23 = u4a.n24(); n23 >= u4a.n34(); n23--)
      {
        for (int n33 = u4a.n34(); n33 >= u4a.n44(); n33--)
        {
          for (int n12 = n13; n12 >= std::max(n23, n11); n12--)
          {
            for (int n22 = std::min(n23, n11); n22 >= n33; n22--)
            {
              int b14 = u4a.n14() - n13, b24 = u4a.n24() - n23,
                  b34 = u4a.n34() - n33, b44 = u4a.n44(), b13 = n13 - n12,
                  b23 = n23 - n22, b33 = n33, b12 = n12 - n11, b22 = n22,
                  b11 = n11;

              assert(b14 >= 0);
              assert(b24 >= 0);
              assert(b34 >= 0);
              assert(b44 >= 0);
              assert(b13 >= 0);
              assert(b23 >= 0);
              assert(b33 >= 0);
              assert(b12 >= 0);
              assert(b22 >= 0);
              assert(b11 >= 0);

              // u4::irrep u4C = u4b;
              int n14 = u4b.n14();
              int n24 = u4b.n24();
              int n34 = u4b.n34();
              int n44 = u4b.n44();

              // u4C.n44(u4C.n44() + b14);
              n44 += b14;
              if (n44 > n34)
                continue;

              // u4C.n34(u4C.n34() + b13);
              n34 += b13;
              if (n34 > n24)
                continue;

              // u4C.n24(u4C.n24() + b12);
              n24 += b12;
              if (n24 > n14)
                continue;

              // u4C.n14(u4C.n14() + b11);
              n14 += b11;

              // u4C.n44(u4C.n44() + b24);
              n44 += b24;
              if (n44 > n34)
                continue;

              // u4C.n34(u4C.n34() + b23);
              n34 += b23;
              if (n34 > n24)
                continue;

              // u4C.n24(u4C.n24() + b22);
              n24 += b22;
              if (n24 > n14)
                continue;

              // u4C.n44(u4C.n44() + b34);
              n44 += b34;
              if (n44 > n34)
                continue;

              // u4C.n34(u4C.n34() + b33);
              n34 += b33;
              if (n34 > n24)
                continue;

              // u4C.n44(u4C.n44() + b44);
              n44 += b44;
              if (n44 > n34)
                continue;

              u4::irrep u4C(n14, n24, n34, n44);

              if (u4C == u4c_temp)
                eta++;
            }
          }
        }
      }
    }

    return eta;
  }

  // alternative called 'mult':
  inline int mult(const u4::irrep& u4a, const u4::irrep& u4b,
                  const u4::irrep& u4c)
  {
    return u4::Eta(u4a, u4b, u4c);
  }
} // namespace u4

#endif

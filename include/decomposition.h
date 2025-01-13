// protection against multiple header file inclusion
#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

#include <cmath>
#include <iostream>
#include <vector>

#include "U4basis.h"

namespace u4
{

  std::vector<u4::irrep> decomposition(u4::irrep& u41, u4::irrep& u42,
                                       std::vector<int>& etamaxs)
  {

    std::vector<u4::irrep> u4cs;

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

    for (int n13 = u4a.n14; n13 >= u4a.n24; n13--)
    {
      for (int n23 = u4a.n24; n23 >= u4a.n34; n23--)
      {
        for (int n33 = u4a.n34; n33 >= u4a.n44; n33--)
        {
          for (int n12 = n13; n12 >= n23; n12--)
          {
            for (int n22 = n23; n22 >= n33; n22--)
            {
              for (int n11 = n12; n11 >= n22; n11--)
              {

                int b14 = u4a.n14 - n13, b24 = u4a.n24 - n23,
                    b34 = u4a.n34 - n33, b44 = u4a.n44, b13 = n13 - n12,
                    b23 = n23 - n22, b33 = n33, b12 = n12 - n11, b22 = n22,
                    b11 = n11;

                u4::irrep u4c = u4b;

                u4c.n44 += b14;
                if (!u4c.check())
                  continue;

                u4c.n34 += b13;
                if (!u4c.check())
                  continue;

                u4c.n24 += b12;
                if (!u4c.check())
                  continue;

                u4c.n14 += b11;
                if (!u4c.check())
                  continue;

                u4c.n44 += b24;
                if (!u4c.check())
                  continue;

                u4c.n34 += b23;
                if (!u4c.check())
                  continue;

                u4c.n24 += b22;
                if (!u4c.check())
                  continue;

                u4c.n44 += b34;
                if (!u4c.check())
                  continue;

                u4c.n34 += b33;
                if (!u4c.check())
                  continue;

                u4c.n44 += b44;
                if (!u4c.check())
                  continue;

                if (u4cs.size() == 0)
                {
                  u4cs.push_back(u4c);
                  etamaxs.push_back(1);
                }
                else
                {
                  int flag = 0;
                  for (int i = 0; i < u4cs.size(); i++)
                  {
                    if (u4c.n14 == u4cs[i].n14 && u4c.n24 == u4cs[i].n24 &&
                        u4c.n34 == u4cs[i].n34 && u4c.n44 == u4cs[i].n44)
                    {
                      etamaxs[i]++;
                      flag = 1;
                    }
                  }

                  if (flag == 0)
                  {
                    u4cs.push_back(u4c);
                    etamaxs.push_back(1);
                  }
                }
              }
            }
          }
        }
      }
    }

    return u4cs;
  }

} // namespace u4

#endif
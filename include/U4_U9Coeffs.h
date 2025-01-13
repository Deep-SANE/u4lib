// protection against multiple header file inclusion
#ifndef U4_U9COEFFS_H
#define U4_U9COEFFS_H

#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "decomposition.h"
#include "U4_U6Z6Coeffs.h"
#include "U4basis.h"

namespace u4
{

  // input by users: u4a, u4b, u4c, u4d, u4ab, u4cd, u4ac, u4bd, u4abcd
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
  U9(u4::irrep& u4a, u4::irrep& u4b, u4::irrep& u4c, u4::irrep& u4d,
     u4::irrep& u4ab, u4::irrep& u4cd, u4::irrep& u4ac, u4::irrep& u4bd,
     u4::irrep& u4abcd, std::vector<int>& etaab, int etaab_max,
     std::vector<int>& etacd, int etacd_max, std::vector<int>& etaab_cd,
     int etaab_cd_max, std::vector<int>& etaac, int etaac_max,
     std::vector<int>& etabd, int etabd_max, std::vector<int>& etaac_bd,
     int etaac_bd_max)
  {
    // a matrix to store U9-coefficients
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> U9coeffs(
        etaac_max * etabd_max * etaab_cd_max,
        etaab_max * etacd_max * etaac_bd_max);

    // std::cout << etaac_max << etabd_max << etaab_cd_max << etaab_max <<
    // etacd_max << etaac_bd_max;

    // couple the I-III and II irreps
    std::vector<int> etamaxsac_b;
    std::vector<u4::irrep> u4ac_b = u4::decomposition(u4ac, u4b, etamaxsac_b);

    // couple the I-II and III irreps
    std::vector<int> etamaxsab_c;
    std::vector<u4::irrep> u4ab_c = u4::decomposition(u4ab, u4c, etamaxsab_c);

    // an array to store the couple I-II-III irreps
    std::vector<int> etamaxsabc;
    std::vector<u4::irrep> u4abc;

    for (int i = 0; i < u4ab_c.size(); i++)
    {
      for (int j = 0; j < u4ac_b.size(); j++)
      {
        if (u4ab_c[i].n14 == u4ac_b[j].n14 && u4ab_c[i].n24 == u4ac_b[j].n24 &&
            u4ab_c[i].n34 == u4ac_b[j].n34 && u4ab_c[i].n44 == u4ac_b[j].n44)
        {
          u4abc.push_back(u4ab_c[i]);
          etamaxsabc.push_back(etamaxsab_c[i]);
          break;
        }
      }
    }

    std::vector<std::vector<int>> etaab_arr, etacd_arr, etaac_arr, etabd_arr,
        etaab_cd_arr, etaac_bd_arr, etaac_b_uarr, etaac_b_zarr, etaab_c_zarr,
        etaab_c_uarr, etaabc_d_arr1, etaabc_d_arr2;
    std::vector<std::vector<double>> U6coeffs_arr1, Z6coeffs_arr, U6coeffs_arr2;

    for (int i = 0; i < u4abc.size(); i++)
    {
      // std::cout << u4abc[i] << std::endl;

      std::vector<int> eab, ecd, eac, ebd, eab_cd, eac_bd, eac_b_u, eac_b_z,
          eab_c_z, eab_c_u, eabc_d_1, eabc_d_2;
      std::vector<double> U6coeffs_1, Z6coeffs, U6coeffs_2;

      U6coeffs_1 = u4::U6(u4ac, u4b, u4abcd, u4d, u4abc[i], eac_b_u, eabc_d_1,
                          u4bd, ebd, eac_bd);
      Z6coeffs = u4::Z6(u4b, u4a, u4abc[i], u4c, u4ab, eab, eab_c_z, u4ac, eac,
                        eac_b_z);
      U6coeffs_2 = u4::U6(u4ab, u4c, u4abcd, u4d, u4abc[i], eab_c_u, eabc_d_2,
                          u4cd, ecd, eab_cd);

      etaab_arr.push_back(eab);
      etacd_arr.push_back(ecd);
      etaac_arr.push_back(eac);
      etabd_arr.push_back(ebd);
      etaab_cd_arr.push_back(eab_cd);
      etaac_bd_arr.push_back(eac_bd);
      etaac_b_uarr.push_back(eac_b_u);
      etaac_b_zarr.push_back(eac_b_z);
      etaab_c_zarr.push_back(eab_c_z);
      etaab_c_uarr.push_back(eab_c_u);
      etaabc_d_arr1.push_back(eabc_d_1);
      etaabc_d_arr2.push_back(eabc_d_2);

      U6coeffs_arr1.push_back(U6coeffs_1);
      U6coeffs_arr2.push_back(U6coeffs_2);
      Z6coeffs_arr.push_back(Z6coeffs);

      /*
                      for(int j=0; j<U6coeffs_1.size(); j++){
                              std::cout << U6coeffs_1[j] << "\t";
                      }
                      std::cout << std::endl;
                      for(int j=0; j<Z6coeffs.size(); j++){
                              std::cout << Z6coeffs[j] << "\t";
                      }
                      std::cout << std::endl;
                      for(int j=0; j<U6coeffs_2.size(); j++){
                              std::cout << U6coeffs_2[j] << "\t";
                      }
                      std::cout << std::endl;
      */
    }

    int row = 0, col = 0;
    for (int eac = 0; eac < etaac_max; eac++)
    {
      for (int ebd = 0; ebd < etabd_max; ebd++)
      {
        for (int eab_cd = 0; eab_cd < etaab_cd_max; eab_cd++)
        {
          etaac.push_back(eac + 1);
          etabd.push_back(ebd + 1);
          etaab_cd.push_back(eab_cd + 1);
        }
      }
    }

    for (int eab = 0; eab < etaab_max; eab++)
    {
      for (int ecd = 0; ecd < etacd_max; ecd++)
      {
        for (int eac_bd = 0; eac_bd < etaac_bd_max; eac_bd++)
        {
          etaab.push_back(eab + 1);
          etacd.push_back(ecd + 1);
          etaac_bd.push_back(eac_bd + 1);
        }
      }
    }

    for (int m = 0; m < etaab_cd.size(); m++)
    { // m+1 is the multiplicity label for etaac, etabd, etaab_cd
      for (int n = 0; n < etaac_bd.size(); n++)
      { // n+1 is the multiplicity label for etaab, etacd, etaac_bd
        U9coeffs(m, n) = 0.0;

        for (int i = 0; i < u4abc.size(); i++)
        {
          for (int j = 0; j < U6coeffs_arr1[i].size(); j++)
          {
            for (int k = 0; k < Z6coeffs_arr[i].size(); k++)
            {
              for (int l = 0; l < U6coeffs_arr2[i].size(); l++)
              {
                if (etaab_arr[i][k] == etaab[n] &&
                    etacd_arr[i][l] == etacd[n] &&
                    etaac_bd_arr[i][j] == etaac_bd[n] &&
                    etaac_arr[i][k] == etaac[m] &&
                    etabd_arr[i][j] == etabd[m] &&
                    etaab_cd_arr[i][l] == etaab_cd[m] &&
                    etaac_b_uarr[i][j] == etaac_b_zarr[i][k] &&
                    etaabc_d_arr1[i][j] == etaabc_d_arr2[i][l] &&
                    etaab_c_zarr[i][k] == etaab_c_uarr[i][l])
                {
                  U9coeffs(m, n) += U6coeffs_arr1[i][j] * Z6coeffs_arr[i][k] *
                                    U6coeffs_arr2[i][l];
                }
              }
            }
          }
        }
      }
    }

    return U9coeffs;
  }

} // namespace u4

#endif

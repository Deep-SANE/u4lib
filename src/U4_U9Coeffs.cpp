#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <decomposition.h>
#include <U4_U9Coeffs.h>
#include <U4basis.h>

int main()
{

  std::cout
      << "This program compute U9-recoupling coefficients of four U(4) irreps"
      << std::endl;

  int n14a, n24a, n34a, n44a;
  std::cout << "Please enter the I. irrep [n14a,n24a,n34a,n44a]: ";
  std::cin >> n14a >> n24a >> n34a >> n44a;
  u4::irrep u4a(n14a, n24a, n34a, n44a);
  if (n44a != 0)
  {
    u4a.normalize();
    std::cout << "Your irrep is normalized to " << u4a << std::endl;
  }

  int n14b, n24b, n34b, n44b;
  std::cout << "Please enter the II. irrep [n14b,n24b,n34b,n44b]: ";
  std::cin >> n14b >> n24b >> n34b >> n44b;
  u4::irrep u4b(n14b, n24b, n34b, n44b);
  if (n44b != 0)
  {
    u4b.normalize();
    std::cout << "Your irrep is normalized to " << u4b << std::endl;
  }

  int n14c, n24c, n34c, n44c;
  std::cout << "Please enter the III. irrep [n14c,n24c,n34c,n44c]: ";
  std::cin >> n14c >> n24c >> n34c >> n44c;
  u4::irrep u4c(n14c, n24c, n34c, n44c);
  if (n44c != 0)
  {
    u4c.normalize();
    std::cout << "Your irrep is normalized to " << u4c << std::endl;
  }

  int n14d, n24d, n34d, n44d;
  std::cout << "Please enter the IV. irrep [n14d,n24d,n34d,n44d]: ";
  std::cin >> n14d >> n24d >> n34d >> n44d;
  u4::irrep u4d(n14d, n24d, n34d, n44d);
  if (n44d != 0)
  {
    u4d.normalize();
    std::cout << "Your irrep is normalized to " << u4d << std::endl;
  }

  // couple the I. and II. irreps
  std::vector<int> etamaxsab;
  std::vector<u4::irrep> u4ab = u4::decomposition(u4a, u4b, etamaxsab);

  std::cout << "The normalized coupled irreps of the I and II are: "
            << std::endl;
  for (int i = 0; i < u4ab.size(); i++)
  {
    std::cout << "(" << i << "): [" << u4ab[i].n14 - u4ab[i].n44 << ","
              << u4ab[i].n24 - u4ab[i].n44 << "," << u4ab[i].n34 - u4ab[i].n44
              << ",0] \t eta = 1 ... " << etamaxsab[i] << std::endl;
  }

  // couple the I. and III. irreps
  std::vector<int> etamaxsac;
  std::vector<u4::irrep> u4ac = u4::decomposition(u4a, u4c, etamaxsac);

  std::cout << "The normalized coupled irreps of the I and III are: "
            << std::endl;
  for (int i = 0; i < u4ac.size(); i++)
  {
    std::cout << "(" << i << "): [" << u4ac[i].n14 - u4ac[i].n44 << ","
              << u4ac[i].n24 - u4ac[i].n44 << "," << u4ac[i].n34 - u4ac[i].n44
              << ",0] \t eta = 1 ... " << etamaxsac[i] << std::endl;
  }

  // couple the II. and IV. irreps
  std::vector<int> etamaxsbd;
  std::vector<u4::irrep> u4bd = u4::decomposition(u4b, u4d, etamaxsbd);

  std::cout << "The normalized coupled irreps of the II and IV are: "
            << std::endl;
  for (int i = 0; i < u4bd.size(); i++)
  {
    std::cout << "(" << i << "): [" << u4bd[i].n14 - u4bd[i].n44 << ","
              << u4bd[i].n24 - u4bd[i].n44 << "," << u4bd[i].n34 - u4bd[i].n44
              << ",0] \t eta = 1 ... " << etamaxsbd[i] << std::endl;
  }

  // couple the III. and IV. irreps
  std::vector<int> etamaxscd;
  std::vector<u4::irrep> u4cd = u4::decomposition(u4c, u4d, etamaxscd);

  std::cout << "The normalized coupled irreps of the III and IV are: "
            << std::endl;
  for (int i = 0; i < u4cd.size(); i++)
  {
    std::cout << "(" << i << "): [" << u4cd[i].n14 - u4cd[i].n44 << ","
              << u4cd[i].n24 - u4cd[i].n44 << "," << u4cd[i].n34 - u4cd[i].n44
              << ",0] \t eta = 1 ... " << etamaxscd[i] << std::endl;
  }

  int i_ab, i_ac, i_bd, i_cd, i_ab_cd, i_ac_bd;
  std::cout << "Please enter the index of the I-II coupled irrep you are "
               "interested: ";
  std::cin >> i_ab;
  std::cout << "Please enter the index of the I-III coupled irrep you are "
               "interested: ";
  std::cin >> i_ac;
  std::cout << "Please enter the index of the II-IV coupled irrep you are "
               "interested: ";
  std::cin >> i_bd;
  std::cout << "Please enter the index of the III-IV coupled irrep you are "
               "interested: ";
  std::cin >> i_cd;

  // couple the 4 irreps in the scheme (I-II)-(III-IV)
  std::vector<int> etamaxsab_cd;
  std::vector<u4::irrep> u4ab_cd =
      u4::decomposition(u4ab[i_ab], u4cd[i_cd], etamaxsab_cd);

  // couple the 4 irreps in scheme (I-III)-(II-IV)
  std::vector<int> etamaxsac_bd;
  std::vector<u4::irrep> u4ac_bd =
      u4::decomposition(u4ac[i_ac], u4bd[i_bd], etamaxsac_bd);

  std::cout << "The normalized coupled irreps of the I, II, III and IV are: "
            << std::endl;
  for (int i = 0; i < u4ab_cd.size(); i++)
  {
    std::cout << "(" << i << "): [" << u4ab_cd[i].n14 - u4ab_cd[i].n44 << ","
              << u4ab_cd[i].n24 - u4ab_cd[i].n44 << ","
              << u4ab_cd[i].n34 - u4ab_cd[i].n44 << ",0] \t eta = 1 ... "
              << etamaxsab_cd[i] << std::endl;
  }
  std::cout << "Please enter the index of the I-II-III-IV coupled irrep you "
               "are interested: ";
  std::cin >> i_ab_cd;

  int flag1 = 0, flag2 = 0;

  for (int j = 0; j < u4ac_bd.size(); j++)
  {
    if (u4ab_cd[i_ab_cd].n14 == u4ac_bd[j].n14 &&
        u4ab_cd[i_ab_cd].n24 == u4ac_bd[j].n24 &&
        u4ab_cd[i_ab_cd].n34 == u4ac_bd[j].n34 &&
        u4ab_cd[i_ab_cd].n44 == u4ac_bd[j].n44)
    {
      i_ac_bd = j;
      flag1++;
      break;
    }
  }

  // couple the 4 irreps in scheme ((I-II)-III)-IV
  std::vector<int> etamaxsab_c;
  std::vector<u4::irrep> u4ab_c =
      u4::decomposition(u4ab[i_ab], u4c, etamaxsab_c);

  // couple the 4 irreps in scheme ((I-III)-II)-IV
  std::vector<int> etamaxsac_b;
  std::vector<u4::irrep> u4ac_b =
      u4::decomposition(u4ac[i_ac], u4b, etamaxsac_b);

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

  for (int i = 0; i < u4abc.size(); i++)
  {
    // std::cout << u4abc[i] << std::endl;
    std::vector<int> etamaxsabc_d;
    std::vector<u4::irrep> u4abc_d =
        u4::decomposition(u4abc[i], u4d, etamaxsabc_d);
    for (int j = 0; j < u4abc_d.size(); j++)
    {
      if (u4ab_cd[i_ab_cd].n14 == u4abc_d[j].n14 &&
          u4ab_cd[i_ab_cd].n24 == u4abc_d[j].n24 &&
          u4ab_cd[i_ab_cd].n34 == u4abc_d[j].n34 &&
          u4ab_cd[i_ab_cd].n44 == u4abc_d[j].n44)
      {
        flag2++;
        break;
      }
    }
  }

  if (flag1 > 0 && flag2 == u4abc.size())
  {
    std::vector<int> etaab, etacd, etaac, etabd, etaab_cd, etaac_bd;

    // Start time measurement
    auto t0 = std::chrono::high_resolution_clock::now();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> coeffs = u4::U9(
        u4a, u4b, u4c, u4d, u4ab[i_ab], u4cd[i_cd], u4ac[i_ac], u4bd[i_bd],
        u4ab_cd[i_ab_cd], etaab, etamaxsab[i_ab], etacd, etamaxscd[i_cd],
        etaab_cd, etamaxsab_cd[i_ab_cd], etaac, etamaxsac[i_ac], etabd,
        etamaxsbd[i_bd], etaac_bd, etamaxsac_bd[i_ac_bd]);

    // End time measurement
    auto t1 = std::chrono::high_resolution_clock::now();

    u4ab[i_ab].normalize();
    u4cd[i_cd].normalize();
    u4ac[i_ac].normalize();
    u4bd[i_bd].normalize();
    u4ab_cd[i_ab_cd].normalize();

    std::cout << u4a << "\t" << u4b << "\t" << u4ab[i_ab] << "\tetaab\t\t";
    for (int i = 0; i < etaab.size(); i++)
      std::cout << etaab[i] << "\t";
    std::cout << std::endl;
    std::cout << u4c << "\t" << u4d << "\t" << u4cd[i_cd] << "\tetacd\t\t";
    for (int i = 0; i < etacd.size(); i++)
      std::cout << etacd[i] << "\t";
    std::cout << std::endl;
    std::cout << u4ac[i_ac] << "\t" << u4bd[i_bd] << "\t" << u4ab_cd[i_ab_cd]
              << "\tetaac_bd\t";
    for (int i = 0; i < etaac_bd.size(); i++)
      std::cout << etaac_bd[i] << "\t";
    std::cout << std::endl;

    std::cout << "etaac\t\tetabd\t\tetaab_cd\t\t" << std::endl;

    for (int i = 0; i < etaac.size(); i++)
    {
      std::cout << etaac[i] << "\t\t" << etabd[i] << "\t\t" << etaab_cd[i]
                << "\t\t\t\t" << coeffs.row(i) << std::endl;
    }

    // Calculate the duration
    std::chrono::duration<double> duration = t1 - t0;

    // Output the time in seconds
    std::cout << "Elapsed time: " << duration.count() << " seconds or "
              << duration.count() / 60.0 << " minutes" << std::endl;
  }
  else
  {
    std::cout << "Recoupling does not exist" << std::endl;
  }

  return 0;
}
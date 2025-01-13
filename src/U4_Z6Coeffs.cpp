#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <decomposition.h>
#include <U4_U6Z6Coeffs.h>
#include <U4basis.h>

int main()
{

  std::cout
      << "This program compute Z6-recoupling coefficients of three U(4) irreps"
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

  int i_ab, i_ac, i_abc;
  std::cout << "Please enter the index of the I-II coupled irrep you are "
               "interested: ";
  std::cin >> i_ab;
  std::cout << "Please enter the index of the I-III coupled irrep you are "
               "interested: ";
  std::cin >> i_ac;

  // couple the three irreps in scheme (I-II)-III
  std::vector<int> etamaxsab_c;
  std::vector<u4::irrep> u4ab_c =
      u4::decomposition(u4ab[i_ab], u4c, etamaxsab_c);

  std::cout << "The normalized coupled irreps of the I, II and III are: "
            << std::endl;
  for (int i = 0; i < u4ab_c.size(); i++)
  {
    std::cout << "(" << i << "): [" << u4ab_c[i].n14 - u4ab_c[i].n44 << ","
              << u4ab_c[i].n24 - u4ab_c[i].n44 << ","
              << u4ab_c[i].n34 - u4ab_c[i].n44 << ",0] \t eta = 1 ... "
              << etamaxsab_c[i] << std::endl;
  }
  std::cout << "Please enter the indices of the I-II-III coupled irrep you are "
               "interested: ";
  std::cin >> i_abc;

  // couple the three irreps in scheme (I-III)-II
  std::vector<int> etamaxsac_b;
  std::vector<u4::irrep> u4ac_b =
      u4::decomposition(u4ac[i_ac], u4b, etamaxsac_b);

  int flag = 0;
  for (int j = 0; j < u4ac_b.size(); j++)
  {
    if (u4ab_c[i_abc].n14 == u4ac_b[j].n14 &&
        u4ab_c[i_abc].n24 == u4ac_b[j].n24 &&
        u4ab_c[i_abc].n34 == u4ac_b[j].n34 &&
        u4ab_c[i_abc].n44 == u4ac_b[j].n44)
    {
      flag++;
      break;
    }
  }

  if (flag > 0)
  {
    std::vector<int> etaab, etaab_c, etaac, etaac_b;

    // Start time measurement
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<double> coeffs =
        u4::Z6(u4b, u4a, u4ab_c[i_abc], u4c, u4ab[i_ab], etaab, etaab_c,
               u4ac[i_ac], etaac, etaac_b);

    // End time measurement
    auto t1 = std::chrono::high_resolution_clock::now();

    u4ab[i_ab].normalize();
    u4ac[i_ac].normalize();
    u4ab_c[i_abc].normalize();

    std::cout << "Z([ni4b], [ni4a],   [ni4],   [ni4c] ; [ni4ab],eab,eab_c, "
                 "[ni4ac],eac,eac_b)"
              << std::endl;
    std::cout << u4b << u4a << u4ab_c[i_abc] << u4c << ";" << u4ab[i_ab]
              << "\t\t" << u4ac[i_ac] << std::endl;
    for (int i = 0; i < coeffs.size(); i++)
    {
      std::cout << "\t\t\t\t\t\t" << etaab[i] << "   " << etaab_c[i] << "\t\t  "
                << etaac[i] << "   " << etaac_b[i] << "\t\t" << coeffs[i]
                << std::endl;
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
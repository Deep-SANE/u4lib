#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <so3Coeffs/SO3_CGC.h>
 
#include <U4basis.h>
#include <U4decomposition.h>
#include <U4U3xU4U3.h>
 
#ifdef U4LIB_USE_SLEPC
#include <slepcsys.h>
#endif 

int main(int argc, char* argv[])
{  
#ifdef U4LIB_USE_SLEPC 
  SlepcInitialize(&argc, &argv, NULL, NULL);   
#endif

  int n14a, n24a, n34a, n44a;
  std::cout << "Please enter the first irrep [n14a,n24a,n34a,n44a]: ";
  std::cin >> n14a >> n24a >> n34a >> n44a;
  u4::irrep u4a(n14a, n24a, n34a, n44a);
  if (n44a != 0)
  { 
    u4a.normalize();
    std::cout << "Your irrep is normalized to " << u4a << std::endl;
  }

  int n14b, n24b, n34b, n44b;
  std::cout << "Please enter the second irrep [n14b,n24b,n34b,n44b]: ";
  std::cin >> n14b >> n24b >> n34b >> n44b;
  u4::irrep u4b(n14b, n24b, n34b, n44b);
  if (n44b != 0)
  { 
    u4b.normalize();
    std::cout << "Your irrep is normalized to " << u4b << std::endl;
  }
  
  auto [u4cs, etamaxs] = u4::decomposition(u4a, u4b);

  std::cout << "The normalized coupled irreps are: " << std::endl;
  for (int i = 0; i < u4cs.size(); i++)
  {
    std::cout << "(" << i << "): [" << u4cs[i].n14() - u4cs[i].n44() << ","
              << u4cs[i].n24() - u4cs[i].n44() << "," << u4cs[i].n34() - u4cs[i].n44()
              << ",0] \t eta = 1 ... " << etamaxs[i] << std::endl; 
  }
     
  int index;
  std::cout
      << "Please enter the index of the coupled irrep you are interested: "; 
  std::cin >> index; 

  // Start time measurement 
  auto t0 = std::chrono::high_resolution_clock::now();

  std::vector<u4::gelfand> gArrA = u4::gelfandBasis(u4a),
                           gArrB = u4::gelfandBasis(u4b),
                           gArrC = u4::gelfandBasis(u4cs[index]); 
  
  u34utils::SO3_CGC cgcs;
                           
  auto coeffs = u4::WigU4U3(u4a, u4b, u4cs[index], etamaxs[index], -1,   
                            gArrA, gArrB, gArrC, cgcs);

  // End time measurement  
  auto t1 = std::chrono::high_resolution_clock::now();

  const auto& u4c = u4cs[index]; 

  for (int n13a = u4a.n14(); n13a >= u4a.n24(); n13a--)
    for (int n23a = u4a.n24(); n23a >= u4a.n34(); n23a--)
      for (int n33a = u4a.n34(); n33a >= u4a.n44(); n33a--) 
        for (int n13b = u4b.n14(); n13b >= u4b.n24(); n13b--)
          for (int n23b = u4b.n24(); n23b >= u4b.n34(); n23b--)
            for (int n33b = u4b.n34(); n33b >= u4b.n44(); n33b--)
              for (int n13c = u4c.n14(); n13c >= u4c.n24(); n13c--)
                for (int n23c = u4c.n24(); n23c >= u4c.n34(); n23c--)   
                  for (int n33c = u4c.n34(); n33c >= u4c.n44(); n33c--)  
                  {   
                    u3::irrep u3a(n13a,n23a,n33a),
                              u3b(n13b,n23b,n33b),
                              u3c(n13c,n23c,n33c);

                    auto [missing, coeffs_] = coeffs.access(u3a,u3b,u3c);  
 
                    if (missing == true)
                      continue;  

                    std::cout << "< " << u4a.n14() << "," << u4a.n24() << "," << u4a.n34() 
                              << ",0;" << n13a << "," << n23a << "," << n33a << " ; " 
                              << u4b.n14() << "," << u4b.n24() << "," << u4b.n34() 
                              << ",0;" << n13b << "," << n23b << "," << n33b << " ||| "
                              << u4c.n14() << "," << u4c.n24() << "," << u4c.n34() << "," << u4c.n44()
                              << ";" << n13c << "," << n23c << "," << n33c << " >"  
                              << std::endl;
                    for (int r = 0; r < coeffs_.rows(); r++)
                      std::cout << "rho = " << r+1 << " --> \t" << coeffs_.row(r) << std::endl;
            } 

  //std::cout << "Reduced coefficients count: " << coeffs.size() << std::endl;

  // Calculate the duration
  std::chrono::duration<double> duration = t1 - t0;

  // Output the time in seconds
  std::cout << "Elapsed time: " << duration.count() << " seconds or "
            << duration.count() / 60.0 << " minutes" << std::endl; 

#ifdef U4LIB_USE_SLEPC 
  SlepcFinalize();
#endif

  return 0;
}
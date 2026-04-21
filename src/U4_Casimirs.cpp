#include <iostream>

#include <u4irrep.h>

int main()
{
  int n14, n24, n34, n44;

  while (true)
  {
	std::cout << "Enter [n14,n24,n34,n44]: ";
	std::cin >> n14 >> n24 >> n34 >> n44;

	u4::irrep irr(n14, n24, n34, n44);
	std::cout << "C2=" << irr.C2() << " C3=" << irr.C3() 
			  << " C4=" << irr.C4() << std::endl;
  }
}

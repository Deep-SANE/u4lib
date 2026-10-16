#ifndef U4LIB_GLOBAL_DEFINITIONS_H
#define U4LIB_GLOBAL_DEFINITIONS_H

#include <iostream>

#include <Eigen/Dense>

#include <ND.h>

namespace u4
{

//////////// COMPILER DEFINITIONS //////////////

#define U4LIB_ENABLE_WIGU4U3_CACHE
// #define TIME_WIGU4U3

#define U4LIB_ENABLE_WIGU4SU2SU2_CACHE
// #define TIME_WIGU4SU2SU2

#define U4LIB_ENABLE_U4U6_CACHE
#define U4LIB_ENABLE_U4Z6_CACHE
// #define TIME_U4U6Z6

#define U4LIB_ENABLE_U4U9_CACHE
// #define TIME_U4U9

//////////// INTEGER DEFINITIONS //////////////

  using uqn_t = uint8_t;  // unsigned quantum numbers: 0 to 255
  using qn_t = int8_t;    // signed quantum numbers: -128 to 127
  using mult_t = uint8_t; // multiplicities: 0 to 255

//////////// COUPLING & RECOUPLING COEFFICIENTS //////////////

/* General rule
1. operator[] -> accessing using index
2. operator() -> accessing using multiplicities
3. access for non-const object: allowing modification
4. access for const object: forbidding modification
5. data() -> exposing internal data
*/
  using coeffs_t = std::vector<double>;

  template <typename T>
  class U4U3
  {
  	T coeffs_;
    size_t etamax_;

  public:
  	U4U3() = default;
  	U4U3(size_t rhomax, size_t etamax) 
      : coeffs_(rhomax * etamax, 0.0), etamax_(etamax)
    {}
  	U4U3(T&& coeffs, size_t etamax)
      : coeffs_(std::move(coeffs)), etamax_(etamax)
    {}
    U4U3(Eigen::MatrixXd&& coeffs, size_t etamax)
      : etamax_(etamax)
    {
      coeffs_.resize(coeffs.size());
      for (size_t r = 0; r < coeffs.rows(); r++)
        for (size_t e = 0; e < coeffs.cols(); e++)
          coeffs_[r * etamax + e] = coeffs(r, e);
    }

  	size_t size() const { return coeffs_.size(); }
  	bool empty() const { return coeffs_.empty(); }
  	const auto& data() const { return coeffs_; }

    size_t etamax() const { return etamax_; }
  	size_t rhomax() const { return coeffs_.size() / etamax_; }

  	double& operator[](size_t i)
  	{
  	  assert(i < coeffs_.size());
      return coeffs_[i];
  	}

  	double operator[](size_t i) const
  	{
  	  assert(i < coeffs_.size());
      return coeffs_[i];
  	}

  	double& operator()(size_t rho, size_t eta)
  	{
  	  return operator[](rho * etamax_ + eta);
  	}

  	double operator()(size_t rho, size_t eta) const
  	{
  	  return operator[](rho * etamax_ + eta);
  	}

    U4U3& operator*=(double s) 
    {
      for (auto& x : coeffs_) x *= s;
      return *this;
    }

    U4U3 operator*(double s) const 
    {
      U4U3 result = *this;
      result *= s;
      return result;
    }

    friend U4U3 operator*(double s, const U4U3& t) 
    {
      return t * s;
    }

    U4U3& operator/=(double s) 
    {
      assert(s != 0.0);
      for (auto& x : coeffs_) x /= s;
      return *this;
    }

    U4U3 operator/(double s) const 
    {
      assert(s != 0.0);
      U4U3 result = *this;
      result /= s;
      return result;
    }

  	friend std::ostream& operator<<(std::ostream& s, const U4U3& x)
  	{
  	  s << "rho\teta\n";
  	  for (size_t r = 0; r < x.rhomax(); r++)
        for (size_t e = 0; e < x.etamax(); e++)
          s << r << "\t" << e << "\t" << x(r, e) << "\n";
  	  return s;
  	}

    void print(std::ostream& s = std::cout) const
    {
      s << *this;
    }
  };

  template <typename T>
  class U4SU2SU2
  {
    T coeffs_;
    utils::NDIndex4 ndi_;

  public:
    U4SU2SU2()
    {
      ndi_.add_dimensions(0, 0, 0, 0);
      assert(coeffs_.size() == ndi_.cardinality());
    }

    U4SU2SU2(size_t z1max, size_t z2max, size_t z3max, size_t etamax)
      : coeffs_(z1max * z2max * z3max * etamax, 0.0)
    {
      ndi_.add_dimensions(z1max, z2max, z3max, etamax); 
    }

    U4SU2SU2(T&& coeffs,
    	       size_t z1max, size_t z2max, size_t z3max, size_t etamax)
      : coeffs_(std::move(coeffs))
    {
      ndi_.add_dimensions(z1max, z2max, z3max, etamax);
      assert(coeffs_.size() == ndi_.cardinality());
    }

    size_t size() const { return coeffs_.size(); }
  	bool empty() const { return coeffs_.empty(); }
  	const auto& data() const { return coeffs_; }

    size_t zetaAmax() const { return ndi_.dimension(0); }
    size_t zetaBmax() const { return ndi_.dimension(1); }
    size_t zetaCmax() const { return ndi_.dimension(2); }
    size_t etamax() const { return ndi_.dimension(3); }

    double& operator[](size_t i)
    {
      assert(i < coeffs_.size());
      return coeffs_[i];
    }

    double operator[](size_t i) const
    {
      assert(i < coeffs_.size());
      return coeffs_[i];
    }

    double& operator()(size_t z1, size_t z2, size_t z3, size_t eta)
    {
      std::array multi_index = {z1, z2, z3, eta};
      size_t i = ndi_.to_linear(multi_index);
      return operator[](i);
    }

    double operator()(size_t z1, size_t z2, size_t z3, size_t eta) const
    {
      std::array multi_index = {z1, z2, z3, eta};
      size_t i = ndi_.to_linear(multi_index);
      return operator[](i);
    }

    U4SU2SU2& operator*=(double s) 
    {
      for (auto& x : coeffs_) x *= s;
      return *this;
    }

    U4SU2SU2 operator*(double s) const 
    {
      U4SU2SU2 result = *this;
      result *= s;
      return result;
    }

    friend U4SU2SU2 operator*(double s, const U4SU2SU2& t) 
    {
      return t * s;
    }

    U4SU2SU2& operator/=(double s) 
    {
      assert(s != 0.0);
      for (auto& x : coeffs_) x /= s;
      return *this;
    }

    U4SU2SU2 operator/(double s) const 
    {
      assert(s != 0.0);
      U4SU2SU2 result = *this;
      result /= s;
      return result;
    }

    friend std::ostream& operator<<(std::ostream& s, const U4SU2SU2& x)
  	{
  	  s << "zetaA\tzetaB\tzetaC\teta\n";
  	  std::vector<size_t> multi(4);
  	  for (size_t i = 0; i < x.size(); i++)
  	  {
  	    x.ndi_.to_multi(i, multi);
  	    s << multi[0] << "\t" << multi[1] << "\t"
  	      << multi[2] << "\t" << multi[3] << "\t -> "
  	      << x[i] << "\n";
  	  }
	  return s;
  	}

    void print(std::ostream& s = std::cout) const
    {
      s << *this;
    }
  };

  template <typename T>
  class U4U6
  {
    T coeffs_;
    utils::NDIndex4 ndi_;

  public:
    U4U6()
    {
      ndi_.add_dimensions(0, 0, 0, 0);
      assert(coeffs_.size() == ndi_.cardinality());
    }

    U4U6(size_t etaABmax, size_t etaAB_Cmax, 
         size_t etaBCmax, size_t etaA_BCmax)
      : coeffs_(etaABmax * etaAB_Cmax * etaBCmax * etaA_BCmax, 0.0)
    {
      ndi_.add_dimensions(etaABmax, etaAB_Cmax, etaBCmax, etaA_BCmax);
    }

    U4U6(T&& coeffs, size_t etaABmax, size_t etaAB_Cmax, 
                     size_t etaBCmax, size_t etaA_BCmax)
      : coeffs_(std::move(coeffs))
    {
      ndi_.add_dimensions(etaABmax, etaAB_Cmax, etaBCmax, etaA_BCmax);
      assert(coeffs_.size() == ndi_.cardinality());
    }

    size_t size() const { return coeffs_.size(); }
  	bool empty() const { return coeffs_.empty(); }
  	const auto& data() const { return coeffs_; }

    size_t etaABmax() const { return ndi_.dimension(0); }
    size_t etaAB_Cmax() const { return ndi_.dimension(1); }
    size_t etaBCmax() const { return ndi_.dimension(2); }
    size_t etaA_BCmax() const { return ndi_.dimension(3); }

    double& operator[](size_t i)
    {
      assert(i < coeffs_.size());
      return coeffs_[i];
    }

    double operator[](size_t i) const
    {
      assert(i < coeffs_.size());
      return coeffs_[i];
    }

    double& operator()(size_t etaAB, size_t etaAB_C,
    				           size_t etaBC, size_t etaA_BC)
    {
      std::array multi_index = {etaAB, etaAB_C, etaBC, etaA_BC};
      size_t i = ndi_.to_linear(multi_index);
      return operator[](i);
    }

    double operator()(size_t etaAB, size_t etaAB_C,
    				          size_t etaBC, size_t etaA_BC) const
    {
      std::array multi_index = {etaAB, etaAB_C, etaBC, etaA_BC};
      size_t i = ndi_.to_linear(multi_index);
      return operator[](i);
    }

    friend std::ostream& operator<<(std::ostream& s, const U4U6& x)
  	{
  	  s << "etaAB\tetaAB_C\tetaBC\tetaA_BC\n";
  	  std::vector<size_t> multi(4);
  	  for (size_t i = 0; i < x.size(); i++)
  	  {
  	    x.ndi_.to_multi(i, multi);
  	    s << multi[0] << "\t" << multi[1] << "\t"
  	      << multi[2] << "\t" << multi[3] << "\t -> "
  	      << x[i] << "\n";
  	  }
	    return s;
  	}

    void print(std::ostream& s = std::cout) const
    {
      s << *this;
    }
  };

  template <typename T>
  class U4Z6
  {
    T coeffs_;
    utils::NDIndex4 ndi_;

  public:
    U4Z6()
    {
      ndi_.add_dimensions(0, 0, 0, 0);
      assert(coeffs_.size() == ndi_.cardinality());
    }

    U4Z6(size_t etaABmax, size_t etaAB_Cmax, 
         size_t etaACmax, size_t etaAC_Bmax)
      : coeffs_(etaABmax * etaAB_Cmax * etaACmax * etaAC_Bmax, 0.0)
    {
      ndi_.add_dimensions(etaABmax, etaAB_Cmax, etaACmax, etaAC_Bmax);
    }

    U4Z6(T&& coeffs, size_t etaABmax, size_t etaAB_Cmax, 
                 		 size_t etaACmax, size_t etaAC_Bmax)
        : coeffs_(std::move(coeffs))
    {
      ndi_.add_dimensions(etaABmax, etaAB_Cmax, etaACmax, etaAC_Bmax);
      assert(coeffs_.size() == ndi_.cardinality());
    }

    size_t size() const { return coeffs_.size(); }
  	bool empty() const { return coeffs_.empty(); }
  	const auto& data() const { return coeffs_; }

    size_t rhoABmax() const { return ndi_.dimension(0); }
    size_t rhoAB_Cmax() const { return ndi_.dimension(1); }
    size_t rhoACmax() const { return ndi_.dimension(2); }
    size_t rhoAC_Bmax() const { return ndi_.dimension(3); }

    double& operator[](size_t i)
    {
      assert(i < coeffs_.size());
      return coeffs_[i];
    }

    double operator[](size_t i) const
    {
      assert(i < coeffs_.size());
      return coeffs_[i];
    }

    double& operator()(size_t etaAB, size_t etaAB_C,
    				           size_t etaAC, size_t etaAC_B)
    {
      std::array multi_index = {etaAB, etaAB_C, etaAC, etaAC_B};
      size_t i = ndi_.to_linear(multi_index);
      return operator[](i);
    }

    double operator()(size_t etaAB, size_t etaAB_C,
    				          size_t etaAC, size_t etaAC_B) const
    {
      std::array multi_index = {etaAB, etaAB_C, etaAC, etaAC_B};
      size_t i = ndi_.to_linear(multi_index);
      return operator[](i);
    }

    friend std::ostream& operator<<(std::ostream& s, const U4Z6& x)
  	{
  	  s << "etaAB\tetaAB_C\tetaAC\tetaAC_B\n";
  	  std::vector<size_t> multi(4);
  	  for (size_t i = 0; i < x.size(); i++)
  	  {
  	    x.ndi_.to_multi(i, multi);
  	    s << multi[0] << "\t" << multi[1] << "\t"
  	      << multi[2] << "\t" << multi[3] << "\t -> "
  	      << x[i] << "\n";
  	  }
	    return s;
  	}

    void print(std::ostream& s = std::cout) const
    {
      s << *this;
    }
  };

  template <typename T>
  class U4U9
  {
    T coeffs_;
    utils::NDIndex6 ndi_;

  public:
    U4U9()
    {
      ndi_.add_dimensions(0, 0, 0, 0, 0, 0);
      assert(coeffs_.size() == ndi_.cardinality());
    }

    U4U9(size_t etaACmax, size_t etaBDmax, size_t etaABCDmax,
         size_t etaABmax, size_t etaCDmax, size_t etaACBDmax)
        : coeffs_(etaACmax * etaBDmax * etaABCDmax * 
        		      etaABmax * etaCDmax * etaACBDmax, 0.0)
    {
      ndi_.add_dimensions(etaACmax, etaBDmax, etaABCDmax, 
      					          etaABmax, etaCDmax, etaACBDmax);
    }

    U4U9(T&& coeffs,
    	   size_t etaACmax, size_t etaBDmax, size_t etaABCDmax,
         size_t etaABmax, size_t etaCDmax, size_t etaACBDmax)
        : coeffs_(std::move(coeffs))
    {
      ndi_.add_dimensions(etaACmax, etaBDmax, etaABCDmax, 
                          etaABmax, etaCDmax, etaACBDmax);
      assert(coeffs_.size() == ndi_.cardinality());
    }

    size_t size() const { return coeffs_.size(); }
  	bool empty() const { return coeffs_.empty(); }
  	const auto& data() const { return coeffs_; }

    size_t etaACmax() const { return ndi_.dimension(0); }
    size_t etaBDmax() const { return ndi_.dimension(1); }
    size_t etaABCDmax() const { return ndi_.dimension(2); }
    size_t etaABmax() const { return ndi_.dimension(3); }
    size_t etaCDmax() const { return ndi_.dimension(4); }
    size_t etaACBDmax() const { return ndi_.dimension(5); }

    double& operator[](size_t i)
    {
      assert(i < coeffs_.size());
      return coeffs_[i];
    }

    double operator[](size_t i) const
    {
      assert(i < coeffs_.size());
      return coeffs_[i];
    }

    double& operator()(size_t etaAC, size_t etaBD, size_t etaABCD, 
    				           size_t etaAB, size_t etaCD, size_t etaACBD)
    {
      std::array multi_index = {etaAC, etaBD, etaABCD, etaAB, etaCD, etaACBD};
      size_t i = ndi_.to_linear(multi_index);
      return operator[](i);
    }

    double operator()(size_t etaAC, size_t etaBD, size_t etaABCD, 
    				          size_t etaAB, size_t etaCD, size_t etaACBD) const
    {
      std::array multi_index = {etaAC, etaBD, etaABCD, etaAB, etaCD, etaACBD};
      size_t i = ndi_.to_linear(multi_index);
      return operator[](i);
    }

    friend std::ostream& operator<<(std::ostream& s, const U4U9& x)
  	{
  	  s << "etaAC\tetaBD\tetaABCD\tetaAB\tetaCD\tetaACBD\n";
  	  std::vector<size_t> multi(6);
  	  for (size_t i = 0; i < x.size(); i++)
  	  {
  	    x.ndi_.to_multi(i, multi);
  	    s << multi[0] << "\t" << multi[1] << "\t"
  	      << multi[2] << "\t" << multi[3] << "\t"
  	      << multi[4] << "\t" << multi[5] << "\t -> "
  	      << x[i] << "\n";
  	  }
	    return s;
  	}

    void print(std::ostream& s = std::cout) const
    {
      s << *this;
    }
  };

} // namespace u3

#endif

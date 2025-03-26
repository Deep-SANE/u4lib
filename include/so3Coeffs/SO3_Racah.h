#ifndef U4LIB_SO3_RACAH_H
#define U4LIB_SO3_RACAH_H

#include <cmath>

#ifdef U4LIB_TIME_LOG
#include <chrono>
#include <iostream>
#endif

extern "C"
{
#include <wigxjpf.h>
}

namespace u4
{
  class SO3_Racah
  {
#ifdef U4LIB_TIME_LOG
    double time_ = 0.0;
#endif

  public:
    SO3_Racah()
    {
      wig_table_init(2 * 100, 9);
      wig_temp_init(2 * 100);
    }

    ~SO3_Racah()
    {
      wig_temp_free();
      wig_table_free();

#ifdef U4LIB_TIME_LOG
      std::cerr << "SO3_Racah::compute time: " << time_ << " [s]" << std::endl;
#endif
    }

    double compute(int ll1, int ll2, int ll, int ll3, int ll12, int ll23)
    {
#ifdef U4LIB_TIME_LOG
      auto t0 = std::chrono::high_resolution_clock::now();
#endif

      double phase = ((ll1 + ll2 + ll3 + ll) % 4 == 0) ? 1 : -1;
      double factor = std::sqrt((static_cast<double>(ll12) + 1.0) * (static_cast<double>(ll23) + 1.0));
      double val6j = phase * factor * wig6jj(ll1, ll2, ll12, ll3, ll, ll23);

#ifdef U4LIB_TIME_LOG
      auto t1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = t1 - t0;
      time_ += duration.count();
#endif

      return val6j;
    }
  };
} // namespace u4

#endif

#ifndef U4LIB_SO3_CGC_H
#define U4LIB_SO3_CGC_H

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
  class SO3_CGC
  {
#ifdef U4LIB_TIME_LOG
    double time_ = 0.0;
#endif

  public:
    SO3_CGC()
    {
      wig_table_init(2 * 100, 9);
      wig_temp_init(2 * 100);
    }

    ~SO3_CGC()
    {
      wig_temp_free();
      wig_table_free();

#ifdef U4LIB_TIME_LOG
      std::cerr << "SO3_CGC::compute time: " << time_ << " [s]" << std::endl;
#endif
    }

    double compute(int ll1, int ll2, int ll3, int mm1, int mm2, int mm3)
    {
#ifdef U4LIB_TIME_LOG
      auto t0 = std::chrono::high_resolution_clock::now();
#endif

      double phase = ((-ll1 + ll2 - mm3) % 4 == 0) ? 1.0 : -1.0;
      double factor = std::sqrt(static_cast<double>(ll3) + 1.0);

      double val3j = wig3jj(ll1, ll2, ll3, mm1, mm2, -mm3) * phase * factor;

#ifdef U4LIB_TIME_LOG
      auto t1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = t1 - t0;
      time_ += duration.count();
#endif

      return val3j;
    }
  };

  /*
  // Custom hash function for tuple<int, int, int, int, int, int>
  struct CGC_hash
  {
    template <typename T>
    static void hash_combine(std::size_t& seed, const T& val)
    {
      seed ^= std::hash<T>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    std::size_t
    operator()(const std::tuple<int, int, int, int, int, int>& key) const
    {
      std::size_t seed = 0;
      hash_combine(seed, std::get<0>(key));
      hash_combine(seed, std::get<1>(key));
      hash_combine(seed, std::get<2>(key));
      hash_combine(seed, std::get<3>(key));
      hash_combine(seed, std::get<4>(key));
      hash_combine(seed, std::get<5>(key));
      return seed;
    }
  };
  */

  /*
  class CGC_SO3
  {
  private:
    std::unordered_map<std::tuple<int, int, int, int, int, int>, double,
                       CGC_hash>
        cache;

  public:
    double compute(int ll1, int ll2, int ll3, int mm1, int mm2, int mm3)
    {
      std::tuple key{ll1, ll2, ll3, mm1, mm2, mm3};
      if (cache.find(key) != cache.end())
        return cache[key];

      double val3j;

      std::tuple key1{ll1, ll2, ll3, -mm1, -mm2, -mm3};
      std::tuple key2{ll2, ll1, ll3, mm2, mm1, mm3};
      std::tuple key3{ll1, ll3, ll2, mm1, -mm3, -mm2};
      std::tuple key4{ll3, ll2, ll1, -mm3, mm2, -mm1};
      std::tuple key5{ll3, ll1, ll2, mm3, -mm1, mm2};
      std::tuple key6{ll2, ll3, ll1, -mm2, mm3, mm1};

      if (cache.find(key1) != cache.end())
      {
        int phase = ((ll1 + ll2 - ll3) % 4 == 0) ? 1 : -1;
        val3j = phase * cache[key1];
      }

      else if (cache.find(key2) != cache.end())
      {
        int phase = ((ll1 + ll2 - ll3) % 4 == 0) ? 1 : -1;
        val3j = phase * cache[key2];
      }

      else if (cache.find(key3) != cache.end())
      {
        int phase = ((ll1 - mm1) % 4 == 0) ? 1 : -1;
        val3j = phase * std::sqrt((ll3 + 1.0) / (ll2 + 1.0)) * cache[key3];
      }

      else if (cache.find(key4) != cache.end())
      {
        int phase = ((ll2 + mm2) % 4 == 0) ? 1 : -1;
        val3j = phase * std::sqrt((ll3 + 1.0) / (ll1 + 1.0)) * cache[key4];
      }

      else if (cache.find(key5) != cache.end())
      {
        int phase = ((ll1 - mm1) % 4 == 0) ? 1 : -1;
        val3j = phase * std::sqrt((ll3 + 1.0) / (ll2 + 1.0)) * cache[key5];
      }

      else if (cache.find(key6) != cache.end())
      {
        int phase = ((ll2 + mm2) % 4 == 0) ? 1 : -1;
        val3j = phase * std::sqrt((ll3 + 1.0) / (ll1 + 1.0)) * cache[key6];
      }

      else if ((ll3 == (ll1 + ll2) &&
                ((mm1 + mm2) == ll3 || (mm1 + mm2) == -ll3)) ||
               ll1 == 0 || ll2 == 0)
      {
        val3j = 1.0;
      }

      else
      {
        int phase = ((-ll1 + ll2 - mm3) % 4 == 0) ? 1 : -1;

        wig_table_init(2 * 100, 9);
        wig_temp_init(2 * 100);

        val3j =
            wig3jj(ll1, ll2, ll3, mm1, mm2, -mm3) * phase * std::sqrt(ll3 + 1);

        wig_temp_free();
        wig_table_free();
      }

      // std::cout << ll1 << "," << ll2 << "," << ll3 << "," << mm1 << "," <<
      // mm2 << "," << mm3 << "->" << val3j << std::endl;

      cache.emplace(key, val3j);
      return val3j;
    }
  };
  */

} // namespace u4

#endif

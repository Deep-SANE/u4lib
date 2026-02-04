#ifndef U4LIB_U4LIB_H
#define U4LIB_U4LIB_H

#include <U4U3xU4U3.h>
#include <U4SU2SU2xU4SU2SU2.h>
#include <U4_U6Z6Coeffs.h>
#include <U4_U9Coeffs.h>

namespace u4
{
  inline void initialize()
  {
    WigU4U3_db::initialize();
    WigU4SU2SU2_db::initialize();
    U4U6_db::initialize();
    U4Z6_db::initialize();
    U4U9_db::initialize();
  }

  inline void finalize() 
  {
    WigU4U3_db::finalize();
    WigU4SU2SU2_db::finalize();
    U4U6_db::finalize();
    U4Z6_db::finalize();
    U4U9_db::finalize();
  }

  inline void initialize_thread() 
  {
    WigU4U3_db::initialize_thread();
    WigU4SU2SU2_db::initialize_thread();
    U4U6_db::initialize_thread();
    U4Z6_db::initialize_thread();
    U4U9_db::initialize_thread();
  }

  inline void finalize_thread()
  {
    WigU4U3_db::finalize_thread();
    WigU4SU2SU2_db::finalize_thread();
    U4U6_db::finalize_thread();
    U4Z6_db::finalize_thread();
    U4U9_db::finalize_thread();
  }
}

#endif

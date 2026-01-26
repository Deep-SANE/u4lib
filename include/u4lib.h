#ifndef U4LIB_U4LIB_H
#define U4LIB_U4LIB_H

#include <U4U3xU4U3.h>

namespace u4
{

  inline void initialize()
  {
    WigU4U3_db::initialize();
  }

  inline void finalize() 
  {
    WigU4U3_db::finalize();
  }

  inline void initialize_thread() 
  {
    WigU4U3_db::initialize_thread();
  }

  inline void finalize_thread()
  {
    WigU4U3_db::finalize_thread();
  }
}

#endif

#ifndef U4LIB_U4LIB_H
#define U4LIB_U4LIB_H

#include <u3lib.h>   // u4 depends on u3; init cascades u4 -> u3 -> utils

#include <U4U3xU4U3.h>
#include <U4SU2SU2xU4SU2SU2.h>
#include <U4_U6Z6Coeffs.h>
#include <U4_U9Coeffs.h>

namespace u4
{
  // initialize()/finalize() cascade down to u3 (and thus the utils
  // foundation), so a caller using u4 only needs u4::initialize(). argc/argv
  // are forwarded and default to null. All four are idempotent
  // (see u3lib.h): repeated calls are no-ops (process-wide / per-thread).
  inline bool& initialized_flag() { static bool f = false; return f; }
  inline bool& thread_initialized_flag() { static thread_local bool f = false; return f; }

  inline void initialize(int* argc = nullptr, char*** argv = nullptr)
  {
    if (initialized_flag()) return;
    initialized_flag() = true;
    u3::initialize(argc, argv);   // cascades: u3 -> utils
    WigU4U3_db::initialize();
    WigU4SU2SU2_db::initialize();
    U4U6_db::initialize();
    U4Z6_db::initialize();
    U4U9_db::initialize();
  }

  inline void finalize()
  {
    if (!initialized_flag()) return;
    initialized_flag() = false;
    WigU4U3_db::finalize();
    WigU4SU2SU2_db::finalize();
    U4U6_db::finalize();
    U4Z6_db::finalize();
    U4U9_db::finalize();
    u3::finalize();   // tears down u3 then the foundation
  }

  inline void initialize_thread()
  {
    if (thread_initialized_flag()) return;
    thread_initialized_flag() = true;
    u3::initialize_thread();
    WigU4U3_db::initialize_thread();
    WigU4SU2SU2_db::initialize_thread();
    U4U6_db::initialize_thread();
    U4Z6_db::initialize_thread();
    U4U9_db::initialize_thread();
  }

  inline void finalize_thread()
  {
    if (!thread_initialized_flag()) return;
    thread_initialized_flag() = false;
    WigU4U3_db::finalize_thread();
    WigU4SU2SU2_db::finalize_thread();
    U4U6_db::finalize_thread();
    U4Z6_db::finalize_thread();
    U4U9_db::finalize_thread();
    u3::finalize_thread();
  }

  inline void clear()
  {
    wigU4U3db().clear();
    wigU4SU2SU2db().clear();
    u4u6db().clear();
    u4z6db().clear();
    u4u9db().clear();
  }
}

#endif

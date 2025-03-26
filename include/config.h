#ifndef U4LIB_CONFIG_H
#define U4LIB_CONFIG_H

#include "gelfand/gelfand.h"
#include "irrep/irrep.h"

namespace u4
{
  using irrep = internal::irrep_int_rep_t;
  using gelfand = internal::gelfand_int_rep_t;
} // namespace u4

#endif

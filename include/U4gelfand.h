#ifndef U4LIB_GELFAND_H
#define U4LIB_GELFAND_H

#include <vector>

#include "internal/gelfand/gelfand.h"

namespace u4
{
  using gelfand = internal::gelfand_int_rep_t;
  using gelfands = std::vector<u4::gelfand>;
}

#endif

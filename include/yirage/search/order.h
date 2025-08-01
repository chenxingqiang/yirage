#pragma once

#include <vector>

namespace yirage {
namespace search {

struct Order {
  std::vector<int> v;
  int type;
  Order(std::vector<int> const &v, int type);
  bool operator<(Order const &) const;
  bool operator<=(Order const &) const;
};

} // namespace search
} // namespace yirage
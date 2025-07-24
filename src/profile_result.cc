#include "yirage/profile_result.h"

#include <limits>

namespace yirage {

ProfileResult ProfileResult::infinity() {
  return ProfileResult{1000};
}

} // namespace yirage

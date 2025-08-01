#pragma once

#include "yirage/kernel/graph.h"
#include "yirage/search/abstract_expr/abstract_expr.h"
#include "yirage/threadblock/graph.h"

namespace yirage {
namespace search {

void abstract_expr_eval(
    threadblock::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>> &patterns);

void abstract_expr_eval(
    kernel::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>> &patterns);

} // namespace search
} // namespace yirage
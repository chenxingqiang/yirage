/* Copyright 2023-2024 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "yirage/threadblock/operator.h"

namespace yirage {
namespace threadblock {

class TBMatmulOp : public yirage::threadblock::TBOperator {
public:
  TBMatmulOp(Graph *_graph, STensor const &A, STensor const &B);
  static bool compitble_layouts(STensor const &A, STensor const &B);
  ~TBMatmulOp();
  operator json() const override;
};

} // namespace threadblock
} // namespace yirage

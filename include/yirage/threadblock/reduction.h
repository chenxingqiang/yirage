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

using namespace cutlass;

class TBReductionOp : public TBOperator {
public:
  TBReductionOp(Graph *graph,
                STensor const &_input,
                int reduce_dim,
                int reduce_size);
  ~TBReductionOp();

  operator json() const override;

public:
  int reduce_dim, reduce_size;
};

} // namespace threadblock
} // namespace yirage

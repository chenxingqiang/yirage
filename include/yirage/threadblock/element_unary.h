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

class TBElementUnaryOp : public TBOperator {
public:
  TBElementUnaryOp(Graph *_graph,
                   STensor const &_input,
                   yirage::type::TBOperatorType _type,
                   float const &scalar);
  ~TBElementUnaryOp();

  operator json() const override;

public:
  float const scalar;
};

class TBClampUnaryOp : public TBElementUnaryOp {
public:
  TBClampUnaryOp(Graph *_graph,
                 STensor const &_input,
                 float const &min_val,
                 float const &max_val);

public:
  float const min_val;
  float const max_val;
};

} // namespace threadblock
} // namespace yirage

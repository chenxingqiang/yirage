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
#include "yirage/kernel/device_tensor.h"
#include "yirage/threadblock/smem_tensor.h"
#include "yirage/type.h"
#include <vector>
#include <vector_types.h>

namespace yirage {
namespace threadblock {

class Graph;

class TBOperator {
public:
  TBOperator(Graph *graph, yirage::type::TBOperatorType);
  TBOperator(Graph *graph, yirage::type::TBOperatorType, STensor const &input1);
  TBOperator(Graph *graph,
             yirage::type::TBOperatorType,
             STensor const &input1,
             STensor const &input2);
  TBOperator(Graph *graph,
             yirage::type::TBOperatorType,
             std::vector<STensor> const &inputs);
  int get_input_stensors(STensor **inputs);
  int get_output_stensors(STensor **inputs);

  virtual ~TBOperator();

  virtual operator json() const = 0;

public:
  Graph *bgraph;
  yirage::type::TBOperatorType op_type;
  std::vector<STensor> input_tensors;
  std::vector<STensor> output_tensors;
};

class TBInputOp : public TBOperator {
public:
  TBInputOp(Graph *_graph,
            yirage::kernel::DTensor const &dtensor,
            int3 input_map,
            int forloop_dim,
            yirage::layout::SmemLayout layout);
  ~TBInputOp();

  operator json() const override;
  size_t get_dtensor_guid();

public:
  yirage::kernel::DTensor dtensor;
  int3 input_map;
  int forloop_dim;
};

class TBOutputOp : public TBOperator {
public:
  TBOutputOp(Graph *_graph,
             STensor const &stensor,
             int3 output_map,
             int forloop_dim,
             yirage::type::TBEpilogueType allreduce);
  ~TBOutputOp();

  operator json() const override;
  size_t get_dtensor_guid();

public:
  yirage::kernel::DTensor dtensor;
  int3 output_map;
  // Note: forloop_dim is reserved for overlapping
  // communication and computation in multi-GPU runs
  // and should always be -1 for now
  int forloop_dim;
  yirage::type::TBEpilogueType epilogue;
};

} // namespace threadblock
} // namespace yirage

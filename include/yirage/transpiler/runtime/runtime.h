// runtime.h - Runtime for Program Generated by Yirage
#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

// The following two functions will be generated by the transpiler
static void _init();
static void _execute_mugraph(std::vector<void const *> input_tensors,
                             std::vector<void *> output_tensors,
                             void *buf,
                             cudaStream_t stream,
                             void *profiler_buffer);

// Runtime libraries
#include "config.h"
#include "kernel/element_binary.h"
#include "kernel/element_unary.h"
#include "kernel/matmul.h"
#include "kernel/reduction.h"
#include "threadblock/threadblock.h"
#include "utils.h"

// Entrypoint for C/C++
extern "C" void execute_mugraph(std::vector<void const *> input_tensors,
                                std::vector<void *> output_tensors,
                                void *buf,
                                cudaStream_t stream,
                                void *profiler_buffer) {

  static bool inited = false;
  if (!inited) {
    _init();
    inited = true;
  }
  _execute_mugraph(input_tensors, output_tensors, buf, stream, profiler_buffer);
}

// A wrappr around `execute_mugraph` which uses C arrays instead of vectors
// Entrypoint for Python
void execute_mugraph_wrapper(void const *input_tensors[],
                             size_t num_input_tensors,
                             void *output_tensors[],
                             size_t num_output_tensors,
                             void *buf,
                             cudaStream_t stream,
                             void *profiler_buffer) {
  std::vector<void const *> input_tensors_vec(
      input_tensors, input_tensors + num_input_tensors);
  std::vector<void *> output_tensors_vec(output_tensors,
                                         output_tensors + num_output_tensors);
  execute_mugraph(
      input_tensors_vec, output_tensors_vec, buf, stream, profiler_buffer);
}

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

#include "mirage/kernel/yica_all_reduce.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/utils/cuda_helper.h"
#include "mirage/utils/fingerprint_functions.h"
#include <cassert>
#include <cooperative_groups.h>

namespace mirage {
namespace kernel {

using namespace mirage::type;
using namespace mirage::config;
using namespace mirage::utils;
namespace cg = cooperative_groups;

/**
 * @brief YICA CIM内存内归约kernel
 */
template<typename T>
__global__ void yica_cim_allreduce_kernel(
    T* input_data,
    T* output_data,
    size_t num_elements,
    int num_cim_arrays,
    int cim_array_size) {
  
  // 使用cooperative groups进行CIM阵列间协调
  cg::grid_group grid = cg::this_grid();
  cg::thread_block block = cg::this_thread_block();
  
  int cim_id = blockIdx.x % num_cim_arrays;
  int elements_per_cim = (num_elements + num_cim_arrays - 1) / num_cim_arrays;
  
  // 计算当前CIM阵列处理的数据范围
  size_t start_idx = cim_id * elements_per_cim;
  size_t end_idx = min(start_idx + elements_per_cim, num_elements);
  
  // 共享内存用于CIM内部归约
  extern __shared__ T sdata[];
  
  // 每个线程处理多个元素进行预归约
  T local_sum = T(0);
  for (size_t i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
    local_sum += input_data[i];
  }
  
  // 将结果存储到共享内存
  sdata[threadIdx.x] = local_sum;
  __syncthreads();
  
  // CIM内部树状归约
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
    __syncthreads();
  }
  
  // 将每个CIM的结果写入全局内存
  if (threadIdx.x == 0) {
    output_data[cim_id] = sdata[0];
  }
  
  // 等待所有CIM完成本地归约
  grid.sync();
  
  // 第一个CIM负责最终的跨CIM归约
  if (cim_id == 0 && threadIdx.x < num_cim_arrays) {
    T final_sum = T(0);
    for (int i = 0; i < num_cim_arrays; i++) {
      final_sum += output_data[i];
    }
    
    // 广播最终结果到所有位置
    if (threadIdx.x == 0) {
      for (size_t i = 0; i < num_elements; i++) {
        output_data[i] = final_sum;
      }
    }
  }
}

/**
 * @brief YICA层次化归约kernel
 */
template<typename T>
__global__ void yica_hierarchical_allreduce_kernel(
    T* data,
    size_t num_elements,
    int hierarchy_level,
    int reduction_factor) {
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = 1 << hierarchy_level;
  
  if (tid < num_elements && tid % (stride * reduction_factor) == 0) {
    T sum = data[tid];
    
    // 在当前层级进行归约
    for (int i = 1; i < reduction_factor && tid + i * stride < num_elements; i++) {
      sum += data[tid + i * stride];
    }
    
    // 将结果写回
    data[tid] = sum;
    
    // 将结果复制到参与归约的所有位置
    for (int i = 1; i < reduction_factor && tid + i * stride < num_elements; i++) {
      data[tid + i * stride] = sum;
    }
  }
}

/**
 * @brief YICA SPM缓冲优化的AllReduce kernel
 */
template<typename T>
__global__ void yica_spm_buffered_allreduce_kernel(
    T* input_data,
    T* output_data,
    T* spm_buffer,
    size_t num_elements,
    size_t spm_buffer_size,
    bool enable_double_buffering) {
  
  cg::thread_block block = cg::this_thread_block();
  
  size_t elements_per_chunk = spm_buffer_size / sizeof(T);
  size_t num_chunks = (num_elements + elements_per_chunk - 1) / elements_per_chunk;
  
  for (size_t chunk = 0; chunk < num_chunks; chunk++) {
    size_t chunk_start = chunk * elements_per_chunk;
    size_t chunk_end = min(chunk_start + elements_per_chunk, num_elements);
    size_t chunk_size = chunk_end - chunk_start;
    
    // 确定当前使用的SPM缓冲区
    T* current_buffer = spm_buffer;
    if (enable_double_buffering) {
      current_buffer = spm_buffer + (chunk % 2) * elements_per_chunk;
    }
    
    // 将数据加载到SPM缓冲区
    for (size_t i = threadIdx.x; i < chunk_size; i += blockDim.x) {
      current_buffer[i] = input_data[chunk_start + i];
    }
    block.sync();
    
    // 在SPM中进行归约
    for (int stride = chunk_size / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride && threadIdx.x + stride < chunk_size) {
        current_buffer[threadIdx.x] += current_buffer[threadIdx.x + stride];
      }
      block.sync();
    }
    
    // 将结果写回全局内存
    if (threadIdx.x == 0) {
      T chunk_result = current_buffer[0];
      for (size_t i = chunk_start; i < chunk_end; i++) {
        output_data[i] = chunk_result;
      }
    }
  }
}

bool YICAAllReduceOp::run() {
  mirage::kernel::DeviceMemoryManager *dmm = 
      mirage::kernel::DeviceMemoryManager::get_instance();
  
  // 获取输入输出数据指针
  float *input_ptr = reinterpret_cast<float*>(
      dmm->data_base_ptr[0] + input_tensors[0].data_offset);
  float *output_ptr = reinterpret_cast<float*>(
      dmm->data_base_ptr[0] + output_tensors[0].data_offset);
  
  size_t num_elements = input_tensors[0].num_elements();
  
  // 根据配置选择执行策略
  switch (yica_config.reduction_strategy) {
    case ReductionStrategy::CIM_PARALLEL: {
      // 使用CIM并行归约
      int num_blocks = yica_config.num_cim_arrays;
      int threads_per_block = 256;
      size_t shmem_size = threads_per_block * sizeof(float);
      
      void* args[] = {
        &input_ptr, &output_ptr, &num_elements, 
        &yica_config.num_cim_arrays, &yica_config.cim_array_size
      };
      
      checkCUDA(cudaLaunchCooperativeKernel(
        (void*)yica_cim_allreduce_kernel<float>,
        num_blocks, threads_per_block, args, shmem_size));
      break;
    }
    
    case ReductionStrategy::HIERARCHICAL_TREE: {
      // 使用层次化树状归约
      for (int level = 0; level < hierarchical_plan_.total_levels; level++) {
        int num_blocks = (num_elements + 255) / 256;
        int threads_per_block = 256;
        
        yica_hierarchical_allreduce_kernel<float><<<num_blocks, threads_per_block>>>(
          inplace ? input_ptr : output_ptr,
          num_elements,
          level,
          2  // 二叉树归约
        );
        checkCUDA(cudaDeviceSynchronize());
      }
      break;
    }
    
    case ReductionStrategy::SPM_BUFFERED: {
      // 使用SPM缓冲优化
      float *spm_buffer = reinterpret_cast<float*>(
          dmm->data_base_ptr[0] + spm_buffer_plan_.chunk_size);  // 临时SPM地址
      
      int num_blocks = 1;  // 单块处理以充分利用SPM
      int threads_per_block = 256;
      
      yica_spm_buffered_allreduce_kernel<float><<<num_blocks, threads_per_block>>>(
        input_ptr, output_ptr, spm_buffer,
        num_elements, yica_config.spm_buffer_size,
        yica_config.enable_spm_double_buffering
      );
      break;
    }
    
    default:
      // 回退到标准实现
      return false;
  }
  
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

/**
 * @brief YICA AllReduce fingerprint计算
 */
__global__ void compute_yica_allreduce_fingerprint(
    mirage::utils::FpPointerList fp_ptr_list,
    int num_gpus,
    int num_elements,
    int num_cim_arrays) {
  
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < num_elements) {
    // 模拟CIM内存内归约的fingerprint计算
    FPType cim_results[32];  // 假设最多32个CIM阵列
    
    // 每个CIM计算本地结果
    for (int cim_id = 0; cim_id < num_cim_arrays; cim_id++) {
      FPType local_sum = 0;
      
      // 模拟CIM内部的并行归约
      for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        int element_idx = (i + cim_id * num_elements / num_cim_arrays) % num_elements;
        local_sum = compute_add_fingerprint(local_sum, fp_ptr_list.ptrs[gpu_id][element_idx]);
      }
      
      cim_results[cim_id] = local_sum;
    }
    
    // 跨CIM归约
    FPType final_result = 0;
    for (int cim_id = 0; cim_id < num_cim_arrays; cim_id++) {
      final_result = compute_add_fingerprint(final_result, cim_results[cim_id]);
    }
    
    // 将最终结果写入所有GPU
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
      fp_ptr_list.ptrs[gpu_id][i] = final_result;
    }
  }
}

bool YICAAllReduceOp::fingerprint(void) {
  // 断言1-D GPU网格
  assert(kgraph->gpu_dim.y == 1);
  assert(kgraph->gpu_dim.z == 1);
  
  assert(input_tensors[0].num_elements() == output_tensors[0].num_elements());
  int num_elements = input_tensors[0].num_elements();
  
  int const num_threads_per_blk = 256;
  int num_blocks = (num_elements + num_threads_per_blk - 1) / num_threads_per_blk;
  
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  
  // 使用GPU dmm->gpu_id计算fingerprint
  checkCUDA(cudaSetDevice(dmm->gpu_id));
  
  mirage::utils::FpPointerList fp_ptr_list;
  for (int gpu_id = 0; gpu_id < kgraph->gpu_dim.x; gpu_id++) {
    fp_ptr_list.ptrs[gpu_id] = reinterpret_cast<mirage::type::FPType *>(
        dmm->fp_base_ptr[gpu_id] + input_tensors[0].fp_offset);
  }
  
  compute_yica_allreduce_fingerprint<<<num_blocks, num_threads_per_blk>>>(
      fp_ptr_list, 
      kgraph->gpu_dim.x, 
      num_elements,
      yica_config.num_cim_arrays);
  
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

size_t YICAAllReduceOp::get_owner_independent_hash() const {
  size_t hash = KNOperator::get_owner_independent_hash();
  hash_combine(hash, inplace);
  hash_combine(hash, yica_config.reduction_strategy);
  hash_combine(hash, yica_config.num_cim_arrays);
  return hash;
}

} // namespace kernel
} // namespace mirage 
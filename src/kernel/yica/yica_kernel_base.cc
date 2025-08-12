/* Copyright 2024-2025 YICA
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

#include "yirage/kernel/yica_kernel_base.h"
#include <iostream>

namespace yirage {
namespace kernel {

YICAKernelBase::YICAKernelBase(Graph* graph, const YICAHardwareConfig& config)
    : graph_(graph), hw_config_(config), execution_time_ms_(0.0) {
    
    std::cout << "🔧 初始化 YICA Kernel 基类..." << std::endl;
    std::cout << "📊 CIM 阵列数量: " << hw_config_.num_cim_arrays << std::endl;
    std::cout << "📊 SPM 大小: " << hw_config_.spm_size_mb << "MB" << std::endl;
    std::cout << "📊 向量宽度: " << hw_config_.vector_width << std::endl;
    std::cout << "✅ YICA Kernel 基类初始化完成" << std::endl;
}

YICAKernelBase::~YICAKernelBase() {
    std::cout << "🧹 清理 YICA Kernel 基类资源" << std::endl;
}







} // namespace kernel
} // namespace yirage
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

// CPU版本的YICA代码生成器头文件 - 无CUDA依赖

#include <string>
#include <vector>
#include <memory>

namespace yirage {
namespace kernel {
    struct Graph;
}

namespace search {
namespace yica {

// 生成的文件结构
struct GeneratedFile {
    std::string filename;
    std::string content;
    std::string file_type;
    size_t size_bytes;
    std::string description;
};

// 代码生成结果
struct GenerationResult {
    bool success = false;
    std::vector<GeneratedFile> generated_files;
    std::string compilation_commands;
    std::string generation_log;
    float estimated_performance_gain = 1.0f;
};

// CPU版本的代码生成器接口
GenerationResult generate_yica_cpu_code(const kernel::Graph& graph);

} // namespace yica
} // namespace search
} // namespace yirage 
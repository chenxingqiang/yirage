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

#include "yirage/kernel/operator.h"
#include "yirage/kernel/device_tensor.h"
#include "yirage/profile_result.h"
#include <vector>

namespace yirage {
namespace kernel {

// YICA MatMul Operator - 继承 KNOperator 以融入 Yirage 体系
class KNYICAMatMulOp : public KNOperator {
public:
    KNYICAMatMulOp(Graph* graph, 
                   const DTensor& A, 
                   const DTensor& B,
                   bool transpose_a = false,
                   bool transpose_b = false);
    virtual ~KNYICAMatMulOp();

    // KNOperator 接口实现
    bool profile(ProfileResult& result) override;
    bool fingerprint(void) override;
    operator json() const override;

private:
    bool transpose_a_, transpose_b_;
    float alpha_ = 1.0f, beta_ = 0.0f;
    double execution_time_ms_ = 0.0;
    
    // YICA 特定方法
    bool execute_yica_matmul();
};

// YICA Element-wise Operations Operator
class KNYICAElementOp : public KNOperator {
public:
    enum class OpType {
        ADD, MUL, RELU, EXP, GELU, FUSED_MUL_ADD
    };

    // Unary constructor
    KNYICAElementOp(Graph* graph, const DTensor& input, OpType op_type);
    // Binary constructor  
    KNYICAElementOp(Graph* graph, const DTensor& input1, const DTensor& input2, OpType op_type);
    // Ternary constructor (FMA)
    KNYICAElementOp(Graph* graph, const DTensor& a, const DTensor& b, const DTensor& c, OpType op_type);
    virtual ~KNYICAElementOp();

    // KNOperator 接口实现
    bool profile(ProfileResult& result) override;
    bool fingerprint(void) override;
    operator json() const override;

private:
    OpType op_type_;
    size_t num_operands_;
    double execution_time_ms_ = 0.0;
    
    // YICA 特定方法
    bool execute_yica_element_ops();
};

// YICA RMS Normalization Operator
class KNYICARMSNormOp : public KNOperator {
public:
    KNYICARMSNormOp(Graph* graph, 
                    const DTensor& input,
                    const std::vector<int>& normalized_shape,
                    float epsilon = 1e-5f);
    KNYICARMSNormOp(Graph* graph,
                    const DTensor& input,
                    const DTensor& weight,
                    const std::vector<int>& normalized_shape,
                    float epsilon = 1e-5f);
    virtual ~KNYICARMSNormOp();

    // KNOperator 接口实现
    bool profile(ProfileResult& result) override;
    bool fingerprint(void) override;
    operator json() const override;

private:
    float epsilon_;
    std::vector<int> normalized_shape_;
    bool has_weight_;
    double execution_time_ms_ = 0.0;
    
    // YICA 特定方法
    bool execute_yica_rms_norm();
};

// YICA AllReduce Operator
class KNYICAAllReduceOp : public KNOperator {
public:
    enum class ReductionOp {
        SUM, MEAN, MAX, MIN, PROD
    };

    KNYICAAllReduceOp(Graph* graph,
                      const DTensor& input,
                      ReductionOp op_type = ReductionOp::SUM,
                      bool inplace = false);
    virtual ~KNYICAAllReduceOp();

    // KNOperator 接口实现
    bool profile(ProfileResult& result) override;
    bool fingerprint(void) override;
    operator json() const override;

private:
    ReductionOp reduction_op_;
    bool inplace_;
    double execution_time_ms_ = 0.0;
    
    // YICA 特定方法
    bool execute_yica_all_reduce();
};

} // namespace kernel
} // namespace yirage

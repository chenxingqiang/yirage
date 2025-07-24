#pragma once

#include <memory>
#include <vector>
#include <deque>
#include <algorithm>
#include "runtime_types.h"

namespace yirage {
namespace search {
namespace yica {

// 神经网络层接口
class NeuralLayer {
public:
    virtual ~NeuralLayer() = default;
    virtual std::vector<float> forward(const std::vector<float>& input) = 0;
    virtual void backward(const std::vector<float>& grad_output, 
                         std::vector<float>& grad_input) = 0;
    virtual void update_weights(float learning_rate) = 0;
    virtual size_t get_output_size() const = 0;
};

// 全连接层
class DenseLayer : public NeuralLayer {
private:
    std::vector<std::vector<float>> weights_;  // [output_size][input_size]
    std::vector<float> biases_;               // [output_size]
    std::vector<std::vector<float>> weight_gradients_;
    std::vector<float> bias_gradients_;
    std::vector<float> last_input_;
    size_t input_size_;
    size_t output_size_;
    
    // 激活函数
    enum class Activation {
        NONE,
        RELU,
        SIGMOID,
        TANH
    };
    
    Activation activation_;
    
    // 应用激活函数
    float apply_activation(float x) const;
    float activation_derivative(float x) const;
    
public:
    DenseLayer(size_t input_size, size_t output_size, 
               Activation activation = Activation::RELU);
    
    std::vector<float> forward(const std::vector<float>& input) override;
    void backward(const std::vector<float>& grad_output, 
                  std::vector<float>& grad_input) override;
    void update_weights(float learning_rate) override;
    size_t get_output_size() const override;
    
    // 初始化权重
    void initialize_weights();
    
    // 获取参数数量
    size_t get_parameter_count() const;
};

// LSTM层
class LSTMLayer : public NeuralLayer {
private:
    struct LSTMCell {
        // 权重矩阵 [hidden_size x (input_size + hidden_size)]
        std::vector<std::vector<float>> Wf, Wi, Wo, Wg;  // forget, input, output, candidate
        std::vector<float> bf, bi, bo, bg;                // biases
        
        // 梯度
        std::vector<std::vector<float>> Wf_grad, Wi_grad, Wo_grad, Wg_grad;
        std::vector<float> bf_grad, bi_grad, bo_grad, bg_grad;
        
        // 状态
        std::vector<float> h_prev, c_prev;  // 前一时刻的隐藏状态和细胞状态
        std::vector<float> h_curr, c_curr;  // 当前时刻的隐藏状态和细胞状态
        
        // 中间计算结果（用于反向传播）
        std::vector<float> f_gate, i_gate, o_gate, g_gate;
        std::vector<float> tanh_c;
    };
    
    size_t input_size_;
    size_t hidden_size_;
    size_t sequence_length_;
    std::vector<LSTMCell> cells_;  // 每个时间步一个cell
    
    // 辅助函数
    float sigmoid(float x) const;
    float tanh_func(float x) const;
    std::vector<float> matrix_vector_multiply(
        const std::vector<std::vector<float>>& matrix,
        const std::vector<float>& vector) const;
    
public:
    LSTMLayer(size_t input_size, size_t hidden_size, size_t sequence_length);
    
    std::vector<float> forward(const std::vector<float>& input) override;
    void backward(const std::vector<float>& grad_output, 
                  std::vector<float>& grad_input) override;
    void update_weights(float learning_rate) override;
    size_t get_output_size() const override;
    
    // 重置状态
    void reset_state();
    
    // 初始化权重
    void initialize_weights();
};

// 性能预测模型
class PerformancePredictionModel {
private:
    std::vector<std::unique_ptr<NeuralLayer>> layers_;
    float learning_rate_ = 0.001f;
    size_t batch_size_ = 32;
    
    // 训练数据缓存
    std::deque<std::pair<TimeSeriesFeatures, PerformanceMetrics>> training_data_;
    size_t max_training_data_ = 10000;
    
    // 预处理
    TimeSeriesFeatures normalize_features(const TimeSeriesFeatures& features) const;
    PerformanceMetrics denormalize_metrics(const PerformanceMetrics& metrics) const;
    
    // 归一化参数
    struct NormalizationParams {
        std::vector<float> feature_mean, feature_std;
        std::vector<float> target_mean, target_std;
        bool initialized = false;
    };
    
    mutable NormalizationParams norm_params_;
    
    // 计算损失
    float compute_loss(const PerformanceMetrics& predicted, 
                      const PerformanceMetrics& actual) const;
    
    // 反向传播
    void backpropagate(const PerformanceMetrics& predicted, 
                      const PerformanceMetrics& actual);
    
public:
    PerformancePredictionModel();
    ~PerformancePredictionModel();
    
    // 构建网络结构
    void build_network(size_t input_size, 
                      const std::vector<size_t>& hidden_sizes,
                      size_t output_size);
    
    // 预测性能
    PerformanceMetrics predict_performance(
        const TimeSeriesFeatures& features,
        const YICAConfig& proposed_config);
    
    // 批量预测
    std::vector<PerformanceMetrics> predict_batch(
        const std::vector<TimeSeriesFeatures>& features_batch,
        const std::vector<YICAConfig>& configs_batch);
    
    // 添加训练数据
    void add_training_data(const TimeSeriesFeatures& features,
                          const PerformanceMetrics& actual_performance);
    
    // 在线更新模型
    void update_model(const TimeSeriesFeatures& features,
                     const PerformanceMetrics& actual_performance);
    
    // 批量训练
    void train_batch(int epochs = 100);
    
    // 保存和加载模型
    bool save_model(const std::string& filename) const;
    bool load_model(const std::string& filename);
    
    // 获取模型统计
    struct ModelStats {
        size_t training_samples = 0;
        float average_loss = 0.0f;
        float validation_accuracy = 0.0f;
        size_t total_parameters = 0;
    };
    
    ModelStats get_model_stats() const;
    
    // 设置超参数
    void set_learning_rate(float lr);
    void set_batch_size(size_t batch_size);
    
    // 重置模型
    void reset();
};

// 在线学习系统
class OnlineLearning {
private:
    // 经验回放缓冲区
    struct Experience {
        TimeSeriesFeatures features;
        YICAConfig config;
        PerformanceMetrics actual_metrics;
        PerformanceMetrics predicted_metrics;
        float prediction_error;
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
    };
    
    std::deque<Experience> experience_buffer_;
    size_t max_buffer_size_ = 50000;
    
    // 自适应学习率
    float base_learning_rate_ = 0.001f;
    float current_learning_rate_ = 0.001f;
    float lr_decay_factor_ = 0.95f;
    int steps_since_lr_update_ = 0;
    int lr_update_interval_ = 1000;
    
    // 任务适应
    struct TaskAdapter {
        std::string task_type;
        std::vector<float> task_weights;
        float adaptation_rate = 0.01f;
        int sample_count = 0;
    };
    
    std::unordered_map<std::string, TaskAdapter> task_adapters_;
    
    // 重要性采样
    std::vector<float> compute_importance_weights(
        const std::vector<Experience>& experiences) const;
    
    // 选择训练样本
    std::vector<Experience> select_training_samples(size_t batch_size) const;
    
    // 检测分布漂移
    bool detect_distribution_drift(const std::vector<Experience>& recent_data) const;
    
public:
    OnlineLearning();
    
    // 添加经验
    void add_experience(const TimeSeriesFeatures& features,
                       const YICAConfig& config,
                       const PerformanceMetrics& actual_metrics,
                       const PerformanceMetrics& predicted_metrics);
    
    // 在线更新
    void online_update(PerformancePredictionModel& model);
    
    // 多任务适应
    void adapt_to_task(const std::string& task_type,
                      PerformancePredictionModel& model);
    
    // 增量学习
    void incremental_learning(PerformancePredictionModel& model,
                             const std::vector<Experience>& new_data);
    
    // 遗忘机制
    void apply_forgetting_mechanism();
    
    // 获取学习统计
    struct LearningStats {
        size_t total_experiences = 0;
        float average_prediction_error = 0.0f;
        float current_learning_rate = 0.0f;
        size_t active_tasks = 0;
        bool distribution_drift_detected = false;
    };
    
    LearningStats get_learning_stats() const;
    
    // 设置参数
    void set_buffer_size(size_t size);
    void set_learning_rate(float lr);
    void set_lr_decay_factor(float factor);
    
    // 重置学习系统
    void reset();
};

// 参数调优器
class ParameterTuner {
private:
    // 参数空间定义
    struct ParameterSpace {
        std::string name;
        float min_value;
        float max_value;
        float current_value;
        float best_value;
        std::vector<float> history;
    };
    
    std::vector<ParameterSpace> parameter_spaces_;
    
    // 贝叶斯优化
    struct GaussianProcess {
        std::vector<std::vector<float>> X;  // 输入参数
        std::vector<float> y;               // 性能值
        float noise_variance = 0.01f;
        float length_scale = 1.0f;
        float signal_variance = 1.0f;
        
        // 核函数
        float rbf_kernel(const std::vector<float>& x1, 
                        const std::vector<float>& x2) const;
        
        // 预测均值和方差
        std::pair<float, float> predict(const std::vector<float>& x) const;
        
        // 更新GP
        void update(const std::vector<float>& x, float y);
    };
    
    GaussianProcess gp_;
    
    // 采集函数
    float expected_improvement(const std::vector<float>& x, 
                              float best_y_so_far) const;
    
    // 优化采集函数
    std::vector<float> optimize_acquisition_function() const;
    
    // 评估参数配置
    float evaluate_parameter_configuration(const std::vector<float>& params,
                                          PerformancePredictionModel& model);
    
public:
    ParameterTuner();
    
    // 定义参数空间
    void define_parameter_space(const std::string& name,
                               float min_val, float max_val, float initial_val);
    
    // 自动调优
    std::vector<float> auto_tune_parameters(
        PerformancePredictionModel& model,
        int max_iterations = 50);
    
    // 网格搜索
    std::vector<float> grid_search(PerformancePredictionModel& model,
                                  int grid_points_per_dim = 10);
    
    // 随机搜索
    std::vector<float> random_search(PerformancePredictionModel& model,
                                    int num_samples = 100);
    
    // 获取最佳参数
    std::vector<float> get_best_parameters() const;
    
    // 获取调优历史
    std::vector<std::vector<float>> get_tuning_history() const;
    
    // 重置调优器
    void reset();
};

// 机器学习优化器主类
class MLOptimizer {
private:
    std::unique_ptr<PerformancePredictionModel> prediction_model_;
    std::unique_ptr<OnlineLearning> online_learning_;
    std::unique_ptr<ParameterTuner> parameter_tuner_;
    
    // 策略推荐缓存
    std::unordered_map<size_t, std::pair<YICAConfig, float>> recommendation_cache_;
    
    // 异常检测
    struct AnomalyDetectionModel {
        std::vector<PerformanceMetrics> normal_patterns_;
        float anomaly_threshold = 2.0f;  // 标准差倍数
        
        bool is_anomaly(const PerformanceMetrics& metrics) const;
        void update_patterns(const PerformanceMetrics& metrics);
    };
    
    AnomalyDetectionModel anomaly_model_;
    
    // 模型集成
    std::vector<std::unique_ptr<PerformancePredictionModel>> ensemble_models_;
    
    // 集成预测
    PerformanceMetrics ensemble_predict(
        const TimeSeriesFeatures& features,
        const YICAConfig& config);
    
public:
    MLOptimizer();
    ~MLOptimizer();
    
    // 初始化ML优化器
    bool initialize(size_t feature_size, const std::vector<size_t>& hidden_sizes);
    
    // 性能预测
    PerformanceMetrics predict_performance(
        const TimeSeriesFeatures& features,
        const YICAConfig& proposed_config);
    
    // 策略推荐
    std::vector<YICAConfig> recommend_strategies(
        const TimeSeriesFeatures& features,
        const std::vector<YICAConfig>& candidate_configs,
        int top_k = 5);
    
    // 参数自动调优
    YICAConfig auto_tune_parameters(
        const YICAConfig& base_config,
        const TimeSeriesFeatures& features);
    
    // 异常检测和处理
    bool detect_performance_anomaly(const PerformanceMetrics& metrics);
    std::vector<YICAConfig> handle_anomaly(
        const PerformanceMetrics& anomaly_metrics,
        const YICAConfig& current_config);
    
    // 在线学习更新
    void update_models(const TimeSeriesFeatures& features,
                      const PerformanceMetrics& actual_performance);
    
    // 多任务适应
    void adapt_to_workload(const std::string& workload_type,
                          const std::vector<PerformanceMetrics>& sample_data);
    
    // 获取模型置信度
    float get_prediction_confidence(const TimeSeriesFeatures& features,
                                   const YICAConfig& config);
    
    // 保存和加载优化器
    bool save_optimizer(const std::string& filename) const;
    bool load_optimizer(const std::string& filename);
    
    // 获取ML统计
    struct MLStats {
        size_t total_predictions = 0;
        float average_prediction_error = 0.0f;
        size_t training_samples = 0;
        float model_confidence = 0.0f;
        size_t anomalies_detected = 0;
    };
    
    MLStats get_ml_stats() const;
    
    // 重置优化器
    void reset();
};

}  // namespace yica
}  // namespace search
}  // namespace yirage 
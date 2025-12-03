# 基于预训练模型的文本分类与可解释性分析

本项目实现了基于DistilBERT的文本分类任务，使用LoRA进行参数高效微调，并提供注意力可视化、SHAP和LIME等可解释性分析工具。

## 项目特点

- **轻量级模型**: 使用DistilBERT，参数量远小于BERT-base，训练和推理速度更快
- **参数高效微调**: 采用LoRA技术，只需训练少量参数即可达到良好效果
- **完整评估**: 提供准确率、F1分数、精确率、召回率等评估指标
- **可解释性分析**: 集成注意力可视化、SHAP和LIME等多种可解释性工具

## 项目结构

```
distilbert/
├── pytorch_model.bin      # 预训练模型权重（已有）
├── config.py              # 配置文件
├── data_loader.py         # 数据加载和预处理
├── model_utils.py         # 模型工具函数（LoRA配置等）
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── explainability.py      # 可解释性分析脚本
├── main.py                # 主运行脚本
├── requirements.txt       # 依赖包列表
└── README.md             # 项目说明文档
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 主要依赖包

- torch >= 2.0.0
- transformers >= 4.30.0
- datasets >= 2.12.0
- peft >= 0.4.0 (用于LoRA)
- shap >= 0.42.0 (用于SHAP分析)
- lime >= 0.2.0 (用于LIME分析)

## 使用方法

### 1. 配置参数

编辑 `config.py` 文件，可以修改以下参数：

- **模型配置**: 模型名称、路径等
- **数据集配置**: 数据集名称、批次大小、最大长度等
- **LoRA配置**: rank、alpha、dropout等
- **训练配置**: 学习率、训练轮数、保存步数等

### 2. 运行完整流程

```bash
# 运行所有步骤（训练、评估、可解释性分析）
python main.py --mode all --dataset imdb
```

### 3. 分步运行

#### 训练模型

```bash
python main.py --mode train --dataset imdb
# 或直接运行
python train.py
```

#### 评估模型

```bash
python main.py --mode evaluate --dataset imdb --model_path ./saved_models/final_model
# 或直接运行
python evaluate.py
```

#### 可解释性分析

```bash
python main.py --mode explain --dataset imdb --model_path ./saved_models/final_model
# 或直接运行
python explainability.py
```

## 数据集

### IMDb电影评论数据集

本项目默认使用IMDb电影评论数据集进行情感分析（二分类：正面/负面）。

数据集会自动从Hugging Face下载，无需手动准备。

### 其他数据集

如需使用其他数据集（如THUCNews中文新闻分类），需要：

1. 在 `data_loader.py` 中实现对应的数据加载函数
2. 修改 `config.py` 中的 `DATASET_NAME` 参数

## 输出结果

### 训练输出

- 模型保存在 `./saved_models/final_model/`
- 训练日志保存在 `./logs/`（可用TensorBoard查看）
- 训练过程中的检查点保存在 `./output/`

### 评估输出

- 评估指标（准确率、F1分数等）打印在控制台
- 混淆矩阵图片保存在 `./output/confusion_matrix.png`
- 分类报告打印在控制台

### 可解释性分析输出

所有可解释性分析结果保存在 `./output/explainability/`：

- `attention_visualization.png`: 注意力权重热力图
- `attention_visualization_bar.png`: Token重要性条形图
- `shap_analysis.png`: SHAP分析结果
- `lime_explanation_*.html`: LIME解释结果（HTML格式）

## 模型可解释性分析说明

### 1. 注意力可视化

- 展示模型在做出分类决策时，对输入文本中不同位置的关注程度
- 通过热力图和条形图直观展示重要词汇

### 2. SHAP分析

- 使用SHAP值量化每个token对最终预测的贡献
- 可以识别对分类结果影响最大的词语

### 3. LIME分析

- 通过局部可解释性模型解释单个样本的预测结果
- 生成HTML格式的交互式解释报告

## 性能优化建议

1. **使用GPU**: 如果有GPU，训练速度会显著提升
2. **调整批次大小**: 根据显存大小调整 `BATCH_SIZE`
3. **LoRA参数**: 可以尝试不同的 `LORA_R` 和 `LORA_ALPHA` 值
4. **混合精度训练**: 代码已自动启用FP16（如果使用GPU）

## 常见问题

### Q: 如何修改分类任务的类别数？

A: 修改 `config.py` 中的相关配置，并在 `data_loader.py` 中确保标签数量正确。

### Q: 如何使用自己的数据集？

A: 在 `data_loader.py` 中添加数据加载函数，参考 `load_imdb_dataset()` 的实现。

### Q: LoRA参数如何选择？

A: 一般 `LORA_R` 选择 4-16，`LORA_ALPHA` 通常是 `LORA_R` 的2倍。可以尝试不同组合找到最佳效果。

### Q: 可解释性分析很慢怎么办？

A: 减少 `config.py` 中的 `NUM_SAMPLES_FOR_EXPLAIN` 参数，只分析少量样本。

## 参考文献

- DistilBERT: [Sanh et al., 2019](https://arxiv.org/abs/1910.01108)
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- SHAP: [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)
- LIME: [Ribeiro et al., 2016](https://arxiv.org/abs/1602.04938)

## 许可证

本项目仅供学习和研究使用。




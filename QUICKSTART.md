# 快速开始指南

## 1. 环境准备

### 安装依赖

```bash
pip install -r requirements.txt
```

### 检查GPU（可选但推荐）

```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
```

## 2. 快速运行

### 方式一：一键运行所有步骤

```bash
python main.py --mode all --dataset imdb
```

这将依次执行：
1. 训练模型（使用LoRA微调）
2. 评估模型性能
3. 进行可解释性分析

### 方式二：分步运行

#### 步骤1：训练模型

```bash
python train.py
```

训练完成后，模型会保存在 `./saved_models/final_model/`

#### 步骤2：评估模型

```bash
python evaluate.py
```

评估结果包括：
- 准确率、F1分数、精确率、召回率
- 混淆矩阵图片

#### 步骤3：可解释性分析

```bash
python explainability.py
```

分析结果保存在 `./output/explainability/`：
- 注意力可视化图片
- SHAP分析结果
- LIME解释HTML文件

## 3. 自定义配置

编辑 `config.py` 可以修改：

- **训练参数**：学习率、批次大小、训练轮数等
- **LoRA参数**：rank、alpha、dropout等
- **数据集**：切换不同的数据集

## 4. 查看结果

### 训练日志

使用TensorBoard查看训练过程：

```bash
tensorboard --logdir ./logs
```

然后在浏览器中打开 `http://localhost:6006`

### 评估结果

- 控制台输出：详细的评估指标
- `./output/confusion_matrix.png`：混淆矩阵可视化

### 可解释性结果

- `./output/explainability/attention_visualization.png`：注意力权重热力图
- `./output/explainability/shap_analysis.png`：SHAP分析结果
- `./output/explainability/lime_explanation_*.html`：LIME交互式解释

## 5. 常见问题排查

### 问题1：CUDA内存不足

**解决方案**：
- 减小 `BATCH_SIZE`（在 `config.py` 中）
- 减小 `MAX_LENGTH`
- 使用CPU训练（会自动切换）

### 问题2：LoRA目标模块错误

**解决方案**：
- 代码已包含自动检测功能
- 如果仍有问题，检查模型结构：
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("distilbert-base-uncased")
print(model)
```

### 问题3：数据集下载失败

**解决方案**：
- 检查网络连接
- 使用代理或镜像源
- 手动下载数据集到本地

### 问题4：SHAP/LIME分析很慢

**解决方案**：
- 减少 `NUM_SAMPLES_FOR_EXPLAIN`（在 `config.py` 中）
- 只分析少量样本

## 6. 性能优化建议

1. **使用GPU**：训练速度提升10-50倍
2. **混合精度训练**：自动启用FP16（GPU）
3. **调整批次大小**：根据显存调整
4. **LoRA参数调优**：
   - `LORA_R=8`：平衡性能和效率
   - `LORA_R=4`：更快但可能性能略低
   - `LORA_R=16`：更好性能但训练更慢

## 7. 下一步

- 尝试不同的数据集
- 调整超参数进行实验
- 分析错误案例，改进模型
- 探索更多的可解释性方法




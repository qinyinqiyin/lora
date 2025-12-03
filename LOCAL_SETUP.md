# 本地运行配置说明

## 修改内容

项目已修改为支持完全本地运行，无需访问Hugging Face：

### 1. 数据集加载（data_loader.py）
- ✅ 支持从本地CSV文件加载IMDB数据集
- ✅ 数据集路径：`./IMDB/`
- ✅ 支持文件：`Train.csv`, `Test.csv`, `Valid.csv`
- ✅ CSV格式：`text,label`（第一行为表头）

### 2. 模型加载（model_utils.py）
- ✅ 优先从本地加载模型权重：`./pytorch_model.bin`
- ✅ 如果本地没有模型结构，会从Hugging Face加载结构（仅首次）
- ✅ 后续运行会优先使用本地权重

### 3. 配置文件（config.py）
新增配置项：
- `DATASET_PATH = "./IMDB"` - 本地数据集路径
- `USE_LOCAL_DATASET = True` - 是否使用本地数据集
- `MODEL_PATH = "./pytorch_model.bin"` - 本地模型权重路径

## 使用方法

### 1. 确保文件结构正确

```
distilbert/
├── pytorch_model.bin      # 模型权重文件
├── IMDB/                  # 数据集目录
│   ├── Train.csv         # 训练集
│   ├── Test.csv          # 测试集
│   └── Valid.csv         # 验证集（可选）
├── config.py
├── data_loader.py
├── model_utils.py
└── ...
```

### 2. 运行项目

```bash
# 激活conda环境
conda activate distilbert_nlp

# 运行完整流程
python main.py --mode all
```

## 注意事项

1. **首次运行**：如果本地没有模型配置和分词器，会从Hugging Face下载（仅首次）
2. **后续运行**：完全使用本地文件，无需网络连接
3. **模型权重**：确保 `pytorch_model.bin` 文件存在且完整
4. **数据集格式**：CSV文件必须包含 `text` 和 `label` 两列

## 故障排查

### 问题1：找不到数据集文件
- 检查 `IMDB/` 目录是否存在
- 检查CSV文件名是否正确（Train.csv, Test.csv）

### 问题2：模型加载失败
- 检查 `pytorch_model.bin` 文件是否存在
- 检查文件是否完整（文件大小应该很大）

### 问题3：分词器加载失败
- 首次运行需要从Hugging Face下载分词器
- 后续运行会使用缓存，无需网络



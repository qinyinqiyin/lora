# GPU设置指南

## 当前状态

✅ **NVIDIA驱动已安装** - 驱动版本: 576.52  
✅ **GPU硬件正常** - NVIDIA GeForce RTX 4050  
✅ **CUDA版本** - 12.9  
❌ **PyTorch版本** - 当前是CPU版本 (2.8.0+cpu)

## 问题诊断

你的系统有GPU硬件和驱动，但PyTorch安装的是CPU版本，所以无法使用GPU加速。

## 解决方案

### 方法一：使用pip安装（推荐）

1. **卸载CPU版本的PyTorch**
```bash
pip uninstall torch torchvision torchaudio
```

2. **安装CUDA版本的PyTorch**

根据你的CUDA 12.9，可以安装CUDA 12.1或12.4版本的PyTorch（向下兼容）：

**CUDA 12.1版本（推荐）:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 12.4版本:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 方法二：使用conda安装

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 方法三：使用提供的脚本（Windows）

直接运行：
```bash
install_gpu_pytorch.bat
```

## 验证安装

安装完成后，运行检测脚本验证：

```bash
python check_gpu.py
```

如果看到以下信息，说明GPU已可用：
```
✅ GPU检测完成，GPU可用！
```

## 使用GPU训练

安装完成后，训练脚本会自动使用GPU。你可以通过以下方式确认：

```python
import torch
print(torch.cuda.is_available())  # 应该输出 True
print(torch.cuda.get_device_name(0))  # 应该显示你的GPU名称
```

## 常见问题

### Q: 安装后仍然检测不到GPU？
A: 
1. 重启Python环境或IDE
2. 重启计算机
3. 确认安装的是CUDA版本（不是CPU版本）

### Q: 如何确认安装的是CUDA版本？
A: 运行以下命令：
```python
import torch
print(torch.__version__)  # 应该包含 'cu' 而不是 'cpu'
print(torch.version.cuda)  # 应该显示CUDA版本号
```

### Q: 显存不足怎么办？
A: 
- 减小批次大小（BATCH_SIZE）
- 减小最大序列长度（MAX_LENGTH）
- 使用梯度累积

## 参考链接

- PyTorch官方安装指南: https://pytorch.org/get-started/locally/
- CUDA兼容性: https://pytorch.org/get-started/previous-versions/


@echo off
echo ============================================================
echo 安装CUDA版本的PyTorch
echo ============================================================
echo.
echo 检测到你的系统:
echo - GPU: NVIDIA GeForce RTX 4050
echo - CUDA版本: 12.9
echo.
echo 正在卸载CPU版本的PyTorch...
pip uninstall torch torchvision torchaudio -y

echo.
echo 正在安装CUDA 12.1版本的PyTorch...
echo (CUDA 12.1兼容CUDA 12.9)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo ============================================================
echo 安装完成！
echo ============================================================
echo.
echo 请运行以下命令验证GPU是否可用:
echo python check_gpu.py
echo.
pause


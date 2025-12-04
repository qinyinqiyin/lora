# -*- coding: utf-8 -*-
"""
GPU检测和诊断脚本
"""
import sys

def check_gpu():
    """检查GPU可用性"""
    print("="*60)
    print("GPU检测和诊断")
    print("="*60)
    
    # 1. 检查PyTorch是否安装
    print("\n1. 检查PyTorch安装...")
    try:
        import torch
        print(f"   ✅ PyTorch版本: {torch.__version__}")
    except ImportError:
        print("   ❌ PyTorch未安装")
        print("   解决方案: pip install torch")
        return False
    
    # 2. 检查CUDA是否可用
    print("\n2. 检查CUDA可用性...")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA可用: {cuda_available}")
    
    if not cuda_available:
        print("\n   ⚠️  CUDA不可用，可能的原因：")
        print("   - 系统没有NVIDIA GPU")
        print("   - 未安装NVIDIA GPU驱动程序")
        print("   - 未安装CUDA工具包")
        print("   - 安装的PyTorch版本不支持CUDA（CPU版本）")
        
        # 检查PyTorch版本信息
        print("\n3. 检查PyTorch构建信息...")
        print(f"   PyTorch版本: {torch.__version__}")
        print(f"   编译时CUDA版本: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'None'}")
        
        # 检查是否安装了CUDA版本的PyTorch
        if torch.version.cuda is None:
            print("\n   ❌ 当前安装的是CPU版本的PyTorch")
            print("\n   解决方案:")
            print("   1. 卸载当前PyTorch:")
            print("      pip uninstall torch torchvision torchaudio")
            print("\n   2. 安装CUDA版本的PyTorch:")
            print("      访问 https://pytorch.org/get-started/locally/")
            print("      选择你的CUDA版本，例如CUDA 11.8:")
            print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("\n   3. 或者使用conda安装:")
            print("      conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
        else:
            print("\n   ⚠️  PyTorch支持CUDA，但当前无法检测到GPU")
            print("\n   可能的原因和解决方案:")
            print("   1. 检查NVIDIA驱动:")
            print("      - Windows: 打开设备管理器 -> 显示适配器，查看是否有NVIDIA GPU")
            print("      - 运行: nvidia-smi (如果命令不存在，说明驱动未安装)")
            print("\n   2. 检查CUDA工具包:")
            print("      - 运行: nvcc --version")
            print("      - 如果不存在，需要安装CUDA工具包")
            print("\n   3. 检查GPU是否被其他程序占用")
            print("\n   4. 重启计算机后重试")
        
        return False
    
    # 3. GPU详细信息
    print("\n3. GPU详细信息:")
    print(f"   GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"   名称: {torch.cuda.get_device_name(i)}")
        print(f"   显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"   计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # 4. 测试GPU计算
    print("\n4. 测试GPU计算...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("   ✅ GPU计算测试成功")
        print(f"   结果张量设备: {z.device}")
    except Exception as e:
        print(f"   ❌ GPU计算测试失败: {e}")
        return False
    
    # 5. 显存信息
    print("\n5. 显存使用情况:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}:")
        print(f"   已分配: {allocated:.2f} GB")
        print(f"   已保留: {reserved:.2f} GB")
        print(f"   总量: {total:.2f} GB")
        print(f"   可用: {total - reserved:.2f} GB")
    
    print("\n" + "="*60)
    print("✅ GPU检测完成，GPU可用！")
    print("="*60)
    return True


def check_nvidia_driver():
    """检查NVIDIA驱动"""
    print("\n检查NVIDIA驱动...")
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ NVIDIA驱动已安装")
            print("\nnvidia-smi输出:")
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi命令执行失败")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi命令不存在，可能未安装NVIDIA驱动")
        return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi命令超时")
        return False
    except Exception as e:
        print(f"❌ 检查驱动时出错: {e}")
        return False


if __name__ == "__main__":
    # 检查NVIDIA驱动
    driver_ok = check_nvidia_driver()
    
    # 检查PyTorch GPU
    gpu_ok = check_gpu()
    
    # 总结
    print("\n" + "="*60)
    print("诊断总结")
    print("="*60)
    print(f"NVIDIA驱动: {'✅ 正常' if driver_ok else '❌ 未检测到'}")
    print(f"PyTorch GPU: {'✅ 可用' if gpu_ok else '❌ 不可用'}")
    
    if not gpu_ok:
        print("\n建议操作步骤:")
        print("1. 确认你的电脑有NVIDIA GPU")
        print("2. 安装/更新NVIDIA驱动: https://www.nvidia.com/Download/index.aspx")
        print("3. 安装CUDA版本的PyTorch: https://pytorch.org/get-started/locally/")
        print("4. 重启计算机")
        print("5. 重新运行此脚本检查")


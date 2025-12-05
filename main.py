# -*- coding: utf-8 -*-
"""
主运行脚本：整合训练、评估和可解释性分析

主要功能：
- main(): 统一入口，支持训练、评估、可解释性分析的单独或组合运行

调用关系：
- 调用 train.train() (train.py第42行) 进行模型训练
- 调用 evaluate.evaluate_model() (evaluate.py第18行) 进行模型评估
- 调用 explainability.analyze_explainability() (explainability.py第228行) 进行可解释性分析
"""
import argparse
import os
from config import Config  # 配置文件
from train import train  # 训练模块，定义在train.py第42行
from evaluate import evaluate_model  # 评估模块，定义在evaluate.py第18行
from explainability import analyze_explainability  # 可解释性模块，定义在explainability.py第228行


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于DistilBERT的文本分类与可解释性分析')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'evaluate', 'explain', 'all'],
                       help='运行模式: train(训练), evaluate(评估), explain(可解释性), all(全部)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径（用于评估和可解释性分析）')
    parser.add_argument('--dataset', type=str, default='imdb',
                       choices=['imdb'],
                       help='数据集名称')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    Config.create_dirs()
    
    # 设置数据集
    Config.DATASET_NAME = args.dataset
    
    if args.mode == 'train' or args.mode == 'all':
        print("="*50)
        print("开始训练模型...")
        print("="*50)
        train()  # 调用train.py第42行定义的train函数
    
    if args.mode == 'evaluate' or args.mode == 'all':
        print("\n" + "="*50)
        print("开始评估模型...")
        print("="*50)
        model_path = args.model_path or os.path.join(Config.SAVE_MODEL_DIR, "final_model")
        if not os.path.exists(model_path):
            print(f"警告: 模型路径不存在 {model_path}")
            print("请先运行训练模式")
        else:
            # 调用evaluate.py第18行定义的evaluate_model函数
            evaluate_model(model_path, args.dataset)
    
    if args.mode == 'explain' or args.mode == 'all':
        print("\n" + "="*50)
        print("开始可解释性分析...")
        print("="*50)
        model_path = args.model_path or os.path.join(Config.SAVE_MODEL_DIR, "final_model")
        if not os.path.exists(model_path):
            print(f"警告: 模型路径不存在 {model_path}")
            print("请先运行训练模式")
        else:
            # 调用explainability.py第228行定义的analyze_explainability函数
            analyze_explainability(model_path, args.dataset)
    
    print("\n" + "="*50)
    print("所有任务完成！")
    print("="*50)


if __name__ == "__main__":
    main()



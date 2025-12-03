# -*- coding: utf-8 -*-
"""
主运行脚本：整合训练、评估和可解释性分析
"""
import argparse
import os
from config import Config
from train import train
from evaluate import evaluate_model
from explainability import analyze_explainability


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
        train()
    
    if args.mode == 'evaluate' or args.mode == 'all':
        print("\n" + "="*50)
        print("开始评估模型...")
        print("="*50)
        model_path = args.model_path or os.path.join(Config.SAVE_MODEL_DIR, "final_model")
        if not os.path.exists(model_path):
            print(f"警告: 模型路径不存在 {model_path}")
            print("请先运行训练模式")
        else:
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
            analyze_explainability(model_path, args.dataset)
    
    print("\n" + "="*50)
    print("所有任务完成！")
    print("="*50)


if __name__ == "__main__":
    main()



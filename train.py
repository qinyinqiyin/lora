# -*- coding: utf-8 -*-
"""
模型训练脚本：使用LoRA进行参数高效微调
"""
import os
import torch
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from config import Config
from data_loader import prepare_datasets
from model_utils import get_model_for_training


def compute_metrics(eval_pred):
    """
    计算评估指标
    
    Args:
        eval_pred: 评估预测结果
        
    Returns:
        dict: 包含准确率、F1分数等的字典
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train():
    """主训练函数"""
    # 创建输出目录
    Config.create_dirs()
    
    # 准备数据集
    tokenizer, train_dataset, val_dataset, test_dataset = prepare_datasets(
        Config.DATASET_NAME
    )
    
    # 加载模型
    num_labels = len(set([item['labels'].item() for item in train_dataset]))
    model = get_model_for_training(num_labels=num_labels)
    
    # 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,  # 评估时使用更大的批次
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,  # 梯度累积
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_steps=Config.WARMUP_STEPS,
        logging_dir=Config.LOG_DIR,
        logging_steps=Config.LOGGING_STEPS,
        save_steps=Config.SAVE_STEPS,
        eval_steps=Config.EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),  # 如果使用GPU，启用混合精度训练
        dataloader_num_workers=0,  # 数据加载器工作进程数（Windows上设为0避免多进程问题）
        dataloader_pin_memory=False,  # Windows上禁用pin_memory
        remove_unused_columns=True,  # 移除未使用的列
    )
    
    # 如果设置了最大训练样本数，限制训练集大小
    if Config.MAX_TRAIN_SAMPLES and Config.MAX_TRAIN_SAMPLES > 0:
        print(f"限制训练样本数为: {Config.MAX_TRAIN_SAMPLES}")
        import random
        indices = list(range(len(train_dataset)))
        random.seed(42)
        random.shuffle(indices)
        selected_indices = indices[:Config.MAX_TRAIN_SAMPLES]
        train_dataset = torch.utils.data.Subset(train_dataset, selected_indices)
        print(f"实际训练样本数: {len(train_dataset)}")
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    final_model_path = os.path.join(Config.SAVE_MODEL_DIR, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"模型已保存到: {final_model_path}")
    
    # 在测试集上评估
    print("在测试集上评估...")
    test_results = trainer.evaluate(test_dataset)
    print("测试集结果:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    return trainer, test_results


if __name__ == "__main__":
    train()



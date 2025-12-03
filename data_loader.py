# -*- coding: utf-8 -*-
"""
数据加载和预处理模块
"""
import os
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
from config import Config


class TextClassificationDataset(Dataset):
    """文本分类数据集类"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            dict: 包含input_ids, attention_mask, labels, text的字典
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
        }
        # 注意：text字段不包含在返回的字典中，因为DataCollator无法处理字符串
        # 如果需要原始文本，可以通过索引访问 self.texts[idx]
        return result


def load_imdb_dataset():
    """
    加载IMDb电影评论数据集（从本地CSV文件）
    
    Returns:
        train_dataset, test_dataset: 训练集和测试集
    """
    if Config.USE_LOCAL_DATASET:
        print("正在从本地加载IMDb数据集...")
        train_path = os.path.join(Config.DATASET_PATH, Config.TRAIN_FILE)
        test_path = os.path.join(Config.DATASET_PATH, Config.TEST_FILE)
        
        # 检查文件是否存在
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"训练集文件不存在: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"测试集文件不存在: {test_path}")
        
        # 读取CSV文件
        print(f"读取训练集: {train_path}")
        train_df = pd.read_csv(train_path)
        print(f"读取测试集: {test_path}")
        test_df = pd.read_csv(test_path)
        
        # 提取文本和标签
        train_texts = train_df['text'].tolist()
        train_labels = train_df['label'].tolist()
        test_texts = test_df['text'].tolist()
        test_labels = test_df['label'].tolist()
        
        print(f"训练集大小: {len(train_texts)}")
        print(f"测试集大小: {len(test_texts)}")
        
        return (train_texts, train_labels), (test_texts, test_labels)
    else:
        # 从Hugging Face加载（备用方案）
        print("正在从Hugging Face加载IMDb数据集...")
        from datasets import load_dataset
        dataset = load_dataset("imdb")
        
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']
        
        print(f"训练集大小: {len(train_texts)}")
        print(f"测试集大小: {len(test_texts)}")
        
        return (train_texts, train_labels), (test_texts, test_labels)


def load_thucnews_dataset():
    """
    加载THUCNews中文新闻数据集
    注意：需要先下载THUCNews数据集到本地
    
    Returns:
        train_dataset, test_dataset: 训练集和测试集
    """
    # 这里需要根据实际的THUCNews数据格式进行加载
    # 示例代码，需要根据实际情况修改
    print("THUCNews数据集加载功能需要根据实际数据格式实现")
    raise NotImplementedError("THUCNews数据集加载需要根据实际数据格式实现")


def prepare_datasets(dataset_name="imdb"):
    """
    准备数据集
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        tokenizer, train_dataset, val_dataset, test_dataset: 分词器和数据集
    """
    # 加载分词器（尝试从本地加载，如果失败则从Hugging Face加载）
    print(f"正在加载分词器: {Config.MODEL_NAME}")
    try:
        # 尝试从本地加载（如果有本地模型目录）
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, local_files_only=True)
        print("从本地加载分词器成功")
    except:
        # 如果本地没有，从Hugging Face加载
        print("本地未找到分词器，从Hugging Face加载...")
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # 加载数据集
    if dataset_name.lower() == "imdb":
        (train_texts, train_labels), (test_texts, test_labels) = load_imdb_dataset()
        
        # 检查是否有本地验证集
        valid_path = os.path.join(Config.DATASET_PATH, Config.VALID_FILE)
        if Config.USE_LOCAL_DATASET and os.path.exists(valid_path):
            print(f"读取验证集: {valid_path}")
            valid_df = pd.read_csv(valid_path)
            val_texts = valid_df['text'].tolist()
            val_labels = valid_df['label'].tolist()
            print(f"验证集大小: {len(val_texts)}")
        else:
            val_texts = None
            val_labels = None
    elif dataset_name.lower() == "thucnews":
        (train_texts, train_labels), (test_texts, test_labels) = load_thucnews_dataset()
        val_texts = None
        val_labels = None
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 创建数据集对象
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, Config.MAX_LENGTH
    )
    test_dataset = TextClassificationDataset(
        test_texts, test_labels, tokenizer, Config.MAX_LENGTH
    )
    
    # 处理验证集
    if val_texts is not None and val_labels is not None:
        # 使用本地验证集
        val_dataset = TextClassificationDataset(
            val_texts, val_labels, tokenizer, Config.MAX_LENGTH
        )
    else:
        # 从训练集中划分验证集
        train_size = int((1 - Config.VALIDATION_SPLIT) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    return tokenizer, train_dataset, val_dataset, test_dataset


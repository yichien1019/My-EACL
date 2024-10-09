import copy
import functools
import json
import logging
import multiprocessing
import operator
import os
import pickle
import random
import time
import timeit
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import OrderedDict as odict
from typing import Optional
import tempfile
import shutil
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import vocab
# from kmeans_pytorch import kmeans
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
# 創建一個記錄器，用來記錄程式的運行情況
logger = logging.getLogger(__name__)
# 設置日誌的基本配置，將日誌級別設置為 INFO
logging.basicConfig(level=logging.INFO)
# 忽略未來的警告
warnings.simplefilter(action='ignore', category=FutureWarning)
CONFIG = {
    'bert_path': 'princeton-nlp/sup-simcse-roberta-large',  # 預訓練 BERT 模型路徑
    'epochs' : 10,                                          # 訓練的總輪數
    'lr' : 1e-3,                                            # 標準參數的學習率
    'ptmlr' : 1e-5,                                         # 預訓練模型的學習率
    'batch_size' : 32,                                      # 批次大小
    'max_len' : 256,                                        # 文本最大長度
    'bert_dim' : 1024,                                      # BERT 模型輸出嵌入向量的維度
    'pad_value' : 1,                                        # padding 的值
    'mask_value' : 2,                                       # mask 的值
    'dropout' : 0.1,                                        # dropout 機率
    'pool_size': 512,                                       # 訓練過程中使用的支持集大小
    'support_set_size': 64,                                 # 支持集的大小，用於度量學習
    'num_classes' : 7,                                      # 類別數量，情緒分類有 7 類
    'warm_up' : 128,                                        # 用於學習率預熱的步驟數
    'dist_func': 'cosine',                                  # 使用餘弦相似度作為距離函數
    'data_path' : './MELD',                                 # 數據集的路徑
    'accumulation_steps' : 1,                               # 梯度累積步驟
    'avg_cluster_size' : 4096,                              # 平均聚類大小
    'max_step' : 1024,                                      # 最大訓練步驟
    'num_positive': 1,                                      # 每個樣本正樣本的數量
    'ratio':1,                                              # 控制負樣本與正樣本的比率
    'mu':0.5,                                               # 用於控制對比學習的損失比例
    'cl':True,                                              # 是否啟用對比學習
    'temperature': 0.08,                                    # 溫度參數，用於 softmax
    'fgm': False,                                           # 是否啟用對抗訓練
    'train_obj': 'psup',                                    # 訓練目標，定義任務類型
    'speaker_vocab' : '',                                   # 說話者詞彙表
    'emotion_vocab' : '',                                   # 情緒詞彙表
    'temp_path': '',                                        # 臨時文件路徑
    'ngpus' : torch.cuda.device_count(),                    # 可用的 GPU 數量
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 設置運行設備（GPU 或 CPU）
}
# 計算兩個向量之間的餘弦距離
def dist(x, y):
    # 餘弦相似度範圍為 [-1, 1]，距離為 1 - 相似度 / 2，並加上微小值防止數值下溢
    return (1-F.cosine_similarity(x, y, dim=-1))/2 + 1e-8
# 計算兩個向量之間的分數（餘弦相似度轉換為 [0, 1] 範圍）
def score_func(x, y):
    # 餘弦相似度的結果轉換到 [0, 1] 範圍
    return (1+F.cosine_similarity(x, y, dim=-1))/2 + 1e-8
# 設置隨機種子，用於結果的可重現性
def set_seed(seed):
    random.seed(seed)  # 設置 Python 隨機數種子
    np.random.seed(seed)  # 設置 numpy 隨機數種子
    torch.manual_seed(seed)  # 設置 PyTorch 隨機數種子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 如果有多個 GPU，設置所有 GPU 的隨機數種子
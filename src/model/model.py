import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F

class CLModel(nn.Module):
    def __init__(self, args, n_classes, tokenizer=None):
        super().__init__()
        self.args = args                # 保存初始化傳入的參數
        self.dropout = args.dropout     # 使用的丟棄比例
        self.num_classes = n_classes    # 分類任務中的類別數量
        self.pad_value = args.pad_value # 批次中句子不同長度的填充部分
        self.mask_value = 50265         # 定義 [mask] 標記的 ID
        # 加載預訓練的 Transformer 模型
        self.f_context_encoder = AutoModel.from_pretrained(args.bert_path)  
        # 獲取 Transformer 模型的詞嵌入矩陣的形狀，保存維度信息
        num_embeddings, self.dim = self.f_context_encoder.embeddings.word_embeddings.weight.data.shape
        # 初始化平均距離列表
        self.avg_dist = []              
        # 增加詞嵌入矩陣的大小，添加 256 個新嵌入
        self.f_context_encoder.resize_token_embeddings(num_embeddings + 256)
        # 防止計算過程中出現除以零的情況
        self.eps = 1e-8                 
        # 根據程序的配置來選擇模型運行的設備
        self.device = "cuda" if self.args.cuda else "cpu"    
        # 定義分類器，將全連接層將特徵映射到指定的類別數量
        self.predictor = nn.Sequential(
            nn.Linear(self.dim, self.num_classes)
        )
        # 定義映射函數，將隱藏狀態映射到較低的維度
        self.map_function = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim), 
            nn.ReLU(),
            nn.Linear(self.dim, args.mapping_lower_dim),
        ).to(self.device)
        # tokenizer : 將文本數據轉換為可供計算機處理的數字形式
        self.tokenizer = tokenizer
        # 根據數據集名稱從指定路徑加載對應的情感錨點和情感標籤
        if args.dataset_name == "IEMOCAP":
            self.emo_anchor = torch.load(f"{args.anchor_path}/iemocap_emo.pt").to(self.device)
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5]).to(self.device)
        elif args.dataset_name == "MELD":
            self.emo_anchor = torch.load(f"{args.anchor_path}/meld_emo.pt").to(self.device)
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        elif args.dataset_name == "EmoryNLP":
            self.emo_anchor = torch.load(f"{args.anchor_path}/emorynlp_emo.pt").to(self.device)
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6]).to(self.device)
    # 返回模型運行的設備信息
    def device(self):
        return self.f_context_encoder.device
    # 計算相似度分數(餘弦相似度)
    def score_func(self, x, y):
        # 分數在 0 和 1 之間，eps 確保在計算中不會遇到零相似度的問題
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + self.eps 
    # 模型的內部計算函數，用於執行模型的前向傳播計算
    def _forward(self, sentences):
        # 生成遮罩 mask 在模型中用來區分填充部分和有效數據
        mask = 1 - (sentences == (self.pad_value)).long()
        # 通過預訓練的 Transformer 模型獲取句子的最後一層隱藏狀態
        utterance_encoded = self.f_context_encoder(
            input_ids=sentences,    # 輸入的句子張量
            attention_mask=mask,    # 指定哪些標記應該被注意，哪些應該被忽略
            output_hidden_states=True,  # 返回模型隱藏狀態
            return_dict=True    # 返回字典格式的結果
        )['last_hidden_state']  # 模型輸出最後一層的隱藏狀態
        # 找到句子中 [mask] 標記的位置
        mask_pos = (sentences == (self.mask_value)).long().max(1)[1]
        # 從隱藏狀態中提取對應於 [mask] 標記的位置的向量
        mask_outputs = utterance_encoded[torch.arange(mask_pos.shape[0]), mask_pos] 
        # 通過映射函數將隱藏狀態映射到低維度空間
        mask_mapped_outputs = self.map_function(mask_outputs)   
        # 對特徵進行隨機丟棄，以減少過擬合
        feature = torch.dropout(mask_outputs, self.dropout, train=self.training)   
        # 使用分類器對特徵進行分類
        feature = self.predictor(feature)  
        # 使用最近鄰機制來計算樣本與情感錨點之間的相似度分數
        if self.args.use_nearest_neighbour:
            # 將情感錨點映射到低維度空間
            anchors = self.map_function(self.emo_anchor)
            # 保存最近的情感錨點
            self.last_emo_anchor = anchors
            # 計算樣本與錨點之間的相似度分數
            anchor_scores = self.score_func(mask_mapped_outputs.unsqueeze(1), anchors.unsqueeze(0))    
        else:
            anchor_scores = None
        # 返回模型的輸出特徵、映射後的 [mask] 標記隱藏狀態、原始的 [mask] 標記隱藏狀態、情感錨點的相似度分數。
        return feature, mask_mapped_outputs, mask_outputs, anchor_scores
    # 模型的前向傳播函數，用於生成每輪對話的特徵表示
    def forward(self, sentences, return_mask_output=False):
        # 調用 _forward 方法以計算模型的輸出
        feature, mask_mapped_outputs, mask_outputs, anchor_scores = self._forward(sentences)
        
        if return_mask_output:
            return feature, mask_mapped_outputs, mask_outputs, anchor_scores
        else:
            return feature  
class Classifier(nn.Module):
    def __init__(self, args, anchors) -> None:
        super(Classifier, self).__init__()
        self.weight = nn.Parameter(anchors) # 初始化模型的錨點權重作為參數
        self.args = args    
    # 計算相似度分數(餘弦相似度)
    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + 1e-8
    def forward(self, emb):
        # self.weight : [N, D] -> [1, N, D] (N 是情感錨點的數量，D 是特徵維度)
        # emb : [B, D] -> [B, 1, D] (B 是樣本的數量，D 是特徵維度)
        # 計算的結果會是形狀為 [B, N] 的相似度矩陣，表示每個樣本特徵與所有情感錨點的相似度
        return self.score_func(self.weight.unsqueeze(0), emb.unsqueeze(1)) / self.args.temp
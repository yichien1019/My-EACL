from config import *
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
@dataclass          # 定義資料類別
class HybridLossOutput:
    ce_loss:torch.Tensor = None # Cross Entropy 損失的 Tensor
    cl_loss:torch.Tensor = None # Contrastive Learning 損失的 Tensor
    sentiment_representations:torch.Tensor = None   # 與情緒相關的表示的 Tensor
    sentiment_labels:torch.Tensor = None            # 情緒標籤的 Tensor
    sentiment_anchortypes:torch.Tensor = None       # 情緒錨點類型的 Tensor
    anchortype_labels:torch.Tensor = None           # 錨點類型標籤的 Tensor
    max_cosine:torch.Tensor = None                  # 最大的餘弦相似度的 Tensor

# 混合損失(CrossEntropyLoss & 監督式對比損失SupConLoss)
def loss_function(log_prob, reps, label, mask, model):
    # 定義 Cross-Entropy 損失函數
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1).to(reps.device)   
    # 定義對比學習損失函數，傳遞模型的參數
    scl_loss_fn = SupConLoss(model.args)    
    # 計算對比學習損失。如果模型處於訓練模式，則不返回情緒表示
    cl_loss = scl_loss_fn(reps, label, model, return_representations=not model.training)
    # 計算 Cross-Entropy 損失，忽略無效的索引（由 mask 指定）
    ce_loss = ce_loss_fn(log_prob[mask], label[mask])
    # 通過 HybridLossOutput 類型來打包並返回多個損失和中間結果
    return HybridLossOutput(
        ce_loss=ce_loss,
        cl_loss=cl_loss.loss,
        sentiment_representations=cl_loss.sentiment_representations,
        sentiment_labels=cl_loss.sentiment_labels,
        sentiment_anchortypes=cl_loss.sentiment_anchortypes,
        anchortype_labels=cl_loss.anchortype_labels,
        max_cosine = cl_loss.max_cosine
    ) 

# 計算角度損失
def AngleLoss(means):
    # 計算錨點均值的全局平均值
    g_mean = means.mean(dim=0)  
    # 中心化每個錨點均值，將全局平均值從每個錨點中減去
    centered_mean = means - g_mean
    # 將中心化的錨點進行歐幾里德範數正規化
    means_ = F.normalize(centered_mean, p=2, dim=1)
    # 計算錨點之間的餘弦相似度
    cosine = torch.matmul(means_, means_.t())
    # 從對角線上減去兩倍的餘弦相似度，轉變為 -1 或其他非 1 的值，因對角線上的元素不再是 1 而排除自我相似性
    # torch.diag(torch.diag(cosine)) 創建了一個只包含對角線元素的矩陣，其他位置是 0
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))
    # 將最大餘弦相似度限制在 -0.99999 到 0.99999 之間
    max_cosine = cosine.max().clamp(-0.99999, 0.99999)
    # 計算損失，即最大餘弦相似度的反餘弦值的平均值
    loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
    return loss, max_cosine

@dataclass
class SupConOutput:
    loss:torch.Tensor = None                        # 訓練損失
    sentiment_representations:torch.Tensor = None   # 情緒表示
    sentiment_labels:torch.Tensor = None            # 情緒標籤
    sentiment_anchortypes:torch.Tensor = None       # 情緒錨點類型
    anchortype_labels:torch.Tensor = None           # 錨點類型標籤
    max_cosine:torch.Tensor = None                  # 最大的餘弦相似度

# 監督對比學習損失類別
class SupConLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.temperature = args.temp
        self.eps = 1e-8
        # 根據數據集加載不同的情緒錨點和標籤
        if args.dataset_name == "IEMOCAP":
            self.emo_anchor = torch.load(f"{args.anchor_path}/iemocap_emo.pt")
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5])
        elif args.dataset_name == "MELD":
            self.emo_anchor = torch.load(f"{args.anchor_path}/meld_emo.pt")
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        elif args.dataset_name == "EmoryNLP":
            self.emo_anchor = torch.load(f"{args.anchor_path}/emorynlp_emo.pt")
            self.emo_label = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        # 計算情緒錨點之間的相似度矩陣
        self.sim = nn.functional.cosine_similarity(self.emo_anchor.unsqueeze(1), self.emo_anchor.unsqueeze(0), dim=2)
        self.args = args
    # 計算兩個向量之間的相似度，將結果標準化到 [0, 1] 範圍內
    def score_func(self, x, y):
        return (1 + F.cosine_similarity(x, y, dim=-1))/2 + self.eps
    # 前向傳播函數
    def forward(self, reps, labels, model, return_representations=False):
        device = reps.device
        batch_size = reps.shape[0]  # 獲取當前批次的大小
        self.emo_anchor = self.emo_anchor.to(device)        # 將情緒錨點移動到相同的設備上
        self.emo_label = self.emo_label.to(device)          # 將情緒標籤移動到相同的設備上
        emo_anchor = model.map_function(self.emo_anchor)    # 使用模型的映射函數處理情緒錨點
        # 如果要求返回情緒表示，則記錄情緒標籤和表示
        if return_representations:
            sentiment_labels = labels
            sentiment_representations = reps.detach()
            sentiment_anchortypes = emo_anchor.detach()
        else:
            sentiment_labels = None
            sentiment_representations = None
            sentiment_anchortypes = None
        # 如果禁用了情緒錨點，僅使用表示進行計算
        if self.args.disable_emo_anchor:
            concated_reps = reps
            concated_labels = labels
            concated_bsz = batch_size
        else:
            # 將表示和情緒錨點串聯在一起
            concated_reps = torch.cat([reps, emo_anchor], dim=0)
            # 將標籤和情緒標籤串聯在一起
            concated_labels = torch.cat([labels, self.emo_label], dim=0)
            # 更新批次大小
            concated_bsz = batch_size + emo_anchor.shape[0]
        # mask1 和 mask2 的構建
        mask1 = concated_labels.unsqueeze(0).expand(concated_labels.shape[0], concated_labels.shape[0])
        mask2 = concated_labels.unsqueeze(1).expand(concated_labels.shape[0], concated_labels.shape[0])
        # 構建負樣本遮罩
        # torch.eye(concated_bsz) : 生成一個大小為 (concated_bsz, concated_bsz) 的單位矩陣
        mask = 1 - torch.eye(concated_bsz).to(reps.device)
        # 構建正樣本遮罩
        # pos_mask = [[1, 0, 1],[0, 1, 0],[1, 0, 1]] <1 表示相同類別的樣本，0 表示不同類別的樣本>
        pos_mask = (mask1 == mask2).long()
        # 擴展表示以進行逐元素比較
        rep1 = concated_reps.unsqueeze(0).expand(concated_bsz, concated_bsz, concated_reps.shape[-1])
        rep2 = concated_reps.unsqueeze(1).expand(concated_bsz, concated_bsz, concated_reps.shape[-1])
        # 計算表示之間的相似度分數
        scores = self.score_func(rep1, rep2)
        scores *= 1 - torch.eye(concated_bsz).to(scores.device)
        # 分數除以溫度參數進行縮放
        scores /= self.temperature
        # 擷取 concated_bsz (批次大小) 前數據
        scores = scores[:concated_bsz]
        pos_mask = pos_mask[:concated_bsz]
        mask = mask[:concated_bsz]
        # 減去分數中的最大值以穩定數值計算
        scores -= torch.max(scores).item()
        # 計算角度損失和最大餘弦相似度
        angleloss, max_cosine = AngleLoss(emo_anchor)
        # print(max_cosine)

        scores = torch.exp(scores)              # 將分數指數化
        pos_scores = scores * (pos_mask * mask) # 計算正樣本分數
        neg_scores = scores * (1 - pos_mask)    # 計算負樣本分數
        # 計算正樣本的概率 (將所有正樣本的分數除以正樣本和負樣本分數的總和)
        probs = pos_scores.sum(-1)/(pos_scores.sum(-1) + neg_scores.sum(-1))
        # 進一步標準化概率，並避免除以零
        probs /= (pos_mask * mask).sum(-1) + self.eps
        loss = - torch.log(probs + self.eps)    # 計算對比損失
        loss_mask = (loss > 0.0).long()         # 將損失中的正值設置為 1，負值設置為 0
        # 計算最終損失
        loss = (loss * loss_mask).sum() / (loss_mask.sum().item() + self.eps)
        # 加入角度損失，並按照預設權重進行縮放
        loss += self.args.angle_loss_weight * angleloss
        return SupConOutput(
            loss=loss,
            sentiment_representations=sentiment_representations,
            sentiment_labels=sentiment_labels,
            sentiment_anchortypes=sentiment_anchortypes,
            anchortype_labels=self.emo_label,
            max_cosine = max_cosine
        )
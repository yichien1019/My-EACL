import numpy as np
import torch
import torch.distributed as dist
# from dataloader import IEMOCAPDataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# 在訓練或評估模型計算每個批次(DataLoader)中的各種性能指標
def train_or_eval_model(model, loss_function, dataloader, epoch, device, args, optimizer=None, lr_scheduler=None, train=False):
    # 初始化列表來儲存損失值、預測結果、真實標籤
    losses, preds, labels = [], [], []
    sentiment_representations, sentiment_labels = [], []
    # 如果是訓練模式，必須有優化器
    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()
    # 判斷是否顯示進度條(tqdm)
    if args.disable_training_progress_bar:
        pbar = dataloader
    else:
        pbar = tqdm(dataloader)
    # 遍歷每個批次的資料
    for batch_id, batch in enumerate(pbar):
        input_ids, label = batch
        # 保存原始的輸入
        input_orig = input_ids
        # 初始化增強的輸入為 None
        input_aug = None
        # 如果使用混合精度運算，加速訓練
        if args.fp16:
            # 自動選擇正確的精度來加速運算
            with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                # 前向傳播計算損失和輸出
                loss, loss_output, log_prob, label, mask, anchor_scores = _forward(model, loss_function, input_orig, input_aug, label, device)
        else:
            loss, loss_output, log_prob, label, mask, anchor_scores = _forward(model, loss_function, input_orig, input_aug, label, device)
        # 選擇用錨點分數還是對數機率來計算預測
        if args.use_nearest_neighbour:
            # 使用錨點分數
            pred = torch.argmax(anchor_scores[mask], dim=-1)
        else:
            # 使用對數概率
            pred = torch.argmax(log_prob[mask], dim = -1)
        # 保存預測結果和真實標籤
        preds.append(pred)
        labels.append(label)
        losses.append(loss.item())
        # 如果是在訓練模式，進行反向傳播並更新模型權重
        if train:   # 訓練模式，進行梯度計算和更新
            loss.backward()
            # torch.nn.utils.clip_grad_norm_ : 梯度裁剪，防止梯度爆炸
            # clip_grad_norm_ : 來限制梯度的最大範圍(args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
            # 當 batch_id 整除 args.accumulation_step 時，才會進行參數的更新
            if batch_id % args.accumulation_step == 0:
                optimizer.step()        # 更新模型參數
                optimizer.zero_grad()   # 清空梯度
        else:       # 評估模式，僅保存模型的輸出
            # 保存情緒表徵和標籤
            sentiment_representations.append(loss_output.sentiment_representations)
            sentiment_labels.append(loss_output.sentiment_labels)
    # 如果有預測結果，計算相關指標
    if len(preds) != 0:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j,l in enumerate(label):
                # 如果標籤 l 不等於無效標籤(-1)，則提取該標籤和對應的預測值，而過濾掉無效標籤(l==-1)
                if l != -1:
                    new_labels.append(l.cpu().item())
                    new_preds.append(preds[i][j].cpu().item())
    else:
        # 如果沒有預測結果，返回 NaN
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []
    # 平均損失計算
    avg_loss = round(np.sum(losses) / len(losses), 4)
    # 平均準確率計算
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    # 最大餘弦相似度計算
    max_cosine = loss_output.max_cosine
    # 加權平均 F1-Score 分數計算
    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
    f1_scores = []
    new_labels = np.array(new_labels)
    new_preds = np.array(new_preds)
    # 確定不同資料集的類別數量
    if args.dataset_name in ['IEMOCAP']:
        n = 6
    else:
        n = 7
    # 分別計算每個類別的 F1 分數
    for class_id in range(n):
        true_label = []
        pred_label = []
        for i in range(len(new_labels)):
            # 如果真實標籤是 class_id
            if new_labels[i] == class_id:
                true_label.append(1)        # 加入 1 表示匹配
                # 如果預測標籤是 class_id
                if new_preds[i] == class_id:
                    pred_label.append(1)    # 加入 1 表示匹配
                else:
                    pred_label.append(0)    # 預測錯誤，加入 0 表示不匹配
            # 如果預測標籤是 class_id
            elif new_preds[i] == class_id:
                pred_label.append(1)        # 加入 1 表示匹配
                # 如果真實標籤是 class_id
                if new_labels[i] == class_id:
                    true_label.append(1)    # 加入 1 表示匹配
                else:
                    true_label.append(0)    # 預測錯誤，加入 0 表示不匹配
        # 計算 F1 分數
        f1 = round(f1_score(true_label, pred_label) * 100, 2)
        f1_scores.append(f1)
    # 返回計算結果
    return avg_loss, avg_accuracy, labels, preds, avg_fscore, f1_scores, max_cosine

# 前向傳播，用於計算損失和模型輸出
def _forward(model, loss_function, input_orig, input_aug, label, device):
    input_ids = input_orig.to(device)
    label = label.to(device)
    mask = torch.ones(len(input_orig)).to(device)
    mask = mask > 0.5       # 用於過濾掉無效的輸入
    # 判斷是否處於訓練模式
    if model.training:      # 訓練模式 : 進行梯度計算，並更新參數
        log_prob, masked_mapped_output, _, anchor_scores = model(input_ids, return_mask_output=True) 
        loss_output = loss_function(log_prob, masked_mapped_output, label, mask, model)
    else:                   # 評估模式 : 不會進行梯度計算
        with torch.no_grad():   
            log_prob, masked_mapped_output, _, anchor_scores = model(input_ids, return_mask_output=True) 
            loss_output = loss_function(log_prob, masked_mapped_output, label, mask, model)
    # 計算損失
    loss = loss_output.ce_loss * model.args.ce_loss_weight + (1 - model.args.ce_loss_weight) * loss_output.cl_loss
    # 返回計算結果
    return loss, loss_output, log_prob, label[mask], mask, anchor_scores

# 再次訓練模型
def retrain(model, loss_function, dataloader, epoch, device, args, optimizer=None, lr_scheduler=None, train=False):
    #  初始化相關變量
    losses, ce_losses, preds, labels = [], [], [], []  # delete []
    for batch in dataloader:
        data, label = batch
        data = data.to(device)
        label = label.to(device)
        if args.fp16:   # 使用混合精度
            with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                log_prob = model(data)  # 前向傳播
        # 計算損失
        loss = loss_function(log_prob, label)
        losses.append(loss.item())
        pred = torch.argmax(log_prob, dim = -1)
        preds.append(pred)
        labels.append(label)
        # 如果是訓練模式，進行反向傳播
        if train:
            loss.backward()
            # 從模型的對數概率輸出中找到最大概率對應的類別代表預測結果
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, norm_type=2)
            optimizer.step()        # 更新模型參數
            optimizer.zero_grad()   # 清空梯度
    # 計算性能指標(與 train_or_eval_model 類似)
    if len(preds) != 0:     # 檢查是否有預測結果（preds 非空）
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j,l in enumerate(label):
                if l != -1:
                    new_labels.append(l.cpu().item())
                    new_preds.append(preds[i][j].cpu().item())
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []
    # 計算平均損失
    avg_loss = round(np.sum(losses) / len(losses), 4)
    # 計算交叉熵損失的平均值
    avg_ce_loss = round(np.sum(ce_losses) / len(ce_losses), 4)
    # 計算預測結果的準確率
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    # 初始化 f1 分數的列表
    f1_scores = []
    # 計算加權平均的 F1 分數
    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
    new_labels = np.array(new_labels)
    new_preds = np.array(new_preds)
    # 確定不同資料集的類別數量
    if args.dataset_name in ['IEMOCAP']:
        n = 6
    else:
        n = 7
    # 迭代每個類別的情緒，計算每個類別的 f1 分數
    for class_id in range(n):
        true_label = [] # 用來存儲該類別的真實標籤
        pred_label = [] # 用來存儲該類別的預測標籤
        for i in range(len(new_labels)):
            # 如果真實標籤是 class_id
            if new_labels[i] == class_id:
                true_label.append(1)        # 加入 1 表示匹配
                # 如果預測標籤是 class_id
                if new_preds[i] == class_id:
                    pred_label.append(1)    # 加入 1 表示匹配
                else:
                    pred_label.append(0)    # 預測錯誤，加入 0 表示不匹配
            # 如果預測標籤是 class_id
            elif new_preds[i] == class_id:
                pred_label.append(1)        # 加入 1 表示匹配
                # 如果真實標籤是 class_id
                if new_labels[i] == class_id:
                    true_label.append(1)    # 加入 1 表示匹配
                else:
                    true_label.append(0)    # 預測錯誤，加入 0 表示不匹配
        # 計算該類別的 f1 分數，並將其轉換為百分比形式
        f1 = round(f1_score(true_label, pred_label) * 100, 2)
        # 將該類別的 f1 分數加入 f1_scores 列表
        f1_scores.append(f1)
    # 返回平均損失、交叉熵損失、準確率、真實標籤、預測結果、加權平均 F1 分數和每個類別的 f1 分數
    return avg_loss, avg_ce_loss, avg_accuracy, labels, preds, avg_fscore, f1_scores

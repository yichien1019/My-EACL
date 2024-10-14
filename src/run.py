import os
import gc
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
from trainer.trainer import  train_or_eval_model, retrain
from dataset import DialogueDataset
from torch.utils.data import DataLoader, sampler, TensorDataset
from transformers import AutoTokenizer
from torch.optim import AdamW
import copy
import warnings
warnings.filterwarnings("ignore")
import logging
from utils.data_process import *
from model.model import CLModel, Classifier
from model.loss import loss_function
import pickle
os.environ["TOKENIZERS_PARALLELISM"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # add
import numpy as np
# 日誌設置函數，用於創建文件和控制台的日誌輸出
def get_logger(filename, verbosity=1, name=None):
    # 設置不同日誌級別
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # 設置日誌格式(時間、文件名、行號、級別、日誌訊息)
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    # 創建日誌紀錄器
    logger = logging.getLogger(name)
    # 設定日誌紀錄器的級別
    logger.setLevel(level_dict[verbosity])
    # 文件處理器：創建一個處理器將日誌訊息寫入到指定的文件中
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # 流處理器：創建一個處理器將日誌訊息輸出到控制台
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def seed_everything(seed):
    random.seed(seed)                           # 設置 Python 的隨機數生成器
    np.random.seed(seed)                        # 設置 NumPy 的隨機數生成器
    torch.manual_seed(seed)                     # 設置 PyTorch 的隨機數生成器
    torch.cuda.manual_seed(seed)                # 設置 PyTorch 中 GPU (CUDA) 的隨機數生成器
    torch.cuda.manual_seed_all(seed)            # 如果使用多個 GPU，為每一個 GPU 設置隨機數生成器的種子
    torch.backends.cudnn.benchmark = False      # 禁用 cudnn 的自動最佳化搜尋
    torch.backends.cudnn.deterministic = True   # 設置 cudnn 以確保每次運行的結果是一致的

def get_paramsgroup(model, warmup=False):
    no_decay = ['bias', 'LayerNorm.weight']     # 不需要進行權重衰減的參數
    pre_train_lr = args.ptmlr                   # 預訓練模型的學習率
    # 獲取預訓練模型的參數的 ID 列表
    bert_params = list(map(id, model.f_context_encoder.parameters()))
    params = []
    warmup_params = []
    for name, param in model.named_parameters():
        lr = args.lr            # 學習率
        weight_decay = 0.01     # 權重衰減係數
        if id(param) in bert_params:            # 如果參數屬於預訓練模型的參數
            lr = pre_train_lr                   # 將學習率設置為預訓練模型的學習率
        if any(nd in name for nd in no_decay):  # 如果參數名稱中包含 'bias' 或 'LayerNorm.weight'
            weight_decay = 0                    # 不進行權重衰減
        # 將當前參數及其學習率和權重衰減添加到參數組列表中
        params.append({
            'params': param,
            'lr': lr,
            'weight_decay': weight_decay
        })
        # 構建 warmup 階段的參數組列表
        warmup_params.append({
            'params': param,
            'lr': args.ptmlr / 4 if id(param) in bert_params else lr,
            'weight_decay': weight_decay
        })
    if warmup:      # 如果是 warmup 階段，返回 warmup_params
        return warmup_params
    # 按照學習率對參數組列表進行排序
    params = sorted(params, key=lambda x: x['lr'])
    return params
# 配置命令行參數
def get_parser():
    parser = argparse.ArgumentParser()  # 初始化 ArgumentParser 用於解析命令行參數
    parser.add_argument('--bert_path', type=str, default='./pretrained/sup-simcse-roberta-large')
    parser.add_argument('--bert_dim', type = int, default=1024)
    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')
    parser.add_argument('--pad_value', type=int, default=1, help='padding')
    parser.add_argument('--mask_value', type=int, default=2, help='padding')
    parser.add_argument('--wp', type=int, default=8, help='past window size')
    parser.add_argument('--wf', type=int, default=0, help='future window size')
    parser.add_argument("--ce_loss_weight", type=float, default=0.1)    # 交叉熵損失的權重
    parser.add_argument("--angle_loss_weight", type=float, default=1.0) # 角度損失的權重
    parser.add_argument('--max_len', type=int, default=256,
                        help='max content length for each text, if set to 0, then the max length has no constrain')
    parser.add_argument("--temp", type=float, default=0.5)              # 用於調整 softmax 輸出分布的平滑度
    parser.add_argument('--accumulation_step', type=int, default=1)     # 累積步數
    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')   # 是否禁用 GPU
    parser.add_argument('--dataset_name', default='IEMOCAP', type= str, help='dataset name, IEMOCAP or MELD or EmoryNLP')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')      # 梯度裁剪的最大值
    parser.add_argument('--lr', type=float, default=4e-4, metavar='LR', help='learning rate')       # 學習率
    parser.add_argument('--ptmlr', type=float, default=1e-5, metavar='LR', help='learning rate')    # 預訓練模型的學習率
    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=8, metavar='E', help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=0, help='type of nodal attention')    # 權重衰減
    # 環境參數
    parser.add_argument("--fp16", type=bool, default=True)      # 是否使用 16 位浮點數進行訓練
    parser.add_argument("--seed", type=int, default=2)          # 隨機種子
    parser.add_argument("--ignore_prompt_prefix", action="store_true", default=True)    # 忽略提示詞的前綴
    parser.add_argument("--disable_training_progress_bar", action="store_true")         # 禁用訓練過程中的進度條顯示
    parser.add_argument("--mapping_lower_dim", type=int, default=1024)                  # 映射到的低維度大小
    # 消融實驗參數
    parser.add_argument("--disable_emo_anchor", action='store_true')    # 是否禁用情緒錨點
    parser.add_argument("--use_nearest_neighbour", action="store_true") # 是否使用最近鄰
    parser.add_argument("--disable_two_stage_training", action="store_true")    # 是否禁用兩階段訓練
    parser.add_argument("--stage_two_lr", default=1e-4, type=float)     # 指定第二階段的學習率
    parser.add_argument("--anchor_path", type=str)                      # 情緒錨點的路徑
    # 分析保存
    parser.add_argument("--save_stage_two_cache", action="store_true")  # 是否保存第二階段的緩存
    parser.add_argument("--save_path", default='./saved_models/', type=str)     # 保存模型的路徑
    args = parser.parse_args()      # 解析命令行參數並返回
    return args

if __name__ == '__main__':
    try:
        args = get_parser()     # 調用 get_parser 來解析命令行參數
        # 如果啟用了 fp16(半精度浮點數)，設置矩陣乘法的精度為中等，避免因為精度過低導致訓練不穩定的問題
        # 相比於標準的 32 位元浮點數(FP32)，能夠顯著減少存儲和計算需求
        if args.fp16:
            torch.set_float32_matmul_precision('medium')
        path = args.save_path   # 設置模型保存路徑，並確保保存目錄存在
        os.makedirs(os.path.join(path, args.dataset_name), exist_ok=True)
        # 設置隨機種子，保證結果的可重現性
        seed_everything(args.seed)
        # 檢查是否有可用的 GPU，並確定是否使用 GPU
        args.cuda = torch.cuda.is_available() and not args.no_cuda
        # 清理 GPU 記憶體，防止 GPU 記憶體不足的情況
        torch.cuda.empty_cache()
        gc.collect()
        # 根據檢查結果，列印使用 CPU 或 GPU
        if args.cuda:
            print('Running on GPU')
        else:
            print('Running on CPU')
        # 初始化 logger，設置 log 文件的保存位置
        logger = get_logger(path + args.dataset_name + '/logging.log')
        logger.info('start training on GPU {}!'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
        logger.info(args)
        # 加載資料集
        logger.info("Loading dataset...") # add
        # 設置訓練參數
        cuda = args.cuda
        n_epochs = args.epochs
        batch_size = args.batch_size
        # 使用 AutoTokenizer 從指定的 BERT 模型路徑加載 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
        tokenizer.add_tokens("<mask>")
        # 根據數據集名稱設置分類的情緒數量
        if args.dataset_name == "IEMOCAP":
            n_classes = 6
        elif args.dataset_name == "EmoryNLP":
            n_classes = 7
        elif args.dataset_name == "MELD":
            n_classes = 7
        # 加載對應的訓練、驗證、測試數據集
        trainset = DialogueDataset(args, dataset_name = args.dataset_name, split='train', tokenizer=tokenizer)
        devset = DialogueDataset(args, dataset_name = args.dataset_name, split='dev', tokenizer=tokenizer)
        testset = DialogueDataset(args, dataset_name = args.dataset_name, split='test', tokenizer=tokenizer)
        # 使用隨機抽樣器在每個 epoch 中隨機抽取樣本，防止過擬合並提高模型的泛化能力
        sampler = torch.utils.data.RandomSampler(trainset)
        # 創建數據加載器，指定批次大小、是否打亂數據和使用的工作執行緒數量
        logger.info("Creating data loaders...") # add
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=sampler, num_workers=1) # adjust
        valid_loader = DataLoader(devset, batch_size=args.batch_size, shuffle=False, num_workers=1) # adjust 
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=1) # adjust 
        # 開始建構模型
        logger.info("Building model...") # add
        # 初始化模型
        model = CLModel(args, n_classes, tokenizer)
        # 檢查設備，將模型移動到 GPU 或 CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device',device)
        model = model.to(device)
        num_training_steps = 1  # 後面未使用到此參數
        num_warmup_steps = 0    # 後面未使用到此參數
        # 初始化優化器和學習率調度器
        optimizer = AdamW(get_paramsgroup(model.module if hasattr(model, 'module') else model))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5, last_epoch=-1)
        # 初始化性能指標變量
        best_fscore,best_acc, best_loss, best_label, best_pred, best_mask = None,None, None, None, None, None
        all_fscore, all_acc, all_loss = [], [], []
        best_acc = 0.
        best_fscore = 0.
        # 開始訓練
        logger.info("Starting training...") # add
        # 使用 deepcopy 備份模型
        best_model = copy.deepcopy(model)
        best_test_fscore = 0
        anchor_dist = []        # 後面未使用到此參數
        # 訓練迴圈
        for e in range(n_epochs):
            logger.info(f"Epoch {e + 1} starting...") # add
            start_time = time.time()
            # 訓練模型，並計算訓練損失、準確率、F1分數等指標
            train_loss, train_acc, _, _, train_fscore, train_detail_f1, max_cosine  = \
                train_or_eval_model(model, loss_function, train_loader, e, device, args, optimizer, lr_scheduler, True)
            lr_scheduler.step()     # 更新學習率
            logger.info(f"Epoch {e + 1} training completed. Starting validation...") # add
            # 驗證模型，並記錄驗證結果
            valid_loss, valid_acc, _, _, valid_fscore, valid_detail_f1, _ = \
                train_or_eval_model(model, loss_function, valid_loader, e, device, args)
            logger.info(f"Epoch {e + 1} validation completed. Starting testing...") # add
            # 測試模型，並記錄測試結果
            test_loss, test_acc, test_label, test_pred, test_fscore, test_detail_f1, _ = \
                train_or_eval_model(model, loss_function, test_loader, e, device, args)
            logger.info(f"Epoch {e + 1} testing completed.") # add
            # 記錄每個 epoch 的結果
            all_fscore.append([valid_fscore, test_fscore, test_detail_f1])
            # 紀錄當前 epoch 的所有指標
            logger.info(f'Epoch: {e + 1}, train_loss: {train_loss}, train_acc: {train_acc}, train_fscore: {train_fscore}, '
                        f'valid_loss: {valid_loss}, valid_acc: {valid_acc}, valid_fscore: {valid_fscore}, '
                        f'test_loss: {test_loss}, test_acc: {test_acc}, test_fscore: {test_fscore}, '
                        f'time: {round(time.time() - start_time, 2)} sec') # add
            # 清空 CUDA 記憶體
            torch.cuda.empty_cache() # add
            # 保存最優測試分數模型
            if test_fscore > best_test_fscore:
                best_model = copy.deepcopy(model)
                best_test_fscore = test_fscore
                print("New best test F-score found. Saving model...") # add
                logger.info("New best test F-score found. Saving model...") # add
                torch.save(model.state_dict(), path + args.dataset_name + '/model_' + '.pkl')
        # 紀錄第一階段訓練結束
        logger.info('finish stage 1 training!')
        # 根據驗證結果排序，選擇最佳模型
        all_fscore = sorted(all_fscore, key=lambda x: (x[0],x[1]), reverse=True)
        # 檢查是否啟用了兩階段訓練，如果沒有，直接記錄最佳結果
        if args.disable_two_stage_training:
            logger.info('Best F-Score based on validation: {}'.format(all_fscore[0][1]))
            logger.info('Best F-Score based on test: {}'.format(max([f[1] for f in all_fscore])))
            logger.info(all_fscore[0][2])
        else:
            torch.cuda.empty_cache()        # 清空 CUDA 記憶體
            # 開始第二階段訓練
            with torch.no_grad():
                # 使用模型中的 map_function 來獲取情緒錨點(emotion anchors)
                anchors = model.map_function(model.emo_anchor)
                # 加載之前保存的最佳模型權重
                model.load_state_dict(torch.load(path + args.dataset_name + '/model_' + '.pkl'))
                # 評估模式(不啟用 Batch Normalization 和 Dropout)
                model.eval()
                # 初始化訓練、驗證和測試數據的嵌入向量和對應標籤的空列表
                emb_train, emb_val, emb_test = [] ,[] ,[]
                label_train, label_val, label_test = [], [], []
                # 遍歷訓練資料集，提取嵌入
                for batch_id, batch in enumerate(train_loader):
                    input_ids, label = batch
                    input_orig = input_ids  # 原始 input_ids
                    input_aug = None        # 預留的增強數據(如果有)
                    # 將 input_ids 和 label 移至指定的設備(GPU 或 CPU)
                    input_ids = input_orig.to(device)
                    label = label.to(device)
                    # 如果啟用了 FP16 精度，則使用自動混合精度進行計算
                    if args.fp16:
                        with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                            # 進行前向傳播，獲取模型的輸出結果，包括嵌入和情緒錨點分數
                            log_prob, masked_mapped_output, masked_outputs, anchor_scores = model(input_ids, return_mask_output=True) 
                    # 將提取出的嵌入從 GPU 移回 CPU 並儲存，避免記憶體溢出
                    emb_train.append(masked_mapped_output.detach().cpu())
                    label_train.append(label.cpu())
                # 將所有訓練數據的嵌入和標籤進行合併    
                emb_train = torch.cat(emb_train, dim=0)
                label_train = torch.cat(label_train, dim=0)
                # 遍歷驗證資料集，提取嵌入(與訓練資料集的流程類似)
                for batch_id, batch in enumerate(valid_loader):
                    input_ids, label = batch
                    input_orig = input_ids  
                    input_aug = None        
                    input_ids = input_orig.to(device)
                    label = label.to(device)
                    if args.fp16:
                        with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                            log_prob, masked_mapped_output, masked_outputs, anchor_scores = model(input_ids, return_mask_output=True) 
                    emb_val.append(masked_mapped_output.detach().cpu())
                    label_val.append(label.cpu())
                emb_val = torch.cat(emb_val, dim=0)
                label_val = torch.cat(label_val, dim=0)
                # 遍歷測試資料集，提取嵌入
                for batch_id, batch in enumerate(test_loader):
                    input_ids, label = batch
                    input_orig = input_ids
                    input_aug = None
                    input_ids = input_orig.to(device)
                    label = label.to(device)
                    if args.fp16:
                        with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                            print(torch.autocast)
                            log_prob, masked_mapped_output, masked_outputs, anchor_scores = model(input_ids, return_mask_output=True) 
                    emb_test.append(masked_mapped_output.detach().cpu())
                    label_test.append(label.cpu())
                emb_test = torch.cat(emb_test, dim=0)
                label_test = torch.cat(label_test, dim=0)
            # 紀錄嵌入數據集已建立
            logger.info("Embedding dataset built")
            # 初始化空列表來儲存所有的 F-score 結果
            all_fscore = []
            # 打包整合數據(訓練、驗證和測試)
            trainset = TensorDataset(emb_train, label_train)
            validset = TensorDataset(emb_val, label_val)
            testset = TensorDataset(emb_test, label_test)
            # 創建新的 DataLoader 來加載嵌入資料
            train_loader = DataLoader(trainset, batch_size=64, shuffle=False, pin_memory=True, sampler=sampler, num_workers=1) # adjust 
            valid_loader = DataLoader(validset, batch_size=64, shuffle=False, num_workers=1) # adjust 
            test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=1) # adjust 
            # 如果啟用了保存第二階段訓練的快取數據
            if args.save_stage_two_cache:
                os.makedirs("cache", exist_ok=True)     # 建立快取資料夾
                # 將訓練、驗證、測試 DataLoader 和錨點保存到快取中
                pickle.dump([train_loader, valid_loader, test_loader, anchors], open(f"./cache/{args.dataset_name}.pkl", 'wb'))
            # 初始化分類器模型
            clf = Classifier(args, anchors).to(device)
            # 使用 Adam 優化器來訓練分類器，設置學習率和權重衰減
            optimizer = torch.optim.Adam(clf.parameters(), lr=args.stage_two_lr, weight_decay=args.weight_decay)
            # 初始化變量以追蹤最佳的驗證分數
            best_valid_score = 0
            # 進行 10 個 epoch 的訓練
            for e in range(10):
                train_loss, train_ce_loss, train_acc, labels, preds, train_fscore, train_detail_f1 = retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).to(device), train_loader, e, device, args, optimizer, train=True)
                valid_loss, valid_ce_loss,  valid_acc, labels, preds, valid_fscore, valid_detail_f1  = retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).to(device), valid_loader, e, device, args, optimizer, train=False)
                test_loss, test_ce_loss,  test_acc, test_label, test_pred, test_fscore, test_detail_f1 = retrain(clf, nn.CrossEntropyLoss(ignore_index=-1).to(device), test_loader, e, device, args, optimizer, train=False)       
                # 紀錄當前 epoch 的結果
                logger.info( 'Epoch: {}, train_loss: {}, train_ce_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_ce_loss:{}, test_acc: {}, test_fscore: {}'. \
                        format(e + 1, train_loss, train_ce_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_ce_loss, test_acc, test_fscore))
                # 將驗證和測試的 F-score 結果添加到 all_fscore 列表中
                all_fscore.append([valid_fscore, test_fscore])
                # 如果當前的驗證 F-score 優於之前的最佳結果，則更新最佳分數，並保存模型
                if valid_fscore > best_valid_score:
                    best_valid_score = valid_fscore
                    # 保存分類器的狀態字典(模型權重)
                    torch.save(clf.state_dict(), path + args.dataset_name + '/clf_' + '.pkl')
                    # 保存測試詳細的 F1 分數
                    f = test_detail_f1
            # 將所有的 F-score 按照驗證分數進行排序
            all_fscore = sorted(all_fscore, key=lambda x: (x[0],x[1]), reverse=True)
            # 將所有的 F-score 按照驗證分數進行排序
            logger.info('Best F-Score based on validation: {}'.format(all_fscore[0][1]))
            logger.info('Best F-Score based on test: {}'.format(max([f[1] for f in all_fscore])))
            logger.info(f) 
            logger.info('Finish training!!!')   # add
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
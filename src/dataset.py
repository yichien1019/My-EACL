import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pickle
from utils.data_process import *
# 定義 DialogueDataset 類別，繼承自 PyTorch 的 Dataset 類別
class DialogueDataset(Dataset):    
    def __init__(self, args, dataset_name = 'IEMOCAP', split = 'train', speaker_vocab=None, label_vocab=None, tokenizer = None):
        self.speaker_vocab = speaker_vocab  # 話者的詞彙表
        self.label_vocab = label_vocab      # 情緒標籤的詞彙表
        self.args = args                    # 參數配置
        self.split = split                  # 資料分割 ('train', 'test', 'dev')
        # if osp.exists(self.save_path(dataset_name)):
        #     self.data, self.labels = torch.load(osp.join(self.save_path, f"{split}.pt"))
        # else:
        self.tokenizer = tokenizer          # 設定 tokenizer，將文本轉換為模型可讀的數字編碼
        self.wp = args.wp                   # 視窗處理參數，決定過去的對話窗口大小(past window size)
        self.wf = args.wf                   # 視窗處理參數，決定未來的對話窗口大小(future window size)
        self.max_len = args.max_len         # 最大序列長度
        self.pad_value = args.pad_value     # 填充值，用於將短於最大長度的輸入進行填充
        self.dataset_name = dataset_name    # 資料集名稱
        # 加載情緒映射
        self.emotion_map = pickle.load(open(f'./data/{dataset_name}/label_vocab.pkl', 'rb'))
        # 反轉詞彙表的鍵值對(value:key -> key:value)，使得索引能對應到情緒標籤
        # 詞彙表中 "happy" 映射到索引 0，這一步會將索引 0 映射回 "happy"
        self.emotion_map = {v:k for k,v in self.emotion_map.items()}
        # 設定特殊符號的編碼
        _special_tokens_ids = tokenizer('<mask>')['input_ids']  # 用來獲取符號的編碼
        self.CLS = _special_tokens_ids[0]   # CLS token 的 ID   (對話開始)
        self.MASK = _special_tokens_ids[1]  # MASK token 的 ID  (遮罩部分)
        self.SEP = _special_tokens_ids[2]   # SEP token 的 ID   (句子結尾)
        # 讀取資料，生成處理過的數據、標籤和話語序列
        self.data, self.labels, self.utterance_sequence = self.read(dataset_name, split, tokenizer)
        # 確認資料和標籤的數量一致
        assert len(self.data) == len(self.labels)
    # 將序列(list_data)填充至指定長度(max_len)
    # 如果序列長度超過最大長度則截斷，若不足則用 pad_value 進行填充
    def pad_to_len(self, list_data, max_len, pad_value):
        list_data = list_data[-max_len:]        # 如果超過最大長度，截取最後一部分
        len_to_pad = max_len - len(list_data)   # 計算需要填充的長度
        pads = [pad_value] * len_to_pad         # 構建填充列表
        list_data.extend(pads)                  # 將填充值附加到列表中
        return list_data
    # 讀取資料的函數
    def read(self, dataset_name, split, tokenizer):
        if dataset_name == "IEMOCAP":
            dialogs = load_iemocap_turn(f'./data/{dataset_name}/{split}_data.json')
        elif dataset_name == "EmoryNLP":
            dialogs = load_emorynlp_turn(f'./data/{dataset_name}/{split}_data.json')
        elif dataset_name == "MELD":
            dialogs = load_meld_turn(f'./data/{dataset_name}/{split}_data.csv')
        # 打印對話的數量
        print("number of dialogs:", len(dialogs))
        data_list = []          # 存儲處理過的資料
        label_list = []         # 存儲對應的標籤
        utterance_sequence = [] # 存儲每個話語的序列
        ret_utterances = []     # 存儲最終返回的話語
        ret_labels = []         # 存儲最終返回的標籤
        # 對所有對話做操作
        for dialogue in dialogs:
            utterance_ids = []  # 存儲每個對話中所有話語的 token IDs
            utterance_seq = []  # 存儲話語的文本，包含話者和對應的情緒標籤
            # turn_data 包含每個話語的數據(如話者、文本和情緒標籤)
            for idx, turn_data in enumerate(dialogue):
                # 將話者的文本拼接成一句話
                text_with_speaker = turn_data['speaker'] + ' says: ' + turn_data['text']
                # 進行編碼，去掉對話起始的 CLS token
                token_ids = tokenizer(text_with_speaker)['input_ids'][1:]
                # 將編碼結果添加到話語 ID 列表
                utterance_ids.append(token_ids)
                # 跳過無效的標籤
                if turn_data['label'] < 0:  
                    continue
                full_context = [self.CLS]   # 初始化上下文
                lidx = 0    # 設置起始索引
                # 迴圈構建上下文
                for lidx in range(idx):
                    # 計算上下文長度(包括一些邊界)
                    total_len = sum([len(item) for item in utterance_ids[lidx:]]) + 8
                    # 確保總長度不超過 max_len 最大值
                    if total_len + len(utterance_ids[idx]) <= self.max_len:
                        break
                # 限制上下文的長度最多包含 8 個話語，避免太長的上下文
                lidx = max(lidx, idx-8) 
                for item in utterance_ids[lidx:]:
                    full_context.extend(item)   # 將之前的話語編碼加到 full_context 中
                # 設置查詢的索引
                query_idx = idx     
                # 去掉當前話語的長度，生成 input_ids
                input_ids = full_context[:-len(utterance_ids[query_idx])]       
                # 添加到返回的話語列表中
                ret_utterances.append((input_ids, turn_data['speaker'], turn_data['text']))
                # 保存情緒標籤
                ret_labels.append(dialogue[query_idx]['label'])
                # 構建話語序列的字典，包含文本和情緒標籤    
                utterance_seq.append({
                    "uttrance": text_with_speaker,
                    "emotion": dialogue[query_idx]['label']
                })
                # 更新話語序列列表
                utterance_sequence.append(utterance_seq + [])
        # 包含所有處理過的話語上下文和話者信息
        data_list = ret_utterances
        # 將標籤轉換為 LongTensor
        label_list = torch.LongTensor(ret_labels)
        # 返回處理過的文本、情緒標籤和話語序列
        return data_list, label_list, utterance_sequence
    # 處理對話數據
    def process(self, data):
        # 解包資料 : 當前話語的編碼、話者名稱、文本內容
        input_ids, speaker, text = data     
        # 構建提示語句，包含話語內容和話者信息，<mask> 用於預測情緒標籤
        # 例 : "For utterance: I am very happy today Alice feels <mask>"
        p2 = 'For utterance: '+ text + " " + speaker + " feels <mask> "
        # 編碼並去掉 CLS token
        p2 = self.tokenizer(p2)['input_ids'][1:]
        # 將編碼結果與原始 input_ids 進行拼接
        p2 = input_ids + p2
        # 將序列填充到最大長度
        p2 = pad_to_len(p2, self.max_len, self.pad_value)
        # 轉換為 LongTensor
        p2 = torch.LongTensor(p2)
        # 序列包含了話語的編碼、說話者信息以及提示語句
        return p2
    def save_path(self, dataset_name):    # 返回保存路徑
        return f'./data/{dataset_name}/processed/{self.split}'
    # 據索引返回處理過的話語編碼和對應的情緒標籤
    def __getitem__(self, index):
        text = self.data[index]     # 根據索引獲取資料
        text = self.process(text)   # 處理資料
        label = self.labels[index]  # 獲取對應的標籤
        return text, label  # 返回處理過的文本和標籤
    # 返回數據集的大小
    def __len__(self):      
        return len(self.data)
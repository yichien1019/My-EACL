import json
import logging
import pickle
import random
import pandas as pd
import torch
import vocab
from tqdm import tqdm
# 使用 pad_value 將數據 list_data 填充到指定長度 max_len
def pad_to_len(list_data, max_len, pad_value):
    list_data = list_data[-max_len:]        # 如果數據長於 max_len，則截取最後 max_len 個元素，只保留需要的部分
    len_to_pad = max_len - len(list_data)   # 計算需要填充的長度
    pads = [pad_value] * len_to_pad         # 創建長度為 len_to_pad 的填充列表
    list_data.extend(pads)                  # 將填充值添加到數據的末尾
    return list_data                        # 返回填充後的列表
# 從 EmoryNLP 數據集中獲取情緒詞彙，並構建詞彙表 label_vocab.pkl
def get_emorynlp_vocabs(file_paths):
    emotion_vocab = vocab.Vocab()   # 創建一個新的詞彙表
    emotion_vocab.word2index('neutral', train=True) # 將中性情緒添加到詞彙表中
    for file_path in file_paths:
        data = json.load(open(file_path, 'r'))      # 打開並讀取 JSON 格式的數據
        # 數據集中每一集的所有內容，並顯示當前處理的文件路徑
        for episode in tqdm(data['episodes'],desc='processing file {}'.format(file_path)):
            for scene in episode['scenes']:
                for utterance in scene['utterances']:
                    emotion = utterance['emotion'].lower()          # 將情緒標籤轉為小寫
                    emotion_vocab.word2index(emotion, train=True)   # 添加情緒標籤到詞彙表中
    torch.save(emotion_vocab.to_dict(), './data/EmoryNLP/label_vocab.pkl')  # 將詞彙表轉換為 Python 字典格式並保存
    logging.info('total {} emotions'.format(len(emotion_vocab)))    # 輸出情緒詞彙表總數
# 從 MELD 數據集中獲取情緒詞彙，並構建詞彙表 label_vocab.pkl
def get_meld_vocabs(file_paths):
    emotion_vocab = vocab.Vocab()   # 創建一個新的詞彙表
    emotion_vocab.word2index('neutral', train=True) # 將中性情緒添加到詞彙表中
    for file_path in file_paths:
        data = pd.read_csv(file_path)               # 讀取 CSV 文件
        for row in tqdm(data.iterrows(),desc='get vocab from {}'.format(file_path)):
            meta = row[1]
            emotion = meta['Emotion'].lower()       # 將情緒標籤轉為小寫
            emotion_vocab.word2index(emotion, train=True)               # 添加情緒標籤到詞彙表中
    torch.save(emotion_vocab.to_dict(), "./data/MELD/label_vocab.pkl")  # 保存情緒詞彙表到文件中
    logging.info('total {} emotions'.format(len(emotion_vocab)))        # 輸出情緒詞彙表總數
# 從 IEMOCAP 數據集中獲取情緒詞彙，並構建詞彙表 label_vocab.pkl
def get_iemocap_vocabs(file_paths):
    emotion_vocab = vocab.Vocab()   # 創建一個新的詞彙表
    emotion_vocab.word2index('neu', train=True)     # 將中性情緒添加到詞彙表中
    for file_path in file_paths:
        data = json.load(open(file_path, 'r'))      # 打開並讀取 JSON 數據
        for dialog in tqdm(data,desc='get vocab from {}'.format(file_path)):
            for utterance in dialog:
                emotion = utterance.get('label')    # 獲取情緒標籤
                if emotion is not None:
                    emotion_vocab.word2index(emotion, train=True)   # 將情緒標籤添加到詞彙表中
    torch.save(emotion_vocab.to_dict(), './data/IEMOCAP/label_vocab.pkl')   # 保存情緒詞彙表到文件中
    logging.info('total {} emotions'.format(len(emotion_vocab)))    # 輸出情緒詞彙表總數
# 構建數據集，將對話轉換為模型可以訓練的格式
def build_dataset(dialogues, train=False):
    ret_utterances = [] # 保存所有轉換為 token 的話語
    ret_labels = []     # 保存對應的情緒標籤
    # dialogues 是多個對話組成的數據集，逐個處理每個對話
    for dialogue in dialogues:
        utterance_ids = []  # 保存當前對話中的所有話語
        query = 'For utterance:'
        # 將 query 轉換為 token 並去除開頭[CLS]和結尾[SEP]的特殊符號
        query_ids = tokenizer(query)['input_ids'][1:-1] 
        for idx, turn_data in enumerate(dialogue):
            text_with_speaker = turn_data['speaker'] + ':' + turn_data['text']
            token_ids = tokenizer(text_with_speaker)['input_ids'][1:]   # 將當前話語轉換為 token，去除開頭 [CLS]
            utterance_ids.append(token_ids)     # 將 token 加入當前對話的話語列表
            if turn_data['label'] < 0:
                continue    # 如果標籤無效(小於0)則跳過
            full_context = [CONFIG['CLS']]  # 開始構建完整上下文包含 [CLS]
            lidx = 0
            for lidx in range(idx):
                # 8 是因為有時會有額外的 token 或查詢句需要計入上下文長度
                total_len = sum([len(item) for item in utterance_ids[lidx:]]) + 8
                # 如果當前長度加上新話語長度不超過最大長度，則跳出循環，表示上下文長度是合理
                if total_len + len(utterance_ids[idx]) <= CONFIG['max_len']:
                    break   
            # 最多只回溯 8 個話語來構建上下文，防止上下文過長，也可以平衡模型訓練的計算量
            lidx = max(lidx, idx-8)     
            for item in utterance_ids[lidx:]:
                full_context.extend(item)   # 將話語添加到完整上下文中
            query_idx = idx                 # 設置查詢索引
            # 查詢句子構建包括當前話語的 token 以及帶有 feels <mask> 的查詢 <mask> 用來預測情緒標籤
            prompt = dialogue[query_idx]['speaker'] + ' feels <mask>'       # 構建查詢句子
            full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:] 
            # full_context 前面構建的上下文，full_query 是剛剛構建的查詢句子
            # 拼接後模型可以根據之前的話語上下文來預測當前說話者的情緒
            input_ids = full_context + full_query               
            input_ids = pad_to_len(input_ids, CONFIG['max_len'], CONFIG['pad_value'])   # 填充至最大長度
            ret_utterances.append(input_ids)                    # 保存話語 token
            ret_labels.append(dialogue[query_idx]['label'])     # 保存對應的情緒標籤
            # 隨機選擇早期話語進行訓練(僅訓練模式)
            if train and idx > 3 and torch.rand(1).item() < 0.2:
                query_idx = random.randint(lidx, idx-1)
                if dialogue[query_idx]['label'] < 0:
                    continue
                prompt = dialogue[query_idx]['speaker'] + ' feels <mask>'
                full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
                input_ids = full_context + full_query
                input_ids = pad_to_len(input_ids, CONFIG['max_len'], CONFIG['pad_value'])   # 再次進行填充
                ret_utterances.append(input_ids)
                ret_labels.append(dialogue[query_idx]['label'])
    # 將結果轉換為 TensorDataset 以便於 PyTorch 的數據加載器使用
    dataset = TensorDataset(
        torch.LongTensor(ret_utterances),
        torch.LongTensor(ret_labels)
    )
    return dataset      # 返回構建好的數據集

# 從 EmoryNLP 數據集中載入每個回合的數據
def load_emorynlp_turn(file_path):
    with open('./data/EmoryNLP/label_vocab.pkl', 'rb') as f:    # 載入情緒詞彙表
        emotion_vocab = pickle.load(f)      # 從 pickle 檔案中載入情緒詞彙表
    data = json.load(open(file_path, 'r'))  # 從指定的 JSON 檔案中載入數據
    dialogues = []      # 初始化對話列表
    speaker_vocab = vocab.Vocab()           # 創建一個新的說話者詞彙表
    # 遍歷每個數據
    for episode in tqdm(data['episodes'],desc='processing file {}'.format(file_path)):
        for scene in episode['scenes']:                 # 遍歷每個場景
            dialogue = []                               # 初始化場景中的對話
            for utterance in scene['utterances']:       # 遍歷每個話語
                text = utterance['transcript']          # 提取話語文本
                speaker = utterance['speakers'][0]      # 提取話語的說話者
                speaker = speaker.split(' ')[0]         # 只取說話者的名字
                emotion = utterance['emotion'].lower()  # 提取情緒並轉為小寫
                emotion_idx = emotion_vocab[emotion]    # 將情緒轉換為詞彙表中的索引
                # 保存話語的各種信息
                turn_data = {}
                turn_data['speaker'] = speaker
                speaker_vocab.word2index(speaker, train=True)   # 將說話者加入詞彙表
                turn_data['text'] = text
                turn_data['label'] = emotion_idx                # 將情緒標籤加入話語
                turn_data['emotion'] = emotion
                dialogue.append(turn_data)              # 將處理好的話語加入對話
            dialogues.append(dialogue)                  # 將場景對話加入對話列表
    return dialogues                                    # 返回處理好的對話列表

# 從 MELD 數據集中載入每個回合的數據
def load_meld_turn(file_path):
    with open('./data/MELD/label_vocab.pkl', 'rb') as f:   
        emotion_vocab = pickle.load(f)          # 從 pickle 檔案中載入情緒詞彙表
    data = pd.read_csv(file_path)               # 讀取 CSV 格式的數據
    pre_dial_id = -1                            # 初始化對話 ID
    dialogues = []                              # 初始化對話列表
    dialogue = []                               # 初始化當前對話
    speaker_vocab = vocab.Vocab()               # 創建一個說話者的詞彙表
    # 遍歷每一行數據
    for row in tqdm(data.iterrows(),desc='processing file {}'.format(file_path)):
        meta = row[1]                           # 提取當前行的數據
        text = meta['Utterance'].replace('’', '\'').replace("\"", '')   # 清理話語文本
        speaker = meta['Speaker']               # 提取說話者的名稱
        emotion = meta['Emotion'].lower()       # 提取情緒並轉為小寫
        emotion_idx = emotion_vocab[emotion]    # 根據情緒標籤詞彙表查找該情緒對應的索引
        # 保存話語的各種信息
        turn_data = {}
        turn_data['speaker'] = speaker          # 保存說話者名稱
        speaker_vocab.word2index(speaker, train=True)   # 將說話者加入詞彙表
        turn_data['text'] = text                # 保存清理過的對話文本
        turn_data['label'] = emotion_idx        # 將情緒標籤加入話語
        dialogue_id = meta['Dialogue_ID']       # 提取當前對話 ID
        if pre_dial_id == -1:   # 第一次處理時將其設置為當前的對話 ID
            pre_dial_id = dialogue_id
        # 如果當前對話 ID 與之前不一致，則將當前對話添加到對話列表並開始新對話
        if dialogue_id != pre_dial_id:
            dialogues.append(dialogue)          # 保存之前所有話語的列表
            dialogue = []                       # 初始化新列表
        pre_dial_id = dialogue_id               # 更新對話 ID
        dialogue.append(turn_data)              # 將當前話語加入當前對話
    dialogues.append(dialogue)                  # 將最後一個對話加入對話列表
    return dialogues                            # 返回處理好的對話列表

# 從 IEMOCAP 數據集中載入每個回合的數據
def load_iemocap_turn(file_path):
    with open('./data/IEMOCAP/label_vocab.pkl', 'rb') as f:    
        emotion_vocab = pickle.load(f)          # 從 pickle 檔案中載入情緒詞彙表
    data = json.load(open(file_path, 'r'))      # 從指定的 JSON 檔案中載入數據
    speaker_pools = json.load(open('./data/IEMOCAP/name_pool', 'r'))    # 載入說話者名稱池
    dialogues = []          # 初始化對話列表
    count = 0               # 計數變量，用來記錄有效情緒標籤的總數
    # 遍歷每個對話
    for dialog in tqdm(data,desc='processing file {}'.format(file_path)):
        dialogue = []                       # 初始化當前對話
        t_vocab = vocab.Vocab()             # 創建一個新的說話者詞彙表
        speaker_vocab = vocab.Vocab()       # 創建一個新的說話者詞彙表
        # 遍歷對話中的每個話語
        for utterance in dialog:
            speaker = utterance.get('speaker').upper()      # 提取並轉換說話者為大寫
            text = utterance.get('text').replace('[LAUGHTER]', '')  # 將話語中的笑聲標記 [LAUGHTER] 移除只保留純文本
            emotion = utterance.get('label')                # 提取情緒標籤
            speaker = speaker_pools[t_vocab.word2index(speaker, train=True)]    # 將說話者轉換為名稱池中的對應名稱
            speaker_vocab.word2index(speaker, train=True)   # 將說話者加入詞彙表
            # 保存話語的各種信息
            turn_data = {}
            turn_data['speaker'] = speaker  # 保存說話者名稱
            turn_data['text'] = text        # 保存話語文本
            turn_data['emotion'] = emotion  # 保存情緒標籤
            if emotion is not None:         # 有效的情緒標籤
                emotion_idx = emotion_vocab[emotion]        # 將情緒轉換為詞彙表中的索引
                count += 1                  # 更新計數器
            else:
                emotion_idx = -1            # 無效情緒標籤設置為 -1
            turn_data['label'] = emotion_idx# 保存情緒標籤的索引
            dialogue.append(turn_data)      # 將處理好的話語加入當前對話
        dialogues.append(dialogue)          # 將當前對話加入對話列表
    print(count)            # 輸出情緒標籤的總數
    return dialogues        # 返回處理好的對話列表

       
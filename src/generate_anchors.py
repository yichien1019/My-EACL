import os
import torch
from transformers import AutoModel, AutoTokenizer, pipeline
import argparse
import warnings
from utils.data_process import *
# 忽略不必要的警告
warnings.filterwarnings("ignore")
# 禁用分詞器的多線程並行處理，防止在多線程環境下出現衝突
os.environ["TOKENIZERS_PARALLELISM"] = "1"
# 定義一個解析命令行參數的函數
def get_parser():
    parser = argparse.ArgumentParser()
    # 定義參數 --bert_path，指定 BERT 模型的路徑，默認為一個預訓練模型的名稱
    parser.add_argument('--bert_path', type=str, default='princeton-nlp/sup-simcse-roberta-large')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parser()
    # 使用 Hugging Face 的 AutoTokenizer 來從指定的 BERT 模型路徑加載分詞器
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    # 加載指定的預訓練 BERT 模型
    model = AutoModel.from_pretrained(args.bert_path)
    model.eval()
    # 保存路徑，從 BERT 路徑中提取模型的名稱作為保存的文件夾名稱
    save_path = args.bert_path.split("/")[-1]
    # 定義三個數據集的情緒標籤
    iemocap_emos = ["neutral","excited","frustrated","sad","happy","angry"]
    meld_emos = ['anger','disgust','fear','joy','sadness','surprise','neutral']
    emorynlp_emos = ['joyful','neutral','powerful','mad','scared','peaceful','sad']
    embeddings = []     # 初始化嵌入向量列表
    # 創建一個 Hugging Face pipeline 特徵提取管道，這裡指定為 "feature-extraction"，用來提取特徵嵌入向量
    feature_extractor = pipeline("feature-extraction",framework="pt",model=args.bert_path)

    embeddings = []     # 初始化 embeddings 列表，用來存儲 IEMOCAP 的嵌入向量
    # 處理 IEMOCAP 數據集的情緒嵌入
    with torch.no_grad():
        for emo in iemocap_emos:   
            # 使用特徵提取管道(feature_extractor)提取每個情緒(emo)的嵌入，並計算每個情緒短語的平均嵌入(mean(0))
            emb = torch.tensor(feature_extractor(emo,return_tensors = "pt")[0]).mean(0)
            embeddings.append(emb.unsqueeze(0))         # 將每個嵌入向量加入列表
        embeddings = torch.cat(embeddings, dim=0)       # 將所有嵌入向量串聯成一個完整的矩陣
        # 保存 IEMOCAP 數據集的情緒嵌入到指定的文件
        torch.save(embeddings, f"./emo_anchors/{save_path}/iemocap_emo.pt")

    embeddings = []     # 清空 embeddings 列表，準備處理 MELD 數據集
    # 處理 MELD 數據集的情緒嵌入
    with torch.no_grad():
        for emo in meld_emos:       
            emb = torch.tensor(feature_extractor(emo,return_tensors = "pt")[0]).mean(0)
            embeddings.append(emb.unsqueeze(0))         # 將每個嵌入向量加入列表
        embeddings = torch.cat(embeddings, dim=0)       # 將所有嵌入向量串聯成一個矩陣
        # 保存 MELD 數據集的情緒嵌入到指定文件
        torch.save(embeddings, f"./emo_anchors/{save_path}/meld_emo.pt")

    embeddings = []     # 清空 embeddings 列表，處理 EmoryNLP 數據集
    # 處理 EmoryNLP 數據集的情緒嵌入
    with torch.no_grad():
        for emo in emorynlp_emos:   
            emb = torch.tensor(feature_extractor(emo,return_tensors = "pt")[0]).mean(0)
            embeddings.append(emb.unsqueeze(0))         # 將每個嵌入向量加入列表
        embeddings = torch.cat(embeddings, dim=0)       # 將所有嵌入向量串聯成一個矩陣
        # 保存 EmoryNLP 數據集的情緒嵌入到指定文件
        torch.save(embeddings, f"./emo_anchors/{save_path}/emorynlp_emo.pt")
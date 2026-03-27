import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)  # 新增：忽略jieba警告

import torch
import random
import pandas as pd
import json
import numpy as np
import nltk
import jieba
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

# 修复1：扩展字典，增加空值/空字符串的映射兜底
label_dict = {
    "real": 0,
    "fake": 1,
    0: 0,
    1: 1,
    "": 0,    # 空字符串映射为0
    None: 0   # None值映射为0
}

label_dict_ftr_pred = {
    "real": 0,
    "fake": 1,
    "other": 2,
    0: 0,
    1: 1,
    2: 2,
    "": 2,     # 空字符串映射为2（other）
    None: 2    # None值映射为2（other）
}

def word2input(texts, max_len, tokenizer):
    # 修复2：处理文本中的空值/空字符串，避免tokenizer报错
    texts = ["" if pd.isna(t) or t == "" else t for t in texts]
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def get_dataloader(path, max_len, batch_size, shuffle, bert_path, data_type, language):
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    if data_type == 'rationale':
        data_list = json.load(open(path, 'r',encoding='utf-8'))
        # 修复3：用列表暂存数据，避免循环append（Pandas2.0+兼容）
        df_list = []
        for item in data_list:
            tmp_data = {}

            # 修复4：所有字段增加空值/空字符串兜底，避免KeyError+空值问题
            # content info（文本字段空值填充为空字符串）
            tmp_data['content'] = item.get('content', "")
            tmp_data['label'] = item.get('label', "")  # 兜底为空字符串，后续映射处理
            tmp_data['id'] = item.get('source_id', 0)  # 空值默认0（整数）

            tmp_data['FTR_2'] = item.get('td_rationale', "")
            tmp_data['FTR_3'] = item.get('cs_rationale', "")

            tmp_data['FTR_2_pred'] = item.get('td_pred', "")
            tmp_data['FTR_3_pred'] = item.get('cs_pred', "")

            # 布尔/数值字段空值默认0（避免空字符串转int失败）
            tmp_data['FTR_2_acc'] = item.get('td_acc', 0)
            tmp_data['FTR_3_acc'] = item.get('cs_acc', 0)

            df_list.append(tmp_data)
        
        # 批量构建DataFrame，替代循环append
        df_data = pd.DataFrame(df_list) if df_list else pd.DataFrame(columns=['content','label','id','FTR_2','FTR_3','FTR_2_pred','FTR_3_pred','FTR_2_acc','FTR_3_acc'])

        # 修复5：所有数值字段强制清洗+类型转换，避免空字符串/None
        # 1. 文本字段（转字符串，空值填充）
        content = df_data['content'].fillna("").astype(str).to_numpy()
        FTR_2 = df_data['FTR_2'].fillna("").astype(str).to_numpy()
        FTR_3 = df_data['FTR_3'].fillna("").astype(str).to_numpy()

        # 2. label字段（先映射，再转int）
        label = df_data['label'].fillna("").apply(lambda c: label_dict.get(c, 0)).astype(int).to_numpy()
        label = torch.tensor(label, dtype=torch.int64)

        # 3. source_id（空值默认0，强制转int）
        id = df_data['id'].fillna(0).astype(str).replace("", "0").astype(int).to_numpy()
        id = torch.tensor(id, dtype=torch.int64)

        # 4. FTR_2_pred/FTR_3_pred（映射+空值兜底+转int）
        FTR_2_pred = df_data['FTR_2_pred'].fillna("").apply(lambda c: label_dict_ftr_pred.get(c, 2)).astype(int).to_numpy()
        FTR_2_pred = torch.tensor(FTR_2_pred, dtype=torch.int64)
        FTR_3_pred = df_data['FTR_3_pred'].fillna("").apply(lambda c: label_dict_ftr_pred.get(c, 2)).astype(int).to_numpy()
        FTR_3_pred = torch.tensor(FTR_3_pred, dtype=torch.int64)

        # 5. FTR_2_acc/FTR_3_acc（空值默认0，强制转int）
        FTR_2_acc = df_data['FTR_2_acc'].fillna(0).astype(str).replace("", "0").astype(int).to_numpy()
        FTR_2_acc = torch.tensor(FTR_2_acc, dtype=torch.int64)
        FTR_3_acc = df_data['FTR_3_acc'].fillna(0).astype(str).replace("", "0").astype(int).to_numpy()
        FTR_3_acc = torch.tensor(FTR_3_acc, dtype=torch.int64)

        # 文本token化（已处理空值）
        content_token_ids, content_masks = word2input(content, max_len, tokenizer)
        FTR_2_token_ids, FTR_2_masks = word2input(FTR_2, max_len, tokenizer)
        FTR_3_token_ids, FTR_3_masks = word2input(FTR_3, max_len, tokenizer)

        # 构建Dataset
        dataset = TensorDataset(content_token_ids,
                                content_masks,
                                FTR_2_pred,
                                FTR_2_acc,
                                FTR_3_pred,
                                FTR_3_acc,
                                FTR_2_token_ids,
                                FTR_2_masks,
                                FTR_3_token_ids,
                                FTR_3_masks,
                                label,
                                id,
                                )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=shuffle
        )
        return dataloader
    else:
        print('No match data type!')
        exit()
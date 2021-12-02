#! -*- coding:utf-8 -*-
# 数据集：MNLI，句子对分类
# https://github.com/sherlcok314159/GLUE-torch
### 待定结果

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from config import mnli_config
from torch.utils.data.dataloader import Dataset, DataLoader
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

def load_data_train(file) -> list:
    '''读取训练集'''
    data = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            if line[0] == "index":
                continue
            text1, text2, label = line[-4], line[-3], line[-1]
            data.append([text1, text2, label])
    return data

def load_data_val(file) -> list:
    '''读取验证集'''
    data = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            if line[0] == "index":
                continue
            text1, text2, label = line[-8], line[-7], line[-1]
            data.append([text1, text2, label])
    return data

def load_data_test(file) -> list:
    '''读取测试集'''
    data = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            if line[0] == "index":
                continue
            text1, text2 = line[-2], line[-1]
            data.append([text1, text2])
    return data

class Mnli_Dataset(Dataset):
    def __init__(self, path, mode):
        if mode == "train":
            self.data = load_data_train(path)
        elif mode == "eval":
            self.data = load_data_val(path)
        else:
            self.data = load_data_test(path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(data):
    tokenizer = BertTokenizer.from_pretrained(mnli_config["model_path"])
    label_map = {"contradiction":0, "entailment":1, "neutral":2}
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = tokenizer(x[0], text_pair=x[1], padding='max_length', truncation=True,\
             max_length=mnli_config["max_seq_length"], return_tensors='pt')
        input_ids.append(text['input_ids'].squeeze().tolist())
        attention_mask.append(text['attention_mask'].squeeze().tolist())
        token_type_ids.append(text['token_type_ids'].squeeze().tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    if x[-1] in label_map.keys():
        label = torch.tensor([label_map[x[-1]] for x in data])
    else:
        label = None
    return input_ids, attention_mask, token_type_ids, label

def train(dataloader, model, optimizer):
    '''训练模型'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.zero_grad()
    model.train()

    tq = tqdm(dataloader)
    acc, loss_num, train_len = 0, 0, 0
    for step, batch in enumerate(tq):
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3]
        }
        ## outputs ==> loss, logits, hidden_states, attentions
        outputs = model(**inputs)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        model.zero_grad()
    
        logits = outputs[1]
        y_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
        labels = inputs["labels"]
        labels = labels.detach().cpu().numpy()
        
        acc += sum(y_pred == labels) 
        loss_num += loss.item()
        train_len += len(labels)
        ## 进度条实时显示
        tq.set_postfix(loss=loss_num / train_len, acc=acc / train_len)

def eval(dataloader, model):
    '''模型评估'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        tq = tqdm(dataloader)
        acc, eval_len = 0, 0
        for step, batch in enumerate(tq):
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]
            }

            ## outputs ==> loss, logits, hidden_states, attentions
            outputs = model(**inputs)
            logits = outputs[1]
            y_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
            labels = inputs["labels"]
            labels = labels.detach().cpu().numpy()
            acc += sum(y_pred == labels)
            eval_len += len(labels)
    return acc / eval_len

def test(dataloader, model, mode):
    '''模型预测'''
    label_map = {0:"contradiction", 1:"entailment", 2:"neutral"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        tq = tqdm(dataloader)
        for step, batch in enumerate(tq):
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2]
            }

            ## outputs ==> logits, hidden_states, attentions
            outputs = model(**inputs)
            logits = outputs[0]
            y_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
    
    del model
    torch.cuda.empty_cache()

    output_path = "predict"
    if not os.path.exists(output_path):
        os.makedirs(output_path)    
    with open(f"{output_path}/mnli_{mode}.tsv", "w") as f:
        f.write("index\tprediction\n")
        for i, pred in enumerate(y_pred):
            f.write(f"{i}\t{label_map[pred]}\n")

def load_all():
    '''预处理全部数据'''
    ## train
    train_path = os.path.join(mnli_config["data_dir"], "train.tsv")
    training_data = Mnli_Dataset(train_path, "train")
    train_dataloader = DataLoader(training_data, batch_size=mnli_config["batch_size"], \
        collate_fn=collate_fn, num_workers=8)

    ## eval_match
    eval_match_path = os.path.join(mnli_config["data_dir"], "dev_matched.tsv")
    eval_match_data = Mnli_Dataset(eval_match_path, "eval")
    eval_match_dataloader = DataLoader(eval_match_data, batch_size=mnli_config["batch_size"], \
        collate_fn=collate_fn, num_workers=8)

    ## eval_dismatch
    eval_mismatch_path = os.path.join(mnli_config["data_dir"], "dev_mismatched.tsv")
    eval_mismatch_data = Mnli_Dataset(eval_mismatch_path, "eval")
    eval_mismatch_dataloader = DataLoader(eval_mismatch_data, batch_size=mnli_config["batch_size"], \
        collate_fn=collate_fn, num_workers=8)

    ## test_match
    test_match_path = os.path.join(mnli_config["data_dir"], "test_matched.tsv")
    test_match_data = Mnli_Dataset(test_match_path, "test")
    test_match_dataloader = DataLoader(test_match_data, batch_size=mnli_config["batch_size"], \
        collate_fn=collate_fn, num_workers=8)

    ## test_mismatch
    test_mismatch_path = os.path.join(mnli_config["data_dir"], "test_mismatched.tsv")
    test_mismatch_data = Mnli_Dataset(test_mismatch_path, "test")
    test_mismatch_dataloader = DataLoader(test_mismatch_data, batch_size=mnli_config["batch_size"], \
        collate_fn=collate_fn, num_workers=8)

    return train_dataloader, eval_match_dataloader, eval_mismatch_dataloader, \
        test_match_dataloader, test_mismatch_dataloader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BertConfig.from_pretrained(mnli_config["model_path"])
    config.num_labels = mnli_config["num_labels"]
    model = BertForSequenceClassification.from_pretrained(mnli_config["model_path"], config=config)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=mnli_config["learning_rate"])

    train_dataloader, eval_match_dataloader, eval_mismatch_dataloader, \
        test_match_dataloader, test_mismatch_dataloader = load_all()

    best_acc_match, best_acc_dismatch = 0, 0
    for e in range(mnli_config["epochs"]):
        train(train_dataloader, model, optimizer)
        acc_match = eval(eval_match_dataloader, model) 
        acc_dismatch = eval(eval_mismatch_dataloader, model)     
        if acc_match > best_acc_match:
            best_acc_match = acc_match
            # 保存模型
            path = mnli_config["output_dir"]
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model.module if hasattr(model, "module") else model, path, \
                _use_new_zipfile_serialization=False)
        if acc_dismatch > best_acc_dismatch:
            best_acc_dismatch = acc_dismatch
    
    model = BertForSequenceClassification.from_pretrained(mnli_config["output_dir"], \
        num_labels=mnli_config["num_labels"])
    
    test(test_match_dataloader, model, "matched")
    test(test_mismatch_dataloader, model, "mismatched")

main()

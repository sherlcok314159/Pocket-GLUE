import os
import torch
import logging
import transformers
import numpy as np
from config import *
from tqdm import tqdm
from helper import set_seed
from load_data import *
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from torch.utils.data.dataloader import Dataset, DataLoader
from transformers import AdamW, BertConfig, BertTokenizer, BertForSequenceClassification

transformers.logging.set_verbosity_error()
set_seed(2021)

TASKS = ["CoLA", "MNLI", "MRPC", "QNLI", "QQP", "RTE", "SST-2", "STS-B", "WNLI"]

Task2Processor = {
    "CoLA": ColaProcessor,
    "SST-2": Sst2Processor,
    "MRPC": MrpcProcessor,
    "STS-B": StsbProcessor,
    "QQP": QqpProcessor,
    "MNLI": MnliProcessor,
    "QNLI": QnliProcessor,
    "RTE": QnliProcessor,
    "WNLI": QqpProcessor
}

class GLUE_Dataset(Dataset):
    '''Load Data To Dataset'''
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def log():
    '''Set Up Log'''
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
        filename=f"log/logs",filemode='w')

def collate_fn(data):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    input_ids, attention_mask, token_type_ids = [], [], []
    text_pair = None
    for x in data:
        if len(x) > 2:
            text_pair = x[1]            
        text = tokenizer(x[0], text_pair=text_pair, padding='max_length', truncation=True,\
             max_length=max_seq_length, return_tensors='pt')
        input_ids.append(text['input_ids'].squeeze().tolist())
        attention_mask.append(text['attention_mask'].squeeze().tolist())
        token_type_ids.append(text['token_type_ids'].squeeze().tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)

    label = torch.tensor([x[-1] for x in data])

    return input_ids, attention_mask, token_type_ids, label

def ToDataloader(task):
    '''Load data to train_dataloader, eval_dataloader and test_dataloader'''
    processor = Task2Processor[task]()
    data_path = os.path.join(data_dir, task)

    train_examples, eval_examples, test_examples = processor.get_train_examples(data_path),\
       processor.get_dev_examples(data_path), processor.get_test_examples(data_path)
    
    labels = processor.get_labels()
    train_data = GLUE_Dataset(train_examples)
    eval_data = GLUE_Dataset(eval_examples)
    test_data = GLUE_Dataset(test_examples)
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    
    if task == "MNLI":
        eval_mis_examples, test_mis_examples = processor.get_dev_mismatched_examples(data_path), \
            processor.get_test_mismatched_examples(data_path)

        eval_mis = GLUE_Dataset(eval_mis_examples)
        test_mis = GLUE_Dataset(test_mis_examples)

        eval_mis_dataloader = DataLoader(eval_mis, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
        test_mis_dataloader = DataLoader(test_mis, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)

        return train_dataloader, eval_dataloader, eval_mis_dataloader, \
                     test_dataloader, test_mis_dataloader, labels

    return train_dataloader, eval_dataloader, test_dataloader, labels

def train(task, dataloader, model, optimizer):
    '''Train The Model'''
    logging.debug("Training Begins!!!\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.zero_grad()
    model.train()

    tq = tqdm(dataloader)
    acc, loss_num, train_len = 0, 0, 0
    y_l, y_p = [], []
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
        if task != "STS-B":
            y_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
        else:
            y_pred = logits.detach().cpu().numpy()
        
        labels = inputs["labels"]
        labels = labels.detach().cpu().numpy()
        
        y_l += list(labels)
        y_p += list(y_pred)

        loss_num += loss.item()
        train_len += len(labels)

        if task == "STS-B":
            y_p = [float(i) for i in y_p]
            pearson_corr = pearsonr(y_p, y_l)[0] 
            spearman_corr = spearmanr(y_p, y_l)[0]

            tq.set_postfix(loss=loss_num / train_len, pearson=pearson_corr, spearman=spearman_corr)
        else:
            acc += sum(y_pred == labels) 
            tq.set_postfix(loss=loss_num / train_len, acc=acc / train_len)

def eval(task, dataloader, model):
    '''Evaluate The Model'''
    logging.debug("Evaluation Begins!!!\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    y_l, y_p = [], []
    with torch.no_grad():
        tq = tqdm(dataloader)
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
            if task != "STS-B":
                y_pred = np.argmax(logits.detach().cpu().numpy(), axis=1)
            else:
                y_pred = logits.detach().cpu().numpy()
            labels = inputs["labels"]
            labels = labels.detach().cpu().numpy()

            y_l += list(labels)
            y_p += list(y_pred)
        
        acc = sum(np.array(y_l) == np.array(y_p)) / len(y_l)

        if task == "STS-B":
            y_p = [float(i) for i in y_p]
            pearson_corr = pearsonr(y_p, y_l)[0] 
            spearman_corr = spearmanr(y_p, y_l)[0]

            logging.debug(f'------pearson_corr:{pearson_corr}, spearman_corr:{spearman_corr}------')
            return pearson_corr

        elif task == "CoLA":
            mcc = matthews_corrcoef(y_l, y_p)
            logging.debug(f'------------Acc:{acc}, Mcc:{mcc}------------')
            return mcc

        else:
            f1 = f1_score(y_l, y_p) if task != "MNLI" else f1_score(y_l, y_p, average="macro")
            logging.debug(f'------------Acc:{acc}, F1:{f1}------------')
            return acc

def test(task, dataloader, model, labels):
    '''Test The Model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = None
    with torch.no_grad():
        tq = tqdm(dataloader)
        for step, batch in enumerate(tq):
            ## labels ==> None
            batch = tuple(t.to(device) for t in batch[:-1])
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2]
            }

            ## outputs ==> logits, hidden_states, attentions
            outputs = model(**inputs)
            logits = outputs[0]
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    if task != "STS-B":
        preds = np.argmax(preds, axis=1)
    else:
        preds = [min(i[0], 5.0) for i in preds]
    del model
    torch.cuda.empty_cache()

    output_path = "predict"
    if not os.path.exists(output_path):
        os.makedirs(output_path)    

    if task == "MNLI":
        task = "MNLI-m"
        if os.path.isfile(f"{output_path}/{task}.tsv"):
            task = "MNLI-mm"

    with open(f"{output_path}/{task}.tsv", "w") as f:
        f.write("index\tprediction\n")
        for i, pred in enumerate(preds):
            if labels != 0.:
                f.write(f"{i}\t{labels[pred]}\n")
            else:
                f.write(f"{i}\t{pred}\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for task in TASKS:
        log()

        if task == "MNLI":
            train_dataloader, eval_dataloader, eval_mis_dataloader, \
                     test_dataloader, test_mis_dataloader, labels = ToDataloader(task)
        else:
            train_dataloader, eval_dataloader, test_dataloader, labels = ToDataloader(task)

        config = BertConfig.from_pretrained(model_path)
        config.num_labels = len(labels) if type(labels) != float else 1
        model = BertForSequenceClassification.from_pretrained(model_path, config=config)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate)

        best_metric = 0
        path = f"pretrained/{task}"
        if not os.path.exists(path):
            os.makedirs(path)

        for e in range(epochs):
            logging.debug(f'------------epoch:{e}------------')
            train(task, train_dataloader, model, optimizer)
            metric = eval(task, eval_dataloader, model) 
            if task == "MNLI":
                dis_metric = eval(task, eval_mis_dataloader, model)
            if metric > best_metric:
                best_metric = metric
                save_path = os.path.join(path, "pytorch_model.bin")
                torch.save(model.module if hasattr(model, "module") else model, save_path, \
                    _use_new_zipfile_serialization=False)

        predict_path = os.path.join(path, "pytorch_model.bin")
        model = torch.load(predict_path)
        model.to(device)

        test(task, test_dataloader, model, labels)
        if task == "MNLI":
            test(task, test_mis_dataloader, model, labels)

        logging.debug(f"---------------------{task} Done!------------------\n\n\n")

main()

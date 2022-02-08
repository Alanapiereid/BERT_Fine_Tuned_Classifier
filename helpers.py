import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def generate_split(X, y):
    X_train, y_train, X_interim, y_interim = train_test_split(X, y, random_state=1, test_size=0.3, stratify=y)
    X_val, y_val, X_test, y_test = train_test_split(X_interim, y_interim, random_state=1, test_size=0.5, stratify=y_interim)
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_loader(X, y):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokens = tokenizer.batch_encode_plus(
    X.tolist(),
    max_length = 100,
    pad_to_max_length=True,
    truncation=True)
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    y = torch.tensor(y.tolist())
    data_wrapper = TensorDataset(seq, mask, y)
    sampler = RandomSampler(data_wrapper)
    dataloader = DataLoader(data_wrapper, sampler=sampler, batch_size=32)
    return dataloader
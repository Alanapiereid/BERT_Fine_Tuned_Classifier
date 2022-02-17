import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



def generate_split(X, y):
    X_train, X_interim, y_train, y_interim = train_test_split(X, y, random_state=1, test_size=0.3, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_interim, y_interim, random_state=1, test_size=0.5, stratify=y_interim)
    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess(X_train, X_val, X_test, y_train, y_val, y_test):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokens_train = tokenizer.batch_encode_plus(
        X_train.tolist(),
        max_length = 150,
        pad_to_max_length=True,
        truncation=True
    )

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
        X_val.tolist(),
        max_length = 150,
        pad_to_max_length=True,
        truncation=True
    )

    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        X_test.tolist(),
        max_length = 150,
        pad_to_max_length=True,
        truncation=True
    )

    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(y_train.tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(y_val.tolist())

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(y_test.tolist())
    batch_size = 32
    train_data = TensorDataset(train_seq, train_mask, train_y)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    return train_dataloader, val_dataloader

# def get_loader(X, y):
#     tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
#     tokens = tokenizer.batch_encode_plus(
#     X.tolist(),
#     max_length = 100,
#     pad_to_max_length=True,
#     truncation=True)
#     seq = torch.tensor(tokens['input_ids'])
#     mask = torch.tensor(tokens['attention_mask'])
#     y = torch.tensor(y.tolist())
#     data_wrapper = TensorDataset(seq, mask, y)
#     sampler = RandomSampler(data_wrapper)
#     dataloader = DataLoader(data_wrapper, sampler=sampler, batch_size=32)
#     return dataloader
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

device = torch.device("cuda")

model = AutoModel.from_pretrained('bert-base-uncased')


class BERT_Fine(nn.Module):

    def __init__(self, model):     
      super(BERT_Fine, self).__init__()
      self.model = model
      self.relu =  nn.ReLU()
      self.fc1 = nn.Linear(768,512)
      self.fc2 = nn.Linear(512,2)
      self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
      _, cls_hs = self.bert(sent_id, attention_mask=mask)
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.fc2(x)
      x = self.softmax(x)

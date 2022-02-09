from BERT_class import BERT_Fine
from transformers import AutoModel
import torch.nn as nn
from helpers import get_loader, generate_split
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_pre = AutoModel.from_pretrained('bert-base-uncased')

# freeze layers of BERT model
for param in bert_pre.parameters():
    param.requires_grad = False

# load pre-trained BERT
model = BERT_Fine(bert_pre)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)  
steps = len(dataloader)



def train(epochs, dataloader, steps):
    for epoch in range(epochs):
        for i, (text, label) in enumerate(dataloader):              
            # # Forward
            outputs = model(text)
            loss = loss_func(outputs, label)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{steps}], Loss: {loss.item():.4f}')


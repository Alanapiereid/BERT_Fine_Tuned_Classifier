from pandas import read_csv
from BERT_class import BERT_Fine
from transformers import AutoModel
import torch.nn as nn
from helpers import preprocess, generate_split
import torch

df = read_csv('all_articles.csv', encoding='utf-8')
df.drop(df.columns[2:],axis=1,inplace=True)
df = df.dropna()
df = df.astype({'Label':'int'})

#print(set(df['Label'].values))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_pre = AutoModel.from_pretrained('bert-base-uncased')

# freeze layers of BERT model
for param in bert_pre.parameters():
    param.requires_grad = False

# load pre-trained BERT
model = BERT_Fine(bert_pre)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
X_train, y_train, X_val, y_val, X_test, y_test = generate_split(df['Text'], df['Label'].values)

train_dataloader, val_dataloader = preprocess(X_train, X_val, X_test, y_train, y_val, y_test)

train_steps = len(train_dataloader)
val_steps = len(val_dataloader)

epochs = 10
loss_func = nn.CrossEntropyLoss()
steps = 5

for epoch in range(epochs):
    for i, batch in enumerate(train_dataloader):              
        # # Forward
        sent_id, mask, labels = batch
        model.zero_grad()        
        preds = model(sent_id, mask)
        loss = loss_func(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{steps}], Loss: {loss.item():.4f}')



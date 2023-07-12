import gc
import numpy as np
import torch
from config import *
import pandas as pd
from sklearn.model_selection import train_test_split
from DisasterDataset import DisasterDataset
from torch.utils.data import DataLoader
from train_module import train_val_class
from torch import nn, optim 
from transformers import DistilBertTokenizer,DistilBertForSequenceClassification, TextClassificationPipeline
import shap
from matplotlib import pyplot as plt


df = pd.read_csv(f"{data_dir}/train.csv")
df.fillna("0", inplace=True)
df = df.reset_index(drop=True)
data = df['text'].tolist()
label = df['target'].tolist()

token = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=2)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(),lr=learning_rate)
loss = nn.CrossEntropyLoss()
print(model)
train_data, val_data, train_label, val_label = train_test_split(data, label,
                                                    test_size=0.30,
                                                    random_state=2024,
                                                    stratify=label)

train_ds = DisasterDataset(tokenizer=token, data=train_data, label=train_label)
train_dl = DataLoader(train_ds, 16, shuffle=True)

val_ds = DisasterDataset(tokenizer=token, data=val_data, label=val_label)
val_dl = DataLoader(val_ds, 16, shuffle=True)

prev_epoch_num = 0
best_valid_loss = np.inf
best_valid_roc = 0.0
train_losses = []
valid_losses = []
valid_rocs = []
train_rocs = []

for epoch in range(0, num_epochs):
    torch.cuda.empty_cache()
    print(gc.collect())

    train_loss, train_roc_auc, train_fpr, train_tpr, train_thresholds = train_val_class(epoch, train_dl, model, optimizer, mixed_precision=True, device='cuda', train=True)
    valid_loss, valid_roc_auc, val_fpr, val_tpr, val_thresholds = train_val_class(epoch, val_dl, model, optimizer, mixed_precision=True, device='cuda', train=False)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_rocs.append(train_roc_auc)
    valid_rocs.append(valid_roc_auc)
    # lr_scheduler.step(valid_loss)
    print("#"*20)
    print(f"Epoch {epoch+1} Report:")
    print(f"Validation Loss: {valid_loss :.4f} Validation ROC_AUC: {valid_roc_auc :.4f}")
    best_state = {'model': model.state_dict(), 
    'optim': optimizer.state_dict(), 
    # 'scheduler':lr_reduce_scheduler.state_dict(), 
            # 'scaler': scaler.state_dict(),
    'best_loss':valid_loss, 'best_acc':valid_roc_auc, 'epoch':epoch}
    # best_valid_loss, best_valid_roc = save_model(valid_loss, valid_roc_auc, best_valid_loss, best_valid_roc, best_state, os.path.join(model_dir, model_name))
    print("#"*20)
torch.save(model.state_dict(), 'model.pt')

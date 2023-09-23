import gc
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from config import config_params, model_params
from AccelerometerDataset import AccelerometerDataset
from train_module import train_val_class
import wandb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pandas as pd
from utils import seed_everything, plot_confusion_matrix

wandb.init(
    project="Accelerometer Project",
    config=config_params,
    name=f"{config_params['model_name']}_{config_params['dataset']}"
)

for key, value in config_params.items():
    if isinstance(value, str):
        exec(f"{key} = '{value}'")
    else:
        exec(f"{key} = {value}")

seed_everything(SEED)

dataset = np.load(filepath, allow_pickle=True)
X = dataset["X"]
y = dataset["y"]

folders=dataset["folders"]
y_col_names = list(dataset['y_col_names'])
xdemographics = dataset["demographics"]
# df = pd.read_csv('data/train.csv').reset_index(drop=True)
# label_id = df['Activity']
# label_encoder = LabelEncoder()
# encoded_labels = label_encoder.fit_transform(label_id)
# label_mapping = {label: encoded_label for label, encoded_label in zip(label_id, encoded_labels)}
# df['Activity']= df['Activity'].apply(lambda x:label_mapping[x])
# X = df.drop('Activity',axis=1).values
# X = X[:,:, np.newaxis]
# y = df['Activity'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

label_id = list(set(y[:, 2]))
new_labels = [i for i in range(len(label_id))]
label_dict = {}
for i in new_labels:
    label_dict.update({label_id[i]: new_labels[i]})

y = [label_dict[i] for i in y[:, 2]]
device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.05, random_state=SEED, stratify=y_train
)
train_dataset = AccelerometerDataset(X_train, y_train)
train_dl = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=1)

val_dataset = AccelerometerDataset(X_val, y_val)
val_dl = DataLoader(val_dataset, batch_size=bs, shuffle=True, num_workers=1)

test_dataset = AccelerometerDataset(X_test, y_test)
test_dl = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=1)

model = model_params[config_params['model_name']]
# model = TransfomerModel(num_channels=1, seq_len=X.shape[1],
#                             dim=d_model, head_size=num_heads, num_classes=len(set(label_id)))
# model = seresnet1d(1, len(label_id))
model = model.to(device)
citerion = CrossEntropyLoss(reduction='sum')
wandb.watch(models=model, criterion=citerion, log='parameters')
# Set the optimizer and scheduler
optim = Adam(model.parameters(),
                            lr=lr,
                            eps=eps,
                            weight_decay=weight_decay
                            )
prev_epoch_num = 0
best_valid_loss = np.inf
best_valid_roc = 0.0
train_losses = []
valid_losses = []
valid_rocs = []
train_rocs = []

for epoch in range(0, n_epochs):
    torch.cuda.empty_cache()
    print(gc.collect())

    train_loss, train_lab, train_pred = train_val_class(epoch, train_dl, label_id, 
                                            model, citerion, optim, mixed_precision=True, device='cuda', train=True)
    valid_loss, val_lab, val_pred = train_val_class(epoch, val_dl, label_id, 
                                            model, citerion, optim, mixed_precision=True, device='cuda', train=False)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    # train_rocs.append(train_roc_auc)
    # valid_rocs.append(valid_roc_auc)
    # lr_scheduler.step(valid_loss)
    # val_roc_auc = roc_auc_score(val_lab, val_pred, average='macro')
    print("-"*20)
    print(f"Epoch {epoch+1} Report:")
    print(f"Validation Loss: {valid_loss :.4f}")
    best_state = {'model': model.state_dict(), 
    'optim': optim.state_dict(), 
    # 'scheduler':lr_reduce_scheduler.state_dict(), 
            # 'scaler': scaler.state_dict(),
    'best_loss':valid_loss, 
    # 'best_roc_auc':val_roc_auc, 
    'epoch':epoch}
    # best_valid_loss, best_valid_roc = save_model(valid_loss, valid_roc_auc, best_valid_loss, best_valid_roc, best_state, os.path.join(model_dir, model_name))
    print("-"*20)
    # plot_confusion_matrix(val_lab, val_pred, label_id)
    torch.save(model.state_dict(), 'model.pt')
wandb.finish()
import gc
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from config import congif_params
from AccelerometerDataset import AccelerometerDataset
from model import AccelerometerModel
from train_module import train_val_class
import wandb

wandb.init(
    project="Accelerometer Project",
    config=congif_params
)

for key, value in congif_params.items():
    if isinstance(value, str):
        exec(f"{key} = '{value}'")
    else:
        exec(f"{key} = {value}")

# filepath = "data/delirium.npz"
dataset = np.load(filepath, allow_pickle=True)
X = dataset["X"]
y = dataset["y"]

folders=dataset["folders"]
y_col_names = list(dataset['y_col_names'])
xdemographics = dataset["demographics"]
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

label_id = list(set(y[:, 2]))
new_labels = [i for i in range(len(label_id))]
label_dict = {}
for i in new_labels:
    label_dict.update({label_id[i]: new_labels[i]})

y = [label_dict[i] for i in y[:, 2]]
device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=None
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=SEED, stratify=None
)
train_dataset = AccelerometerDataset(X_train, y_train)
train_dl = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2)

val_dataset = AccelerometerDataset(X_val, y_val)
val_dl = DataLoader(val_dataset, batch_size=bs, shuffle=True, num_workers=2)

test_dataset = AccelerometerDataset(X_test, y_test)
test_dl = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=2)

model = AccelerometerModel(dim=d_model, head_size=num_heads)
wandb.watch(model)
model = model.to(device)
citerion = CrossEntropyLoss(reduction='sum')

# Set the optimizer and scheduler
optim = Adam(model.parameters(),
                            lr=lr,
                            # eps=eps,
                            # weight_decay=weight_decay
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

    train_loss, train_acc = train_val_class(epoch, train_dl, model, citerion, optim, mixed_precision=True, device='cuda', train=True)
    valid_loss, valid_acc = train_val_class(epoch, val_dl, model, citerion, optim, mixed_precision=True, device='cuda', train=False)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    # train_rocs.append(train_roc_auc)
    # valid_rocs.append(valid_roc_auc)
    # lr_scheduler.step(valid_loss)
    print("#"*20)
    print(f"Epoch {epoch+1} Report:")
    print(f"Validation Loss: {valid_loss :.4f} Validation Accuracy: {valid_acc :.4f}")
    best_state = {'model': model.state_dict(), 
    'optim': optim.state_dict(), 
    # 'scheduler':lr_reduce_scheduler.state_dict(), 
            # 'scaler': scaler.state_dict(),
    'best_loss':valid_loss, 'best_acc':valid_acc, 'epoch':epoch}
    # best_valid_loss, best_valid_roc = save_model(valid_loss, valid_roc_auc, best_valid_loss, best_valid_roc, best_state, os.path.join(model_dir, model_name))
    print("#"*20)
torch.save(model.state_dict(), 'model.pt')
wandb.finish()
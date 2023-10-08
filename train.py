import gc
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from catalyst.data.sampler import BalanceClassSampler
from config import config_params, model_params
from AccelerometerDataset import AccelerometerDataset
from losses.focal import criterion_margin_focal_binary_cross_entropy
from train_module import train_val_class
import wandb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import pandas as pd
from matplotlib import pyplot as plt
from utils import classification_report_df, save_model, seed_everything, plot_confusion_matrix, model_summary

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
target_outcome = "brain_status"
# for folder_idx in range(len(folders)):
folder_idx = len(folders) - 1
print(f"Splitting with Folder {folder_idx}")
test_idx = folders[folder_idx][0]
train_idx = folders[folder_idx][1]
X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx, y_col_names.index(target_outcome)]
y_test = y[test_idx, y_col_names.index(target_outcome)]

# Create Pandas Series for both lists
series1 = pd.Series(y_train, name='Train')
series2 = pd.Series(y_test, name='Test')

plt.figure(figsize=(12, 9), dpi=100)
ax = series1.value_counts().sort_index().plot(kind='bar', alpha=0.7, label='Train', color='b')
series2.value_counts().sort_index().plot(kind='bar', alpha=0.7, label='Test', color='r', ax=ax)
plt.xlabel('Elements')
plt.ylabel('Frequency')
plt.title('Element Frequency Histogram for Two Datasets')
plt.legend()
plt.savefig('data_hist.png')
wandb.log({"Data Histogram": wandb.Image("data_hist.png")})
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
# X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
label_id = list(set(y[:, 2]))
new_labels = [i for i in range(len(label_id))]
label_dict = {}
for i in new_labels:
    label_dict.update({label_id[i]: new_labels[i]})

y_train = [label_dict[i] for i in y_train]
y_test = [label_dict[i] for i in y_test]
device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
# )
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.35, random_state=SEED, stratify=y_train
# )
train_dataset = AccelerometerDataset(X_train, y_train)
if sampling_mode is not None:
    sampler = BalanceClassSampler(labels=train_dataset.get_labels(), mode=sampling_mode)
else: sampler = None

train_dl = DataLoader(train_dataset, batch_size=bs, sampler=sampler,
                shuffle=False, num_workers=2)

val_dataset = AccelerometerDataset(X_test, y_test, )
val_dl = DataLoader(val_dataset, batch_size=bs, shuffle=True, num_workers=2)

test_dataset = AccelerometerDataset(X_test, y_test)
test_dl = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=2)

model = model_params[config_params['model_name']]
total_params = sum(p.numel() for p in model.parameters())
wandb.log({'# Model Params': total_params})
model = model.to(device)
citerion = CrossEntropyLoss(reduction='sum')
# citerion = criterion_margin_focal_binary_cross_entropy
wandb.watch(models=model, criterion=citerion, log='parameters')
data = torch.randn(2, num_channels, seq_len).to(device)
optim = Adam(model.parameters(), lr=lr)
lr_scheduler = ReduceLROnPlateau(optim, mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=True)

prev_epoch_num = 0
best_valid_loss = np.inf
best_state = None
best_f1 = 0.0
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
    
    lr_scheduler.step(valid_loss)
    train_f1_score = f1_score(train_lab, train_pred, average='macro')
    val_f1_score = f1_score(val_lab, val_pred, average='macro')
    wandb.log({"Train Loss": train_loss, "Epoch": epoch})
    wandb.log({"Validation Loss": valid_loss, "Epoch": epoch})
    wandb.log({"Train F1 Score": train_f1_score, "Epoch": epoch})
    wandb.log({"Validation F1 Score": val_f1_score, "Epoch": epoch})
    
    print("="*50)
    print(f"Epoch {epoch+1} Report:")
    print(f"Validation Loss: {valid_loss :.4f} Validation F1 Score : {val_f1_score :.4f}")
    model_dict = {'model': model.state_dict(), 
    'optim': optim.state_dict(), 
    # 'scheduler':lr_reduce_scheduler.state_dict(), 
            # 'scaler': scaler.state_dict(),
    'best_loss':valid_loss, 
    'best_f1_score':val_f1_score, 
    'epoch':epoch}
    best_valid_loss, best_f1, best_state = save_model(valid_loss, 
                best_valid_loss, val_f1_score, best_f1, model_dict, 
                model_name, 'model_dir')
    print("="*70)
    
best_state = torch.load(f"model_dir/{model_name}_f1.pth")
print(f"Best Validation result was found in epoch {best_state['epoch']}\n")
print(f"Best Validation Loss {best_state['best_loss']}\n")
print(f"Best Validation F1 score {best_state['best_f1_score']}\n")
print("Loading best model")
model.load_state_dict(best_state['model'])
test_loss, test_lab, test_pred = train_val_class(-1, test_dl, label_id, 
                                            model, citerion, optim, 
                                            mixed_precision=True, device='cuda',
                                              train=False)
test_f1_score = f1_score(test_lab, test_pred, average='macro')
    
print(f"Test Loss: {test_loss :.4f} Test F1 Score : {test_f1_score :.4f}")
wandb.log({"Test Loss":test_loss, "Test F1": test_f1_score})
plot_confusion_matrix(test_lab, test_pred, label_dict)
wandb.log({"Confusion Matrix": wandb.Image("conf.png")})
label_names = [k for k,_ in label_dict.items()]
report = classification_report(test_lab, test_pred, target_names=label_names)
print("\nClassification Report:")
print(report)
report = classification_report_df(test_lab, test_pred, label_names)
wandb_tbl = wandb.Table(data=report)
wandb.log({"Classification Report": wandb_tbl})
wandb.finish()
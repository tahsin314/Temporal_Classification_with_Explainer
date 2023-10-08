import os

from sklearn import metrics
from sklearn.model_selection import train_test_split
from config import *
import random
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import cv2
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm as T
import pandas as pd
from torch import nn
# from gradcam.gradcam import GradCAM, GradCAMpp

def onehot_encoder(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data(filepath, folder_idx= 4, type='acuity'):
    dataset = np.load(filepath, allow_pickle=True)
    X = dataset["X"]
    y = dataset["y"]
    y_col_names = list(dataset['y_col_names'])
    xdemographics = dataset["demographics"]
    if type == "delirium":
        label_id = list(set(y[:, 2]))
        new_labels = [i for i in range(len(label_id))]
        label_dict = {}
        for i in new_labels:
            label_dict.update({label_id[i]: new_labels[i]})
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
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
        
    elif type == "acuity":
        X = np.vstack(X[:, 0]).reshape(len(X), 144000, -1)
        label_id = list(set(y[:, 0]))
        new_labels = [i for i in range(len(label_id))]
        label_dict = {}
        for i in new_labels:
            label_dict.update({label_id[i]: new_labels[i]})
        target_outcome = "timestamp" # Should be acuity
        folders=dataset["train_kfold"]
        print(f"Splitting with Folder {folder_idx}")
        test_idx = folders[folder_idx][0]
        train_idx = folders[folder_idx][1]
        
        train_idx = folders[folder_idx][0]
        val_idx = folders[folder_idx][1]
        test_idx = list(dataset['test'])
        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]
        y_train = y[train_idx, y_col_names.index(target_outcome)]
        y_train = [label_dict[i] for i in y_train]
        y_val = y[val_idx, y_col_names.index(target_outcome)]
        y_val = [label_dict[i] for i in y_val]
        y_test = y[test_idx, y_col_names.index(target_outcome)]
        y_test = [label_dict[i] for i in y_test]    

    return X_train, X_val, X_test, y_train, y_val, y_test, label_id, label_dict


def plot_confusion_matrix(actual_labels, predictions, label_dict):
    id2label = {v:k for k,v in label_dict.items()}
    actual_labels = [id2label[i] for i in actual_labels]
    predictions = [id2label[i] for i in predictions]
    labels = [k for k,_ in label_dict.items()]
    cm = confusion_matrix(actual_labels, predictions, labels=labels)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('conf.png')
    plt.close()

def model_summary(model, sample_input):
    # Initialize variables to collect layer information
    layers_info = []
    x = sample_input
    for name, layer in model.named_children():
        input_shape = tuple(x.shape) if x is not None else None
        if isinstance(layer, nn.Linear):
            x = x.view(x.size(0), -1)  # Example reshape operation
        x = layer(x)
        output_shape = tuple(x.shape) if hasattr(layer, 'weight') else None
        num_params = sum(p.numel() for p in layer.parameters())
        layers_info.append([name, input_shape, output_shape, num_params])

    # Compute the total number of parameters
    total_params = sum(num_params for _, _, _, num_params in layers_info)
    columns=["Layer Name", "Input Shape", "Output Shape", "Param #"]
    # Create a Pandas DataFrame
    df = pd.DataFrame(layers_info, columns=columns)
    return df, columns, total_params

def save_model(valid_loss, best_valid_loss, valid_f1, best_valid_f1, model_dict, model_name, save_dir):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if valid_loss<best_valid_loss:
        print(f'Validation loss has decreased from:  {best_valid_loss:.4f} to: {valid_loss:.4f}')
        best_valid_loss = valid_loss
        save_file_path = os.path.join(save_dir, f'{model_name}_loss.pth')
        torch.save(model_dict, save_file_path)
    
    if valid_f1>best_valid_f1:
        print(f'Validation f1 has increased from:  {best_valid_f1:.4f} to: {valid_f1:.4f}')
        best_valid_f1 = valid_f1
        save_file_path = os.path.join(save_dir, f'{model_name}_f1.pth')
        torch.save(model_dict, save_file_path)

    return best_valid_loss, best_valid_f1, model_dict 

def classification_report_df(y_true, y_pred, label_names):
    report = metrics.classification_report(y_true, y_pred, 
                                           target_names=label_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.round(3)        
    df_report = df_report.astype({'support': int})    
    df_report.loc['accuracy',['precision','recall','support']] = [None,None,df_report.loc['macro avg']['support']]
    return df_report
    

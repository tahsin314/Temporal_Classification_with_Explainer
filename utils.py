import os
import random
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import torch

def save_model(valid_loss, valid_roc_auc, best_valid_loss, best_valid_roc_auc, best_state, epoch, save_dir):
    if valid_loss<best_valid_loss:
        print(f'Validation loss has decreased from:  {best_valid_loss:.4f} to: {valid_loss:.4f}')
        best_valid_loss = valid_loss

    if valid_roc_auc>best_valid_roc_auc:
        print(f'Validation roc_auc has increased from:  {best_valid_roc_auc:.4f} to: {valid_roc_auc:.4f}')
        best_valid_roc_auc = valid_roc_auc


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
    torch.save(best_state, save_file_path)

    return best_valid_loss, best_valid_roc_auc
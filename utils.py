import os
from config import *
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
import cv2
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm as T
# from gradcam.gradcam import GradCAM, GradCAMpp


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def plot_confusion_matrix(actual_labels, predictions, labels):
    cm = confusion_matrix(actual_labels, predictions, labels=labels)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('conf.png')
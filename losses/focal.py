# Courtesy: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/155201
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def criterion_margin_focal_binary_cross_entropy(logit, truth):
    weight_pos=2
    weight_neg=1
    gamma=2
    margin=0.2
    em = np.exp(margin)
    logit = F.softmax(logit, dim=1)
    logit = logit.view(-1)
    truth = truth.view(-1)
    log_pos = -F.logsigmoid( logit)
    log_neg = -F.logsigmoid(-logit)

    log_prob = truth*log_pos + (1-truth)*log_neg
    prob = torch.exp(-log_prob)
    margin = torch.log(em +(1-em)*prob)

    weight = truth*weight_pos + (1-truth)*weight_neg
    loss = margin + weight*(1 - prob) ** gamma * log_prob
    loss = loss.sum()
    return loss
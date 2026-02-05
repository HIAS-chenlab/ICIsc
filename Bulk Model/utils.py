import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def auc_score(logits, labels):
    prob = torch.sigmoid(logits).cpu().numpy()
    labels = labels.cpu().numpy()
    return roc_auc_score(labels, prob)

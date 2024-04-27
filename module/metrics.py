import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

def AUC_and_EER(loader, model, device):
    model.eval()
    with torch.no_grad():
        y_true = []
        y_score = []
        for (x, y) in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            y_hat = torch.softmax(y_hat, dim=1)[:,1]
            y_true.extend(y.cpu().numpy())
            y_score.extend(y_hat.cpu().numpy())
        return roc_auc_score(y_true, y_score), eer(y_true, y_score)
    
def eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchmetrics.regression import ( PearsonCorrCoef, 
                                    ConcordanceCorrCoef, 
                                    MeanSquaredError,
                                    MeanAbsoluteError,
                                    ExplainedVariance, 
                                    SpearmanCorrCoef)  
import numpy as np


def PearsonCorrCoef_loss(x, y, slide_ids):
    mean_x = torch.mean(x, dim=0, keepdim=True)  # (1, 512)
    mean_y = torch.mean(y, dim=0, keepdim=True)  # (1, 512)

    xm = x - mean_x  
    ym = y - mean_y 

    r_num = torch.sum(xm * ym, dim=0)  # (512,)
    r_den = torch.sqrt(torch.sum(xm ** 2, dim=0)) * torch.sqrt(torch.sum(ym ** 2, dim=0))  # (512,)

    pearson_corr = r_num / (r_den + 1e-8)  # ゼロ除算を防ぐ
    pearson_corr = torch.nan_to_num(pearson_corr, nan=-1)  # NaN を -1 に変換
    return 1-pearson_corr

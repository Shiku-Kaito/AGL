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

def Prime_PearsonCorrCoef_loss(x, y, slide_ids):
    mean_x = torch.mean(x, dim=0, keepdim=True)  # (1, 512)
    mean_y = torch.mean(y, dim=0, keepdim=True)  # (1, 512)

    xm = x - mean_x  
    ym = y - mean_y 

    r_num = torch.sum(xm * ym, dim=0)  # (512,)
    r_den = torch.sqrt(torch.sum(xm ** 2, dim=0)) * torch.sqrt(torch.sum(ym ** 2, dim=0))  # (512,)

    pearson_corr = r_num / (r_den + 1e-8)  # ゼロ除算を防ぐ
    pearson_corr = torch.nan_to_num(pearson_corr, nan=-1)  # NaN を -1 に変換
    return torch.sum(pearson_corr)


def patch_weighted_PearsonCorrCoef_loss(x, y, loss_weight, slide_ids):
    # 重み付き平均
    mean_x = torch.sum(loss_weight * x, dim=0, keepdim=True)  # (1, 512)
    mean_y = torch.sum(loss_weight * y, dim=0, keepdim=True)  # (1, 512)

    # 中心化
    xm = x - mean_x  # (バッチサイズ, 512)
    ym = y - mean_y  # (バッチサイズ, 512)

    # 共分散の計算（分子）
    r_num = torch.sum(loss_weight * xm * ym, dim=0)  # (512,)

    # 分散の計算（分母）
    r_den_x = torch.sqrt(torch.clamp(torch.sum(loss_weight * (xm ** 2), dim=0), min=1e-8))  # (512,)
    r_den_y = torch.sqrt(torch.clamp(torch.sum(loss_weight * (ym ** 2), dim=0), min=1e-8))  # (512,)

    # Pearson 相関係数の計算
    pearson_corr = r_num / (r_den_x * r_den_y + 1e-8)  # ゼロ除算を防ぐ
    pearson_corr = torch.nan_to_num(pearson_corr, nan=-1)  # NaN を -1 に変換
    return 1-pearson_corr



def task_Weighted_PearsonCorrCoef_loss(x, y, loss_weight, slide_ids):
    mean_x = torch.mean(x, dim=0, keepdim=True)  # (1, 512)
    mean_y = torch.mean(y, dim=0, keepdim=True)  # (1, 512)

    xm = x - mean_x  
    ym = y - mean_y 

    r_num = torch.sum(xm * ym, dim=0)  # (512,)
    r_den = torch.sqrt(torch.sum(xm ** 2, dim=0)) * torch.sqrt(torch.sum(ym ** 2, dim=0))  # (512,)

    pearson_corr = r_num / (r_den + 1e-8)  # ゼロ除算を防ぐ
    pearson_corr = torch.nan_to_num(pearson_corr, nan=-1)  # NaN を -1 に変換
    return torch.sum(pearson_corr*loss_weight)


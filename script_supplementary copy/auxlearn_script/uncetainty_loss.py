import math

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, loss_pcc):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        # self.log_vars = torch.zeros((task_num))

        # self.model = model
        self.loss_pcc = loss_pcc

        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential()
        # self.resnet18 = nn.Sequential(*list(self.resnet18 .children())[:-2])
        self.regressor = nn.Linear(512, task_num)

    def forward(self, patch, st, slide_ids):
        img_feat = self.feature_extractor(patch) # B x 512 x 7 x 7
        
        y = self.regressor(img_feat)
        if y.shape[1]!=1:
            y = y.squeeze()

        pcc_loss = self.loss_pcc(y, st, np.array(slide_ids))

        loss = torch.mean(torch.exp(-self.log_vars) * pcc_loss + self.log_vars)
        
        # losses = []
        # for idx in range(len(pcc_loss)):
        #     precision = torch.exp(-self.log_vars[idx])
        #     loss = precision * pcc_loss[idx] + self.log_vars[idx]
        #     losses.append(loss)
        # losses = torch.stack(losses)
        # loss = torch.mean(losses)
        return loss, self.log_vars.data.tolist(), y
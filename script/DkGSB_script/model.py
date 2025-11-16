import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange


class ST_estimater(nn.Module):
    def __init__(self, args, num_outputs):
        super().__init__()
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential()
        self.regressor = nn.Linear(512, num_outputs)
        
    def forward(self, img):
        img_feat = self.feature_extractor(img)
        
        y = self.regressor(img_feat).squeeze()
        return {"y": y, "feat": img_feat}

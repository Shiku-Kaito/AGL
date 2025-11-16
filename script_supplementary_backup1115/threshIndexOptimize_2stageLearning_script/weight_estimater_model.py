import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from torch import Tensor
import torch.nn.functional as F
import numpy as np


class MAXL_weight_estimater(nn.Module):
    def __init__(self, args, train_highly_gene, hvg_score, all_gene_name):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.temper = args.temper
        self.scaling_factor = 10
        
        train_highly_gene_idx = np.where(np.isin(all_gene_name, train_highly_gene))[0]

        self.hvg_score = torch.tensor(hvg_score).to(args.device)
        self.hvg_score = self.hvg_score[train_highly_gene_idx]
        self.sorted_hvg_score, self.sorted_hvg_idx = torch.sort(self.hvg_score, descending=True)
        
        # ソート後の順位を元のインデックス順に並び替える
        self.index = torch.empty_like(self.sorted_hvg_idx).to(args.device)
        self.index[self.sorted_hvg_idx] = torch.arange(len(self.sorted_hvg_idx)).to(args.device)
        
        # self.index = torch.arange(0, len(train_highly_gene), dtype=torch.float32)
        self.unmorm_index = self.index
        self.index = (self.index - self.index.min()) / (self.index.max() - self.index.min())
        
        mean_idx = torch.mean(self.index).cpu().detach().numpy().item()
        self.thresh = nn.Parameter(torch.tensor(mean_idx))
        
        # self.thresh = nn.Parameter(torch.tensor(1e-8))
        
    def forward(self, y, eval_gene_idx, train_highly_gene_idx, train_low_gene_idx):  
        # thresh = F.relu(self.thresh)
        thresh_list = self.thresh.repeat(len(train_highly_gene_idx))
        mask = torch.cat([thresh_list.unsqueeze(1), self.index.unsqueeze(1)], dim=1)
        # mask = mask / mask.sum(dim=1, keepdim=True)
        mask = self.softmax(mask/self.temper)
        mask = mask[:, 0]
        
        ones_tensor = torch.ones(len(eval_gene_idx)+len(train_highly_gene_idx)+len(train_low_gene_idx) , device=y.device)  # yと同じデバイスに作成
        ones_tensor[train_highly_gene_idx] = mask
        
        diff = torch.abs(self.index - self.thresh)
        k = self.unmorm_index[torch.argmin(diff)]
        return {"w": ones_tensor, "row_w": ones_tensor, "thresh": self.thresh, "k": k}

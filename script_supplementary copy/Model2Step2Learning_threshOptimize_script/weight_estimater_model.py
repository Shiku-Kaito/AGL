import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# class MAXL_weight_estimater(nn.Module):
#     def __init__(self, args, train_highly_gene, hvg_score):
#         super().__init__()
#         self.thresh = nn.Parameter(torch.tensor(0.0))
#         self.softmax = nn.Softmax(dim=1)
#         self.temper = args.temper
#         self.scaling_factor = 10
        
#         self.hvg_score = torch.tensor(hvg_score).to(args.device)
        
#     def forward(self, y, eval_gene_idx, train_highly_gene_idx, train_low_gene_idx):  
#         hvg_score = self.hvg_score[train_highly_gene_idx]
#         thresh_list = self.thresh.repeat(len(train_highly_gene_idx))
#         mask = torch.cat([hvg_score.unsqueeze(1), thresh_list.unsqueeze(1)], dim=1)
#         mask = self.softmax(mask/self.temper)
#         mask = mask[:, 0]
        
#         ones_tensor = torch.ones(len(eval_gene_idx)+len(train_highly_gene_idx)+len(train_low_gene_idx) , device=y.device)  # yと同じデバイスに作成
#         ones_tensor[train_highly_gene_idx] = mask
#         return {"w": ones_tensor, "row_w": ones_tensor}

class MAXL_weight_estimater(nn.Module):
    def __init__(self, args, train_highly_gene, hvg_score, all_gene_name):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.temper = args.temper
        self.scaling_factor = 10
        
        train_highly_gene_idx = np.where(np.isin(all_gene_name, train_highly_gene))[0]
        
        # self.hvg_score = torch.tensor(hvg_score).to(args.device)
        # self.hvg_score = (self.hvg_score - self.hvg_score.min()) / (self.hvg_score.max() - self.hvg_score.min() + 1e-8)
        
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

# import matplotlib.pyplot as plt

# logits = torch.tensor([-2, 0.0, 2])
# taus = [1.0, 0.1, 0.01]
# n_samples = 10000

# # Plot
# fig, axes = plt.subplots(len(logits), len(taus), figsize=(20, 10), sharex=False, sharey=True)

# for i, logit in enumerate(logits):
#     for j, tau in enumerate(taus):
#         logit_list = torch.full((20000,), logit)
#         samples = gumbel_sigmoid(logit_list, tau)
#         ax = axes[i, j]
#         ax.hist(samples.numpy(), bins=50, density=False, alpha=0.7)
#         ax.set_title(f"logit={logit.item()}, tau={tau}")
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 20000)
        
# plt.tight_layout()
# plt.suptitle("Gumbel-Sigmoid Output Distributions for Different Tau Values", y=1.02)
# plt.show()
# plt.savefig("/media/user/HD-QHAU3/ST_estimation_data/result/img/TENX152/5-fold/w_pretrain_lr=3e-5/train_top_genes=2000/higher_task_weight_weight_estimater_Lr=0.00300000_sumplingNum=360_iniWgt=0.0_GumbelSigmoid_temper=0.010_UpdateEpoch=20_strEpoch=100_loss=pcc/pn.png")
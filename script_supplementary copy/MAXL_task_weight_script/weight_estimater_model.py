# import torch
# import torch.nn as nn
# from torchvision.models import resnet18
# import math
# from torch.nn.utils.rnn import pad_sequence
# from einops import rearrange

# class MAXL_weight_estimater(nn.Module):
#     def __init__(self, args, num_outputs, prim_gene_num):
#         super().__init__()
#         self.lambdas = nn.Parameter(torch.randn(num_outputs))
#         self.prim_gene_num = prim_gene_num
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, patch):
#         lambdas = self.sigmoid(self.lambdas)
#         ones_tensor = torch.ones(self.prim_gene_num , device=patch.device)  # yと同じデバイスに作成
#         lambdas = torch.cat((lambdas, ones_tensor), dim=0)  # shape [32, 541]
#         lambdas = lambdas / lambdas.sum(dim=0, keepdim=True)
#         return {"w": lambdas}
    
    
import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange

class MAXL_weight_estimater(nn.Module):
    def __init__(self, args, num_outputs, prim_gene_num):
        super().__init__()
        # self.lambdas = nn.Parameter(torch.randn(num_outputs))
        # uniform = torch.distributions.Uniform(-2, 2)
        # self.lambdas = nn.Parameter(uniform.sample((num_outputs,)))
        
        
        # self.lambdas = nn.Parameter(torch.ones(num_outputs))  # 初期値を1に設定
        # self.lambdas = nn.Parameter(torch.zeros(num_outputs))  # 初期値を1に設定
        
        # self.lambdas = nn.Parameter(torch.full((num_outputs,), 1e-8))
        # self.lambdas = nn.Parameter(torch.full((num_outputs,), -2.0))
        
        # self.lambdas = nn.Parameter(torch.full((num_outputs,), 2.0))
        
        self.lambdas = nn.Parameter(torch.full((num_outputs,), 1.0))
        
        
        self.prim_gene_num = prim_gene_num
        # self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()
        

    def forward(self, y, eval_gene_idx, aux_gene_idx):  
        ones_tensor = torch.ones(len(eval_gene_idx)+len(aux_gene_idx) , device=y.device)  # yと同じデバイスに作成
        ones_tensor[aux_gene_idx] = self.ReLU(self.lambdas)
        return {"w": ones_tensor}
import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from torch import Tensor


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class MAXL_weight_estimater(nn.Module):
    def __init__(self, args, train_highly_gene, train_low_gene):
        super().__init__()
        self.highly_variablegene_lambdas = nn.Parameter(torch.full((len(train_highly_gene),), 0.0))
        self.sigmoid = nn.Sigmoid()
        self.temper = args.temper
        self.scaling_factor = 10
        

    def forward(self, y, eval_gene_idx, train_highly_gene_idx, train_low_gene_idx):  
        ones_tensor = torch.ones(len(eval_gene_idx)+len(train_highly_gene_idx)+len(train_low_gene_idx) , device=y.device)  # yと同じデバイスに作成
        ones_tensor[train_highly_gene_idx] = self.sigmoid(self.highly_variablegene_lambdas)
        # ones_tensor[train_highly_gene_idx] = gumbel_sigmoid(logits=self.scaling_factor*self.highly_variablegene_lambdas, tau=self.temper, hard=False)
        
        temp_tensor = torch.ones(len(eval_gene_idx)+len(train_highly_gene_idx)+len(train_low_gene_idx) , device=y.device)  # yと同じデバイスに作成
        temp_tensor[train_highly_gene_idx] = self.highly_variablegene_lambdas
        return {"w": ones_tensor, "row_w": temp_tensor}

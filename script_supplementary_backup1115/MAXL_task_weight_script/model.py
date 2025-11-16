import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange


class MAXL_ST_estimater(nn.Module):
    def __init__(self, args, num_outputs):
        super().__init__()
        # self.feature_extractor, _ = create_model_from_pretrained("conch_ViT-B-16", device=args.device, force_image_size=512, checkpoint_path="./result/pretrained_conch/checkpoints/conch/pytorch_model.bin")
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Sequential()
        # self.resnet18 = nn.Sequential(*list(self.resnet18 .children())[:-2])
        self.regressor = nn.Linear(512, num_outputs)
        
    def forward(self, img):
        img_feat = self.feature_extractor(img) # B x 512 x 7 x 7
        
        y = self.regressor(img_feat).squeeze()
        return {"y": y, "feat": img_feat}

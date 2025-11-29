import os
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch.nn.functional as F
from statistics import mean
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from skimage import io
import glob
import torchvision.transforms as T
from torch.utils.data import WeightedRandomSampler
import random
import copy
from utils import *
import h5py
import scanpy as sc

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomApply, RandomRotation, ToPILImage

def image_transform(image_size, is_train):
    if is_train:
        transforms = Compose([
                ToPILImage(),
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomApply([RandomRotation((90, 90))]),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
    else:
        transforms = Compose([
            ToPILImage(),
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    return transforms

class Hest_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, adata, patch, patch_idx, slide_id, is_train):
        np.random.seed(args.seed)
        self.adata = adata
        self.patch = patch
        self.transform = image_transform(224, is_train)
        self.patch_idx = patch_idx
        self.len = len(self.patch_idx)
        self.args = args
        self.slide_id = args.dataset
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # load img
        patch_adata = self.adata[self.patch_idx[idx]]
        patch_idx = int(patch_adata.obs["patch_index"].values[0])
        patch = self.patch[patch_idx]
        patch = self.transform(patch)
        # load st
        st = patch_adata.X.toarray()
        st = torch.tensor((np.array(st)[0]))
        return {"patch": patch, "st": st, "slide_ids": self.slide_id, "patch_idx": patch_idx}


def load_data(args): 
    # All genes (target + aux genes)
    if args.module=="PGL":
        all_gene_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_top50_matching_patch_adata.h5ad' % (args.input_path, args.dataset))
    elif args.module=="AGL":
        all_gene_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_topall_matching_patch_adata.h5ad' % (args.input_path, args.dataset))
    all_gene_name = np.array(all_gene_adata.var.index)
    all_gene_num = all_gene_name.shape[0]
    
    # Target genes
    target_gene_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_top%d_matching_patch_adata.h5ad' % (args.input_path, args.dataset, args.target_genes_num))
    target_gene_name = np.array(target_gene_adata.var.index)

    # Load img data
    with h5py.File("%s/data/org/hest_data/patches/%s.h5" % (args.input_path, args.dataset), "r") as h5file:
        patch = h5file["img"][:]
    patch_idx_dict = np.load("%s/data/preprocessed_data/hest_1slide/%s_cv/fold=%d.npy" % (args.input_path, args.dataset, args.fold), allow_pickle=True).item()
    
    # Train / valdation / test dataloader
    train_dataset = Hest_Dataset(args=args, adata=all_gene_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataset = Hest_Dataset(args=args, adata=all_gene_adata, patch=patch, patch_idx=patch_idx_dict["validation"], slide_id=args.dataset, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)  
    test_dataset = Hest_Dataset(args=args, adata=all_gene_adata, patch=patch, patch_idx=patch_idx_dict["test"], slide_id=args.dataset, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)    
    return train_loader, val_loader, test_loader, {"all": all_gene_name, "target": target_gene_name}, all_gene_num

def load_meta_learning_data(args): 
    # All genes (target + aux genes)
    all_gene_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_topall_matching_patch_adata.h5ad' % (args.input_path, args.dataset))
    sc.pp.highly_variable_genes(all_gene_adata, n_top_genes=len(all_gene_adata.var.index))  
    all_gene_name = np.array(all_gene_adata.var.index)
    all_gene_num = all_gene_name.shape[0]
    
    # Target genes
    target_gene_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_top50_matching_patch_adata.h5ad' % (args.input_path, args.dataset))
    target_gene_name = np.array(target_gene_adata.var.index)
    
    # Auxiliary genes
    aux_gene_name =  all_gene_name[np.where(~np.isin(all_gene_name, target_gene_name))[0]]
    
    # Obtain prior knowledge-based ranking gene
    hvg_scores = all_gene_adata.var['dispersions_norm']
    hvg_scores = np.array(hvg_scores)
    
    # Load img data
    with h5py.File("%s/data/org/hest_data/patches/%s.h5" % (args.input_path, args.dataset), "r") as h5file:
        patch = h5file["img"][:]
    patch_idx_dict = np.load("%s/data/preprocessed_data/hest_1slide/%s_cv/fold=%d.npy" % (args.input_path, args.dataset, args.fold), allow_pickle=True).item()
    
    # Train / valdation / test dataloader
    train_dataset = Hest_Dataset(args=args, adata=all_gene_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataset = Hest_Dataset(args=args, adata=all_gene_adata, patch=patch, patch_idx=patch_idx_dict["validation"], slide_id=args.dataset, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)  
    test_dataset = Hest_Dataset(args=args, adata=all_gene_adata, patch=patch, patch_idx=patch_idx_dict["test"], slide_id=args.dataset, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)    
 
    # Bi-level optimization dataloader
    meta_train_dataset = Hest_Dataset(args=args, adata=all_gene_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=True)
    meta_train_loader = torch.utils.data.DataLoader(meta_train_dataset, shuffle=True, batch_size=args.bilevel_sampling_num, num_workers=args.num_workers)
    meta_val_dataset = Hest_Dataset(args=args, adata=all_gene_adata, patch=patch, patch_idx=patch_idx_dict["validation"], slide_id=args.dataset, is_train=True)
    meta_val_loader = torch.utils.data.DataLoader(meta_val_dataset, batch_size=args.bilevel_sampling_num, shuffle=True,  num_workers=args.num_workers)     
    
    return {"train": train_loader, "meta_train_sampler": meta_train_loader, "meta_val_sampler": meta_val_loader}, val_loader, test_loader, {"all": all_gene_name,  "target": target_gene_name, "aux": aux_gene_name}, all_gene_num, hvg_scores
    

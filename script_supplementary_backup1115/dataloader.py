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

# def _convert_to_rgb(image):
#     return image.convert('RGB')

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
            # CenterCrop(image_size),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    return transforms


class Hest_9slide_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, adata, patch_dict, slide_ids, is_train):
        np.random.seed(args.seed)
        self.adata = adata[adata.obs['batch'].isin(slide_ids)]
        self.patch_dict = patch_dict
        self.len = len(self.adata)
        
        self.transform = image_transform(224, is_train)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        patch_adata = self.adata[idx]
        slide_id = patch_adata.obs["batch"].values[0]
        patch_idx = int(patch_adata.obs["patch_index"].values[0])
        
        patch = self.patch_dict[slide_id][patch_idx]
        patch = self.transform(patch)
        
        st = patch_adata.X
        st = torch.tensor((np.array(st)[0]))
        return {"patch": patch, "st": st, "slide_ids": slide_id}


def hest_9slide_load(args):  # LIMUC
    ######### load data #######
    preprocessed_adata = sc.read("/home/user/data/mnt/ST_estimation/data/preprocessed_data/hest_9slide/matching_patch_adata.h5ad")
    patch_dict = {}
    cv_dict = np.load("/home/user/data/mnt/ST_estimation/data/preprocessed_data/hest_9slide/cv/fold=%d.npy" % args.fold, allow_pickle=True).item()
    slide_id_list = list(cv_dict['train']) + list(cv_dict['validation']) + list(cv_dict['test'])
    for slide_id in slide_id_list:        
        with h5py.File("/home/user/data/mnt/ST_estimation/data/org/hest_data/patches/%s.h5" % slide_id, "r") as h5file:
            patch_dict[slide_id] = h5file["img"][:]

    slide_label_count = []
    train_adda = preprocessed_adata[preprocessed_adata.obs['batch'].isin(cv_dict["train"])]
    for slide_id in cv_dict["train"]:
        slide_label_count.append(len(train_adda[train_adda.obs["batch"]==slide_id]))
    slide_label_count = np.array(slide_label_count)
    class_weight = 1 / slide_label_count    
    sample_weight = []
    # [class_weight[np.where(cv_dict["train"]==train_adda[i].obs["batch"].values[0])[0]] for i in range(len(train_adda))]
    for i in range(len(slide_label_count)):
        sample_weight.extend(np.full(slide_label_count[i], class_weight[i], dtype=np.float64))
    sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(train_adda), replacement=True)
    
    train_dataset = Hest_9slide_Dataset(args=args, adata=preprocessed_adata, patch_dict=patch_dict, slide_ids=cv_dict["train"], is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataset = Hest_9slide_Dataset(args=args, adata=preprocessed_adata, patch_dict=patch_dict, slide_ids=cv_dict["validation"], is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)  
    test_dataset = Hest_9slide_Dataset(args=args, adata=preprocessed_adata, patch_dict=patch_dict, slide_ids=cv_dict["test"], is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)    
    return train_loader, val_loader, test_loader


class Hest_1slide_Dataset(torch.utils.data.Dataset):
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
        patch_adata = self.adata[self.patch_idx[idx]]
        # slide_id = patch_adata.obs["batch"].values[0]
        patch_idx = int(patch_adata.obs["patch_index"].values[0])
        
        patch = self.patch[patch_idx]

        if self.args.img_or_token_feat_or_feat == "img":
            patch = self.transform(patch)
        
        if self.args.dataset == "TENX152" or self.args.dataset == "MISC131" or self.args.dataset == "MISC132" or self.args.dataset == "MISC133" or self.args.dataset == "TENX61" or self.args.dataset == "TENX62" or self.args.dataset == "TENX65" or self.args.dataset == "TENX89":
            st = patch_adata.X.toarray()
        else:
            st = patch_adata.X
        st = torch.tensor((np.array(st)[0]))
        return {"patch": patch, "st": st, "slide_ids": self.slide_id, "patch_wise_unmask_gene": [], "patch_idx": patch_idx}


def hest_1slide_load(args):  # LIMUC
    ######### load data #######
    if args.train_highly_variable_top_genes!=None:
        preprocessed_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_%s%d_matching_patch_adata.h5ad' % (args.input_path, args.dataset, args.gene_select_type, args.train_highly_variable_top_genes))
    elif args.train_highly_variable_top_genes==None:
        preprocessed_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_topall_matching_patch_adata.h5ad' % (args.input_path, args.dataset))
    
    eval_gene_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_top%d_matching_patch_adata.h5ad' % (args.input_path, args.dataset, args.eval_highly_variable_top_genes))
    train_gene_name = np.array(preprocessed_adata.var.index)
    eval_gene_name = np.array(eval_gene_adata.var.index)
    
    all_gene_num = train_gene_name.shape[0]

    with h5py.File("%s/data/org/hest_data/patches/%s.h5" % (args.input_path, args.dataset), "r") as h5file:
        patch = h5file["img"][:]

            
    patch_idx_dict = np.load("%s/data/preprocessed_data/hest_1slide/%s_cv/fold=%d.npy" % (args.input_path, args.dataset, args.fold), allow_pickle=True).item()

    
    train_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["validation"], slide_id=args.dataset, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)  
    test_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["test"], slide_id=args.dataset, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)    
    return train_loader, val_loader, test_loader, {"train": train_gene_name, "eval": eval_gene_name}, all_gene_num

class RandomSamplingDataset(torch.utils.data.Dataset):
    def __init__(self, args, adata, patch, patch_idx, slide_id, num_sampling=32, num_iterations=1000):
        """
        data: 元のデータ（リストやテンソル）
        num_samples: 各サンプリングで取得するデータの数（デフォルトは128）
        num_iterations: サンプリングを行う回数（デフォルトは100）
        """
        np.random.seed(args.seed)
        self.adata = adata
        self.patch = patch
        
        self.transform = image_transform(224, is_train=True)
        
        self.patch_idx = patch_idx
        self.len = len(self.patch_idx)
        self.args = args
        self.slide_id = args.dataset
        
        self.num_sampling = num_sampling
        self.num_iterations = num_iterations

    def __len__(self):
        return self.num_iterations  # 100回サンプリングを行う

    def __getitem__(self, idx):
        """
        idx は 0 から (num_iterations - 1) までの値
        その都度、ランダムに num_samples 個のデータを取得
        """
        sampling_indices = random.sample(range(len(self.patch_idx)), self.num_sampling)
        
        patch_adatas = self.adata[sampling_indices]
        # slide_id = patch_adata.obs["batch"].values[0]
        
        patchs, sts = [], []
        for patch_adata in patch_adatas:
            patch_idx = int(patch_adata.obs["patch_index"].values[0])
            
            patch = self.patch[patch_idx]

            if self.args.img_or_token_feat_or_feat == "img":
                patch = self.transform(patch)
            
            if self.args.dataset == "TENX152" or self.args.dataset == "MISC131" or self.args.dataset == "MISC132" or self.args.dataset == "MISC133" or self.args.dataset == "TENX61" or self.args.dataset == "TENX62" or self.args.dataset == "TENX65" or self.args.dataset == "TENX89":
                st = patch_adata.X.toarray()
            else:
                st = patch_adata.X
            st = torch.tensor((np.array(st)[0]))
            
            patchs.append(patch), sts.append(st)

        patchs, sts = torch.stack(patchs), torch.stack(sts)
        return {"patch": patchs, "st": sts, "slide_ids": self.slide_id, "patch_wise_unmask_gene": [], "patch_idx": patch_idx}

class RandomSamplingDataset_on_train(torch.utils.data.Dataset):
    def __init__(self, args, adata, patch, patch_idx, slide_id, num_sampling=32, num_iterations=1000):
        """
        data: 元のデータ（リストやテンソル）
        num_samples: 各サンプリングで取得するデータの数（デフォルトは128）
        num_iterations: サンプリングを行う回数（デフォルトは100）
        """
        np.random.seed(args.seed)
        self.adata = adata
        self.patch = patch
        
        self.transform = image_transform(224, is_train=True)
        
        self.patch_idx = patch_idx
        self.len = len(self.patch_idx)
        self.args = args
        self.slide_id = args.dataset
        
        self.num_sampling = num_sampling
        self.num_iterations = num_iterations
        
        self.batch_size = args.batch_size

    def __len__(self):
        return self.num_iterations * self.batch_size  # 100回サンプリングを行う

    def __getitem__(self, idx):
        """
        idx は 0 から (num_iterations - 1) までの値
        その都度、ランダムに num_samples 個のデータを取得
        """
        sampling_indices = random.sample(range(len(self.patch_idx)), 1)
        
        patch_adata = self.adata[sampling_indices]
        # slide_id = patch_adata.obs["batch"].values[0]
        
        patch_idx = int(patch_adata.obs["patch_index"].values[0])
        patch = self.patch[patch_idx]
        
        patch = self.transform(patch)
        st = patch_adata.X.toarray()

        st = torch.tensor((np.array(st)[0]))

        patch, st = torch.tensor(patch), torch.tensor(st)
        return {"patch": patch, "st": st, "slide_ids": self.slide_id, "patch_wise_unmask_gene": [], "patch_idx": patch_idx}


def hest_1slide_meta_load(args):  # LIMUC
    ######### load data #######
    if args.train_highly_variable_top_genes!=None:
        preprocessed_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_%s%d_matching_patch_adata.h5ad' % (args.input_path, args.dataset, args.gene_select_type, args.train_highly_variable_top_genes))
    elif args.train_highly_variable_top_genes==None:
        preprocessed_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_topall_matching_patch_adata.h5ad' % (args.input_path, args.dataset))
    
    eval_gene_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_top%d_matching_patch_adata.h5ad' % (args.input_path, args.dataset, args.eval_highly_variable_top_genes))
    train_gene_name = np.array(preprocessed_adata.var.index)
    eval_gene_name = np.array(eval_gene_adata.var.index)
    print(train_gene_name)
    
    all_gene_num = train_gene_name.shape[0]

    with h5py.File("%s/data/org/hest_data/patches/%s.h5" % (args.input_path, args.dataset), "r") as h5file:
        patch = h5file["img"][:]
            
    patch_idx_dict = np.load("%s/data/preprocessed_data/hest_1slide/%s_cv/fold=%d.npy" % (args.input_path, args.dataset, args.fold), allow_pickle=True).item()
    
    train_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    
    val_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["validation"], slide_id=args.dataset, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)  
    test_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["test"], slide_id=args.dataset, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)    
 
    mera_train_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=True)
    mera_train_loader = torch.utils.data.DataLoader(mera_train_dataset, shuffle=True, batch_size=args.sampling_num, num_workers=args.num_workers)
    mera_val_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["validation"], slide_id=args.dataset, is_train=True)
    mera_val_loader = torch.utils.data.DataLoader(mera_val_dataset, batch_size=args.sampling_num, shuffle=True,  num_workers=args.num_workers)     
    
    aux_gene_name =  train_gene_name[np.where(~np.isin(train_gene_name, eval_gene_name))[0]]
    
    return {"train": train_loader, "meta_train_sampler": mera_train_loader, "meta_val_sampler": mera_val_loader}, val_loader, test_loader, {"train": train_gene_name, "eval": eval_gene_name, "aux_gene": [], "train_toy_gene": [], "aux_gene": aux_gene_name}, all_gene_num
    

def hest_1slide_2stage_meta_load(args):  # LIMUC
    ######### load data #######
    if args.train_highly_variable_top_genes!=None:
        preprocessed_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_%s%d_matching_patch_adata.h5ad' % (args.input_path, args.dataset, args.gene_select_type, args.train_highly_variable_top_genes))
    elif args.train_highly_variable_top_genes==None:
        preprocessed_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_topall_matching_patch_adata.h5ad' % (args.input_path, args.dataset))
        sc.pp.highly_variable_genes(preprocessed_adata, n_top_genes=len(preprocessed_adata.var.index))  
    
    eval_gene_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_top%d_matching_patch_adata.h5ad' % (args.input_path, args.dataset, args.eval_highly_variable_top_genes))
    train_gene_name = np.array(preprocessed_adata.var.index)
    eval_gene_name = np.array(eval_gene_adata.var.index)
    
    hvg_scores = preprocessed_adata.var['dispersions_norm']
    hvg_scores = np.array(hvg_scores)
    
    print(train_gene_name)
    hvg = preprocessed_adata.var[preprocessed_adata.var['highly_variable']]
    hvg_sorted = hvg.sort_values(by='dispersions_norm', ascending=False)
    scores = hvg_sorted['dispersions_norm'].to_numpy()
    sorted_hvg_train_gene_names = hvg_sorted.index.to_numpy()
    
    # make_task_variance_line_graph(args, sorted_hvg_train_gene_names, scores, "%s/img/%s/hvg_order2.png" % (args.output_path, args.dataset))    
    
    all_gene_num = train_gene_name.shape[0]
    with h5py.File("%s/data/org/hest_data/patches/%s.h5" % (args.input_path, args.dataset), "r") as h5file:
        patch = h5file["img"][:]
    patch_idx_dict = np.load("%s/data/preprocessed_data/hest_1slide/%s_cv/fold=%d.npy" % (args.input_path, args.dataset, args.fold), allow_pickle=True).item()
    
    if args.best_highlyvariable_geneNum!=0:
        best_highlyvariable_preprocessed_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_%s%d_matching_patch_adata.h5ad' % (args.input_path, args.dataset, args.gene_select_type, args.best_highlyvariable_geneNum))
    elif args.best_highlyvariable_geneNum==0:
        best_highlyvariable_preprocessed_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_topall_matching_patch_adata.h5ad' % (args.input_path, args.dataset))
        
    best_highlyvariable_gene_name = np.array(best_highlyvariable_preprocessed_adata.var.index)
    
    train_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    train_val_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=False)
    train_val_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    val_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["validation"], slide_id=args.dataset, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)  
    test_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["test"], slide_id=args.dataset, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)    
 
    mera_train_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=True)
    mera_train_loader = torch.utils.data.DataLoader(mera_train_dataset, shuffle=True, batch_size=args.sampling_num, num_workers=args.num_workers)
    mera_val_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["validation"], slide_id=args.dataset, is_train=True)
    mera_val_loader = torch.utils.data.DataLoader(mera_val_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)     
    
    aux_gene_name =  train_gene_name[np.where(~np.isin(train_gene_name, eval_gene_name))[0]]
    best_highlyvariable_gene_name = aux_gene_name[np.where(np.isin(aux_gene_name, best_highlyvariable_gene_name))[0]]
    lowvariable_gene_name =  aux_gene_name[np.where(~np.isin(aux_gene_name, best_highlyvariable_gene_name))[0]]
    return {"train": train_loader, "meta_train_sampler": mera_train_loader, "meta_val_sampler": mera_val_loader, "train_val_loader":train_val_loader}, val_loader, test_loader, {"train": train_gene_name, "train_hvg_Sort":sorted_hvg_train_gene_names, "hvg_score":hvg_scores,  "eval": eval_gene_name, "train_highly_gene": best_highlyvariable_gene_name, "train_low_gene": lowvariable_gene_name}, all_gene_num
    

def hest_1slide_3stepLearning_meta_load(args):  # LIMUC
    ######### load data #######
    if args.train_highly_variable_top_genes!=None:
        preprocessed_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_%s%d_matching_patch_adata.h5ad' % (args.input_path, args.dataset, args.gene_select_type, args.train_highly_variable_top_genes))
    elif args.train_highly_variable_top_genes==None:
        preprocessed_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_topall_matching_patch_adata.h5ad' % (args.input_path, args.dataset))
    
    eval_gene_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_top%d_matching_patch_adata.h5ad' % (args.input_path, args.dataset, args.eval_highly_variable_top_genes))
    train_gene_name = np.array(preprocessed_adata.var.index)
    eval_gene_name = np.array(eval_gene_adata.var.index)
    
    print(train_gene_name)
    hvg = preprocessed_adata.var[preprocessed_adata.var['highly_variable']]
    hvg_sorted = hvg.sort_values(by='dispersions_norm', ascending=False)
    scores = hvg_sorted['dispersions_norm'].to_numpy()
    sorted_hvg_train_gene_names = hvg_sorted.index.to_numpy()
    
    # make_task_variance_line_graph(args, sorted_hvg_train_gene_names, scores, "%s/img/%s/hvg_order.png" % (args.output_path, args.dataset))
    
    all_gene_num = train_gene_name.shape[0]
    with h5py.File("%s/data/org/hest_data/patches/%s.h5" % (args.input_path, args.dataset), "r") as h5file:
        patch = h5file["img"][:]
    patch_idx_dict = np.load("%s/data/preprocessed_data/hest_1slide/%s_cv/fold=%d.npy" % (args.input_path, args.dataset, args.fold), allow_pickle=True).item()
    
    best_highlyvariable_preprocessed_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_%s%d_matching_patch_adata.h5ad' % (args.input_path, args.dataset, args.gene_select_type, args.best_highlyvariable_geneNum))
    best_gene_name = np.array(best_highlyvariable_preprocessed_adata.var.index)
    
    train_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    train_val_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=False)
    train_val_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    val_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["validation"], slide_id=args.dataset, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)  
    test_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["test"], slide_id=args.dataset, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)    
 
    mera_train_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=True)
    mera_train_loader = torch.utils.data.DataLoader(mera_train_dataset, shuffle=True, batch_size=args.sampling_num, num_workers=args.num_workers)
    mera_val_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["validation"], slide_id=args.dataset, is_train=True)
    mera_val_loader = torch.utils.data.DataLoader(mera_val_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)     
    
    aux_gene_name =  train_gene_name[np.where(~np.isin(train_gene_name, eval_gene_name))[0]]
    best_highlyvariable_gene_name = aux_gene_name[np.where(np.isin(aux_gene_name, best_gene_name))[0]]
    lowvariable_gene_name =  aux_gene_name[np.where(~np.isin(aux_gene_name, best_highlyvariable_gene_name))[0]]
    return {"train": train_loader, "meta_train_sampler": mera_train_loader, "meta_val_sampler": mera_val_loader, "train_val_loader":train_val_loader}, val_loader, test_loader, {"bestmodel_gene_name": best_gene_name, "all_gene_name": train_gene_name, "train_hvg_Sort":sorted_hvg_train_gene_names, "eval": eval_gene_name, "train_highly_gene": best_highlyvariable_gene_name, "train_low_gene": lowvariable_gene_name}, all_gene_num
    


def hest_1slide_patch_meta_load(args):  # LIMUC
    ######### load data #######
    if args.train_highly_variable_top_genes!=None:
        preprocessed_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_%s%d_matching_patch_adata.h5ad' % (args.input_path, args.dataset, args.gene_select_type, args.train_highly_variable_top_genes))
    elif args.train_highly_variable_top_genes==None:
        preprocessed_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_topall_matching_patch_adata.h5ad' % (args.input_path, args.dataset))
    
    eval_gene_adata = sc.read('%s/data/preprocessed_data/hest_1slide/%s_top%d_matching_patch_adata.h5ad' % (args.input_path, args.dataset, args.eval_highly_variable_top_genes))
    train_gene_name = np.array(preprocessed_adata.var.index)
    eval_gene_name = np.array(eval_gene_adata.var.index)
    print(train_gene_name)
    
    all_gene_num = train_gene_name.shape[0]

    with h5py.File("%s/data/org/hest_data/patches/%s.h5" % (args.input_path, args.dataset), "r") as h5file:
        patch = h5file["img"][:]
            
    patch_idx_dict = np.load("%s/data/preprocessed_data/hest_1slide/%s_cv/fold=%d.npy" % (args.input_path, args.dataset, args.fold), allow_pickle=True).item()
    
    train_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    
    val_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["validation"], slide_id=args.dataset, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)  
    test_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["test"], slide_id=args.dataset, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)    
 
    mera_train_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=True)
    mera_train_loader = torch.utils.data.DataLoader(mera_train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    mera_val_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["validation"], slide_id=args.dataset, is_train=True)
    mera_val_loader = torch.utils.data.DataLoader(mera_val_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)     
    
    return {"train": train_loader, "meta_train_sampler": mera_train_loader, "meta_val_sampler": mera_val_loader}, val_loader, test_loader, {"train": train_gene_name, "eval": eval_gene_name, "train_highly_gene": [], "train_toy_gene": []}, all_gene_num

class RandomSampling_on_validation:
    def __init__(self, args, adata, patch, patch_idx, slide_id, num_sampling=32, batchsize=8):
        """
        data: 元のデータ（リストやテンソル）
        num_samples: 各サンプリングで取得するデータの数（デフォルトは128）
        num_iterations: サンプリングを行う回数（デフォルトは100）
        """
        np.random.seed(args.seed)
        self.adata = adata
        self.patch = patch
        
        self.transform = image_transform(224, is_train=True)
        
        self.patch_idx = patch_idx
        self.len = len(self.patch_idx)
        self.args = args
        self.slide_id = args.dataset
        
        self.num_sampling = num_sampling
        self.batchsize = batchsize

    def __call__(self):
        """
        idx は 0 から (num_iterations - 1) までの値
        その都度、ランダムに num_samples 個のデータを取得
        """
        # slide_id = patch_adata.obs["batch"].values[0]
        patchs_list, sts_list = [], []
        for _ in range(self.batchsize):
            sampling_indices = random.sample(range(len(self.patch_idx)), self.num_sampling)
            patch_adatas = self.adata[sampling_indices]
            patchs, sts = [], []
            for patch_adata in patch_adatas:
                patch_idx = int(patch_adata.obs["patch_index"].values[0])
                
                patch = self.patch[patch_idx]

                if self.args.img_or_token_feat_or_feat == "img":
                    patch = self.transform(patch)
                
                if self.args.dataset == "TENX152" or self.args.dataset == "MISC131" or self.args.dataset == "MISC132" or self.args.dataset == "MISC133" or self.args.dataset == "TENX61" or self.args.dataset == "TENX62" or self.args.dataset == "TENX65" or self.args.dataset == "TENX89":
                    st = patch_adata.X.toarray()
                else:
                    st = patch_adata.X
                st = torch.tensor((np.array(st)[0]))
                
                patchs.append(patch), sts.append(st)

            patchs, sts = torch.stack(patchs), torch.stack(sts)
            
            patchs_list.append(patchs)
            sts_list.append(sts)
            
        patchs_list, sts_list = torch.stack(patchs_list), torch.stack(sts_list)
        return {"patch": patchs_list, "st": sts_list, "slide_ids": self.slide_id}

def hest_1slide_pathway_load(args):  # LIMUC
    ######### load data #######
    preprocessed_adata = sc.read('/media/user/HD-QHAU3/ST_estimation_data/data/preprocessed_data/hest_1slide/%s_topall_matching_patch_adata.h5ad' % (args.dataset))
    
    train_gene_name = np.array(preprocessed_adata.var.index)

    with h5py.File("/media/user/HD-QHAU3/ST_estimation_data/data/org/hest_data/patches/%s.h5" % args.dataset, "r") as h5file:
        patch = h5file["img"][:]

    patch_idx_dict = np.load("/media/user/HD-QHAU3/ST_estimation_data/data/preprocessed_data/hest_1slide/%s_cv/fold=%d.npy" % (args.dataset, args.fold), allow_pickle=True).item()
    
    train_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["train"], slide_id=args.dataset, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["validation"], slide_id=args.dataset, is_train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)  
    test_dataset = Hest_1slide_Dataset(args=args, adata=preprocessed_adata, patch=patch, patch_idx=patch_idx_dict["test"], slide_id=args.dataset, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers)    
    return train_loader, val_loader, test_loader, {"all": train_gene_name, "train": args.train_highly_variable_top_genes_name, "eval": args.eval_highly_variable_top_genes_name}, 0

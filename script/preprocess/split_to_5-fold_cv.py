import numpy as np
import random

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
import h5py
import scanpy as sc

slide_id_list = np.array(["TENX65", "TENX89", "TENX152"])

num_fold = 5
num_slide = len(slide_id_list)
for slide_id in slide_id_list:
    preprocessed_adata = sc.read("./data/preprocessed_data/hest_1slide/%s_topall_matching_patch_adata.h5ad" % slide_id)
    if not os.path.exists("./data/preprocessed_data/hest_1slide/%s_cv" % (slide_id)):
        os.mkdir("./data/preprocessed_data/hest_1slide/%s_cv" % (slide_id))

    patch_data = preprocessed_adata    
    patxh_idx = np.arange(len(patch_data))
    np.random.shuffle(patxh_idx)
    split_len = int(len(patxh_idx)/num_fold)
    
    with h5py.File("./data/org/hest_data/patches/%s.h5" % slide_id, "r") as h5file:
        tmp = h5file["img"][:]
    
        fold_dict = {'0': patxh_idx[0:split_len], 
                     '1': patxh_idx[split_len:(split_len*2)], 
                     '2': patxh_idx[(split_len*2):(split_len*3)], 
                     '3': patxh_idx[(split_len*3):(split_len*4)], 
                     '4': patxh_idx[(split_len*4):]}
        
    for fold in range(num_fold):
        # Create the splits
        test_idx = fold_dict['%d'%(fold%5)]
        val_idx = fold_dict['%d'%((fold+1)%5)]
        train_idx = np.concatenate((fold_dict['%d'%((2+fold)%5)], fold_dict['%d'%((3+fold)%5)], fold_dict['%d'%((4+fold)%5)]))

        splits = {
            "train": train_idx,
            "validation": val_idx,
            "test": test_idx
        }       
         
        np.save("./data/preprocessed_data/hest_1slide//%s_cv/fold=%d.npy" % (slide_id, fold), splits)

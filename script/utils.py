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
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
import glob
from torchmetrics.regression import ( PearsonCorrCoef, 
                                    ConcordanceCorrCoef, 
                                    MeanSquaredError,
                                    MeanAbsoluteError,
                                    ExplainedVariance, 
                                    SpearmanCorrCoef)  

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm

def make_folder(args):
    os.makedirs(args.output_path, exist_ok=True)
    if not os.path.exists(args.output_path + "/acc_graph"):
        os.mkdir(args.output_path + "/acc_graph")
    if not os.path.exists(args.output_path + "/cm"):
        os.mkdir(args.output_path + "/cm")
    if not os.path.exists(args.output_path + "/log_dict"):
        os.mkdir(args.output_path + "/log_dict")
    if not os.path.exists(args.output_path + "/loss_graph"):
        os.mkdir(args.output_path + "/loss_graph")
    if not os.path.exists(args.output_path + "/model"):
        os.mkdir(args.output_path + "/model")
    if not os.path.exists(args.output_path + "/k_value"):
        os.mkdir(args.output_path + "/k_value")   
    return

def culc_metric(gt_st, pred_st, slide_ids, gene_name_dict):
    all_gene_name, target_gene_name = gene_name_dict["all"], gene_name_dict["target"]
    target_gene_idx = np.where(np.isin(all_gene_name, target_gene_name))[0]
    
    mse_all, pearsonr_all, spearman_all = 0, 0, 0
    unique_slide_ids = np.unique(slide_ids)
    for slide_id in unique_slide_ids:
        # MSE
        mse = ((gt_st[slide_ids==slide_id][:, target_gene_idx]-pred_st[slide_ids==slide_id][:, target_gene_idx])**2).mean()
        mse_all += mse
        # Pearsonr
        metric_pearson = PearsonCorrCoef(num_outputs=gt_st[:, target_gene_idx].shape[-1])
        pearsonr = metric_pearson(torch.tensor(pred_st[slide_ids==slide_id][:, target_gene_idx]), torch.tensor(gt_st[slide_ids==slide_id][:, target_gene_idx]))
        pearsonr = pearsonr.nanmean()
        pearsonr_all += pearsonr   
        # Spearman
        metric_spearman = SpearmanCorrCoef(num_outputs=gt_st[:, target_gene_idx].shape[-1])
        spearman = metric_spearman(torch.tensor(pred_st[slide_ids==slide_id][:, target_gene_idx]), torch.tensor(gt_st[slide_ids==slide_id][:, target_gene_idx]))
        spearman = spearman.nanmean()
        spearman_all += spearman
    
    mse_all /= len(unique_slide_ids)
    pearsonr_all /= len(unique_slide_ids)
    spearman_all /= len(unique_slide_ids)
    return {"mse": mse_all, "pearsonr": pearsonr_all, "spearman": spearman_all}

def Make_loss_graph(args, train_loss, valid_loss, path):
    #loss graph save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_loss, label = 'train')
    ax.plot(valid_loss, label = 'valid')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("Losses")
    plt.legend()
    fig.savefig(path)
    plt.close() 
    return

def Make_acc_graph(args, train_acc, val_acc, test_acc, path):
    #Bag level accuracy save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_acc, label = 'train')
    ax.plot(val_acc, label = 'valid')
    ax.plot(test_acc, label = 'test')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("accuracy")
    plt.legend()
    fig.savefig(path)
    plt.close()
    return

def Make_CutOff_k_graph(args, thresh_list, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresh_list, label = 'thresh value')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("thresh value")
    plt.legend()
    fig.savefig(path)
    plt.close()
    return
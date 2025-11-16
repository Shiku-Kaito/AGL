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
    # if not os.path.exists(args.output_path):
    #     os.mkdir(args.output_path)
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
    if not os.path.exists(args.output_path + "/test_metrics"):
        os.mkdir(args.output_path + "/test_metrics")
    if not os.path.exists(args.output_path + "/unmask_gene_dict"):
        os.mkdir(args.output_path + "/unmask_gene_dict")   
    if not os.path.exists(args.output_path + "/task_wgt"):
        os.mkdir(args.output_path + "/task_wgt")   
    return

def save_confusion_matrix(cm, path, title=''):
    plt.figure(figsize=(10, 8), dpi=300)
    cm = cm / cm.sum(axis=-1, keepdims=1)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', annot_kws={"size": 40})
    plt.xlabel('pred', fontsize=24)  # x軸ラベルの文字サイズを指定
    plt.ylabel('GT', fontsize=24)  # y軸ラベルの文字サイズを指定
    # sns.heatmap(cm, annot=True, cmap='Blues_r', fmt='.2f', annot_kws={"size": 36})
    # plt.xlabel('pred', fontsize=24)  # x軸ラベルの文字サイズを指定
    # plt.ylabel('GT', fontsize=24)  # y軸ラベルの文字サイズを指定
    plt.title(title)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def cal_OP_PC_mIoU(cm):
    num_classes = cm.shape[0]

    TP_c = np.zeros(num_classes)
    for i in range(num_classes):
        TP_c[i] = cm[i][i]

    FP_c = np.zeros(num_classes)
    for i in range(num_classes):
        FP_c[i] = cm[i, :].sum()-cm[i][i]

    FN_c = np.zeros(num_classes)
    for i in range(num_classes):
        FN_c[i] = cm[:, i].sum()-cm[i][i]

    OP = TP_c.sum() / (TP_c+FP_c).sum()
    PC = (TP_c/(TP_c+FP_c)).mean()
    mIoU = (TP_c/(TP_c+FP_c+FN_c)).mean()

    return OP, PC, mIoU


def cal_mIoU(cm):
    num_classes = cm.shape[0]

    TP_c = np.zeros(num_classes)
    for i in range(num_classes):
        TP_c[i] = cm[i][i]

    FP_c = np.zeros(num_classes)
    for i in range(num_classes):
        FP_c[i] = cm[i, :].sum()-cm[i][i]

    FN_c = np.zeros(num_classes)
    for i in range(num_classes):
        FN_c[i] = cm[:, i].sum()-cm[i][i]

    mIoU = (TP_c/(TP_c+FP_c+FN_c)).mean()

    return mIoU



def culc_metric(gt_st, pred_st, slide_ids, gene_name_dict):
    train_gene_name, eval_gene_name = gene_name_dict["train"], gene_name_dict["eval"]
    eval_gene_idx = np.where(np.isin(train_gene_name, eval_gene_name))[0]
    # MSE
    mse_all, pearsonr_all, spearman_all = 0, 0, 0
    pearsonr_eval_all_gene, pearsonr_eval_all_gene_name  = [], []
    unique_slide_ids = np.unique(slide_ids)
    for slide_id in unique_slide_ids:
        mse = ((gt_st[slide_ids==slide_id][:, eval_gene_idx]-pred_st[slide_ids==slide_id][:, eval_gene_idx])**2).mean()

        # pearsonr
        metric_pearson = PearsonCorrCoef(num_outputs=gt_st[:, eval_gene_idx].shape[-1])
        pearsonr = metric_pearson(torch.tensor(pred_st[slide_ids==slide_id][:, eval_gene_idx]), torch.tensor(gt_st[slide_ids==slide_id][:, eval_gene_idx]))
        pearsonr_eval_all_gene.append(pearsonr)
        pearsonr_eval_all_gene_name.append(train_gene_name[eval_gene_idx])
        pearsonr = pearsonr.nanmean()
        
        # spearman
        metric_spearman = SpearmanCorrCoef(num_outputs=gt_st[:, eval_gene_idx].shape[-1])
        spearman = metric_spearman(torch.tensor(pred_st[slide_ids==slide_id][:, eval_gene_idx]), torch.tensor(gt_st[slide_ids==slide_id][:, eval_gene_idx]))
        spearman = spearman.nanmean()

        mse_all += mse
        pearsonr_all += pearsonr
        spearman_all += spearman
    
    mse_all /= len(unique_slide_ids)
    pearsonr_all /= len(unique_slide_ids)
    spearman_all /= len(unique_slide_ids)
    return {"mse": mse_all, "pearsonr": pearsonr_all, "spearman": spearman_all, "pearsonr_eval_all_gene": pearsonr_eval_all_gene[0], "pearsonr_eval_all_gene_name": pearsonr_eval_all_gene_name[0]}


def culc_metric_prim_input(gt_st, pred_st, slide_ids):
    # MSE
    mse_all, pearsonr_all, spearman_all = 0, 0, 0
    unique_slide_ids = np.unique(slide_ids)
    for slide_id in unique_slide_ids:
        mse = ((gt_st[slide_ids==slide_id]-pred_st[slide_ids==slide_id])**2).mean()

        # pearsonr
        metric_pearson = PearsonCorrCoef(num_outputs=gt_st.shape[-1])
        pearsonr = metric_pearson(torch.tensor(pred_st[slide_ids==slide_id]), torch.tensor(gt_st[slide_ids==slide_id]))
        pearsonr = pearsonr.nanmean()
        
        # spearman
        metric_spearman = SpearmanCorrCoef(num_outputs=gt_st.shape[-1])
        spearman = metric_spearman(torch.tensor(pred_st[slide_ids==slide_id]), torch.tensor(gt_st[slide_ids==slide_id]))
        spearman = spearman.nanmean()

        mse_all += mse
        pearsonr_all += pearsonr
        spearman_all += spearman
    
    mse_all /= len(unique_slide_ids)
    pearsonr_all /= len(unique_slide_ids)
    spearman_all /= len(unique_slide_ids)
    return {"mse": mse_all, "pearsonr": pearsonr_all, "spearman": spearman_all}


def make_loss_graph(args, keep_train_loss, keep_valid_loss, path):
    #loss graph save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(keep_train_loss, label = 'train')
    ax.plot(keep_valid_loss, label = 'valid')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("Losses")
    plt.legend()
    fig.savefig(path)
    plt.close() 
    return

def make_bag_acc_graph(args, train_bag_acc, val_bag_acc, test_bag_acc, path):
    #Bag level accuracy save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_bag_acc, label = 'train bag acc')
    ax.plot(val_bag_acc, label = 'valid bag acc')
    ax.plot(test_bag_acc, label = 'test bag acc')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("accuracy")
    plt.legend()
    fig.savefig(path)
    plt.close()
    return

def make_ins_acc_graph(args, train_ins_acc, val_ins_acc, test_ins_acc, path):
    #instance level accuracy save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_ins_acc, label = 'train instance acc')
    ax.plot(val_ins_acc, label = 'valid instans acc')
    ax.plot(test_ins_acc, label = 'test ins acc')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("accuracy")
    plt.legend()
    fig.savefig(path)
    plt.close()
    return

def make_task_wgt_graph(args, task_wgt, path):
    plt.figure(figsize=(8, 5))
    x = np.arange(task_wgt.shape[0])
    for idx in range(len(task_wgt[0])):
        y = task_wgt[:, idx]
        plt.plot(x, y, marker='o', alpha=0.7)  # 各リストをプロット（マーカー付き）

    # Y軸の範囲を指定
    plt.ylim(0, 1.1)

    # 軸ラベルと凡例
    plt.xlabel("Epoch numbers")
    plt.ylabel("task weight")
    plt.title("task weight Plots")
    # plt.legend([f"Dataset {i+1}" for i in range(len(y_values))])
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    return

def make_task_row_wgt_graph(args, task_wgt, path):
    plt.figure(figsize=(8, 5))
    x = np.arange(task_wgt.shape[0])
    for idx in range(len(task_wgt[0])):
        y = task_wgt[:, idx]
        plt.plot(x, y, marker='o', alpha=0.7)  # 各リストをプロット（マーカー付き）

    # Y軸の範囲を指定
    plt.ylim(task_wgt.min(), task_wgt.max()+0.1)

    # 軸ラベルと凡例
    plt.xlabel("Epoch numbers")
    plt.ylabel("task weight")
    plt.title("task weight Plots")
    # plt.legend([f"Dataset {i+1}" for i in range(len(y_values))])
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    return

def make_toy_task_wgt_graph(args, task_wgt, train_highly_gene_idx, train_toy_gene_idx, path):
    plt.figure(figsize=(8, 5))
    x = np.arange(task_wgt.shape[0])
    color_list =np.full((task_wgt.shape[1],), "o")
    color_list[train_highly_gene_idx] = 'o'
    color_list[train_toy_gene_idx] = 'b'
    
    for idx in range(task_wgt.shape[1]):
        if color_list[idx]=="o":
            color = "orange"
        else:
            color = "blue"
            
        y = task_wgt[:, idx]
        plt.plot(x, y, marker='o', color=color, alpha=0.7)  # 各リストをプロット（マーカー付き）

    # Y軸の範囲を指定
    plt.ylim(0, 1.1)
    # plt.ylim(0, task_wgt.max())
    

    # 軸ラベルと凡例
    plt.xlabel("Epoch numbers")
    plt.ylabel("task weight")
    plt.title("task weight Plots")
    # plt.legend([f"Dataset {i+1}" for i in range(len(y_values))])
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    return


def make_task_wgt_line_graph(args, task_wgt, gene_name, sorted_gene_name, path):
    # 1. gene_namesのインデックスをdictに変換
    name_to_index = {name: i for i, name in enumerate(gene_name)}
    # 2. sorted_gene_namesに対応するインデックスを一括取得
    indices = np.vectorize(name_to_index.get)(sorted_gene_name)
    # 3. 対応する分散を取得
    sorted_task_wgt = task_wgt[indices]

    plt.figure(figsize=(10, 5))  # 必要に応じてサイズ調整
    # 点を小さくする（s=10など、好みに応じて変更可能）
    plt.scatter(sorted_gene_name, sorted_task_wgt, s=10)
    plt.xlabel('gene name (Descending order of variance)', fontsize=10)
    plt.ylabel('weight value', fontsize=10)

    # x軸ラベルを斜め＋小さいフォントで表示
    plt.xticks(rotation=45, fontsize=8)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return

def make_task_variance_line_graph(args, variance, gene_name, path):
    plt.figure(figsize=(10, 5))  # 必要に応じてサイズ調整

    # 点を小さくする（s=10など、好みに応じて変更可能）
    plt.scatter(variance, gene_name, s=10)

    plt.xlabel('gene name (Descending order of variance)', fontsize=10)
    plt.ylabel('Variance value', fontsize=10)

    # x軸ラベルを斜め＋小さいフォントで表示
    plt.xticks(rotation=45, fontsize=8)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return

# def make_task_variance_line_graph(args, variance, gene_name, path):
#     plt.scatter(variance, gene_name)
#     plt.xlabel('gene name (Descending order of variance)')
#     plt.ylabel('Variance value')
#     plt.xticks(rotation=45)
#     plt.savefig(path)
#     plt.close()
#     return


def make_thresh_graph(args, thresh_list, path):
    #Bag level accuracy save
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresh_list, label = 'thresh value')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("thresh value")
    plt.legend()
    fig.savefig(path)
    plt.close()
    return
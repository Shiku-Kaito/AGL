
import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from utils import *
from higher_task_weight_script.weight_estimater_model import MAXL_weight_estimater as higher_gene_mask_estimater
# from amal_script.weight_estimater_model import MAXL_weight_estimater as amal_gene_mask_estimater


def eval_net(args, model, test_loader, metric_func, gene_name_dict):
    # eval_gene_idx = np.where(np.isin(gene_name_dict["train"], np.array(["GUCA2B"])))[0]
    # gene_name_dict["eval"] = gene_name_dict["train"][eval_gene_idx]
    prime_gene_name, aux_gene_name, all_gene_name =  gene_name_dict["eval"], gene_name_dict["train"][np.where(~np.isin(gene_name_dict["train"], gene_name_dict["eval"]))[0]], gene_name_dict["train"]
    prime_gene_name, aux_gene_name, all_gene_name =  gene_name_dict["eval"], gene_name_dict["train"][np.where(~np.isin(gene_name_dict["train"], gene_name_dict["eval"]))[0]], gene_name_dict["train"]
    eval_gene_idx, aux_gene_idx = np.where(np.isin(all_gene_name, prime_gene_name))[0], np.where(np.isin(all_gene_name, aux_gene_name))[0]
    train_highly_gene_idx = np.where(np.isin(all_gene_name, gene_name_dict["train_highly_gene"]))[0]
    
    train_highly_gene_idx = np.where(np.isin(all_gene_name, gene_name_dict["train_highly_gene"]))[0]
    train_low_gene_idx = np.where(np.isin(all_gene_name, gene_name_dict["train_low_gene"]))[0]
    
    weight_estimater = higher_gene_mask_estimater(args, train_highly_gene=gene_name_dict["train_highly_gene"], train_low_gene=gene_name_dict["train_low_gene"]).to(args.device)
    weight_estimater.load_state_dict(torch.load(("%s/model/fold=%d_seed=%d_epoch=20-best_weight_estimater.pkl") % (args.output_path, args.fold, args.seed) ,map_location=args.device))
    
    fix_seed(args.seed)
    result_dict = {}
    ################## test ###################
    s_time = time()
    model.eval()
    weight_estimater.eval()
    gt_st, pred_st, losses, slide_ids_list = [], [], [], []
    with torch.no_grad():
        for iteration, data in enumerate(tqdm(test_loader, leave=False)):
            patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
            patch, st = patch.to(args.device), st.to(args.device)

            y = model(patch)
            loss_weight = weight_estimater(y["y"], eval_gene_idx, train_highly_gene_idx, train_low_gene_idx)  # generate auxiliary labels                   

            gt_st.extend(st.cpu().detach().numpy()), pred_st.extend(y["y"].cpu().detach().numpy())
            slide_ids_list.extend(slide_ids)

    gt_st, pred_st, slide_ids_list = np.array(gt_st), np.array(pred_st), np.array(slide_ids_list)
    metrics = metric_func(gt_st, pred_st, slide_ids_list, gene_name_dict)
    
    task_wgt = loss_weight["w"].cpu().detach().numpy().reshape(1, -1)[0]
    # make_task_wgt_line_graph(args, task_wgt, all_gene_name, gene_name_dict["train_hvg_Sort"],  "%s/task_wgt/fold=%d_seed=%d_epoch=20_task_wgt_line_graph.png" % (args.output_path, args.fold, args.seed))

    result_dict["mse"], result_dict["pearsonr"], result_dict["spearman"] = metrics["mse"], metrics["pearsonr"], metrics["spearman"]
    result_dict["pearsonr_eval_all_gene"], result_dict["pearsonr_eval_all_gene_name"] =  np.array(metrics["pearsonr_eval_all_gene"]), np.array(metrics["pearsonr_eval_all_gene_name"])
    return result_dict
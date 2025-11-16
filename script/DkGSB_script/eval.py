
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

def eval_net(args, 
             model, 
             test_loader, 
             metric_func, 
             gene_name_dict):
    
    fix_seed(args.seed)
    result_dict = {}
    ################## test ###################
    s_time = time()
    model.eval()
    gt_st, pred_st, losses, slide_ids_list = [], [], [], []
    with torch.no_grad():
        for iteration, data in enumerate(tqdm(test_loader, leave=False)):
            patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
            patch, st = patch.to(args.device), st.to(args.device)

            y = model(patch)

            gt_st.extend(st.cpu().detach().numpy()), pred_st.extend(y["y"].cpu().detach().numpy())
            slide_ids_list.extend(slide_ids)

    gt_st, pred_st, slide_ids_list = np.array(gt_st), np.array(pred_st), np.array(slide_ids_list)
    metrics = metric_func(gt_st, pred_st, slide_ids_list, gene_name_dict)
    result_dict["mse"], result_dict["pearsonr"], result_dict["spearman"] = metrics["mse"], metrics["pearsonr"], metrics["spearman"]
    return result_dict
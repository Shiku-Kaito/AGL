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
from collections import OrderedDict
import copy
import torch.nn.utils.stateless as stateless
import torch.nn as nn
from itertools import cycle
import higher

def train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function, metric_func, gene_name_dict):
    st_estimater, CutOff_estimater = model["st_estimater"], model["CutOff_estimater"]
    st_estimater, CutOff_estimater = st_estimater.to(args.device), CutOff_estimater.to(args.device) 
    st_estimater_optimizer, CutOff_estimater_optimizer = optimizer["st_estimater"], optimizer["CutOff_estimater"]
    
    target_gene_name, aux_gene_name, all_gene_name =  gene_name_dict["target"], gene_name_dict["aux"], gene_name_dict["all"]
    target_gene_idx, aux_gene_idx = np.where(np.isin(all_gene_name, target_gene_name))[0], np.where(np.isin(all_gene_name, aux_gene_name))[0]
    
    fix_seed(args.seed)
    log_dict = {"train_mse":[], "train_pearsonr":[], "train_spearman":[], "train_mse_loss":[], 
                "val_mse":[], "val_pearsonr":[], "val_spearman":[], "val_mse_loss":[],
                "test_mse":[], "test_pearsonr":[], "test_spearman":[], "test_mse_loss":[], 
                "k":[]}
    
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("%s/log_dict/fold=%d_seed=%d_training_setting.log" %  (args.output_path, args.fold, args.seed))
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
    logging.info(args)

    train_data_loader, meta_train_loader, meta_val_loader = train_loader["train"], train_loader["meta_train_sampler"], train_loader["meta_val_sampler"]

    best_epoch = 0
    cnt = 0
    best_val_pearsonr = -100000000
    for epoch in range(args.num_epochs):
        if epoch!=0:            
        ############ Bi-level optimization ###################
            logging.info("####### Bi-level optimization step ######### ")
            st_estimater_copy = copy.deepcopy(st_estimater).to(args.device)

            st_estimater_copy.train()
            CutOff_estimater.train()
            gt_st, pred_st, losses, slide_ids_list = [], [], [], []
            for t_data, v_data in tqdm(zip(meta_train_loader, cycle(meta_val_loader)), total=len(meta_train_loader), leave=False):
                inner_opt = torch.optim.Adam(st_estimater_copy.parameters(), lr=args.alpha)          
                CutOff_estimater_optimizer.zero_grad()
            
                with higher.innerloop_ctx(st_estimater_copy, inner_opt, copy_initial_weights=False, track_higher_grads=True) as (fnet, diffopt):
                    t_patch, t_st, t_slide_ids = t_data["patch"], t_data["st"], t_data["slide_ids"]
                    t_patch, t_st = t_patch.to(args.device), t_st.to(args.device)
                    for k in range(args.H):
                        y = fnet(t_patch)
                        loss_weight = CutOff_estimater(y["y"], target_gene_idx, aux_gene_idx)  # generate auxiliary labels                   
                        
                        loss = loss_function(y["y"], t_st, np.array(t_slide_ids))
                        
                        all_train_loss = (loss * loss_weight["w"].unsqueeze(0)).sum(1) /sum(loss_weight["w"].cpu().detach().numpy())
                        all_train_loss = all_train_loss.sum()/64
                        diffopt.step(all_train_loss)
                    
                    #######  Update weight model on train data
                    v_patch, v_st, v_slide_ids = v_data["patch"], v_data["st"], v_data["slide_ids"]
                    v_patch, v_st = v_patch.to(args.device), v_st.to(args.device)
                    y = fnet(v_patch)
                    
                    loss_prime = loss_function(y["y"][:, target_gene_idx], v_st[:, target_gene_idx], np.array(v_slide_ids))
                    val_loss = sum(loss_prime)/len(target_gene_name)
                    val_loss.backward()
                    logging.info("== outer loop: loss: %.7f" % val_loss.item())
                    
                CutOff_estimater_optimizer.step()
                gt_st.extend(v_st.cpu().detach().numpy()), pred_st.extend(y["y"].cpu().detach().numpy())
                slide_ids_list.extend(v_slide_ids)

            log_dict["k"].append(loss_weight["k"].cpu().detach().numpy().item())

        ############ train ###################
        s_time = time()
        st_estimater.train()
        CutOff_estimater.eval()
        gt_st, pred_st, losses, slide_ids_list = [], [], [], []
        for iteration, data in enumerate(tqdm(train_data_loader, leave=False)): #enumerate(tqdm(train_loader, leave=False)):
            patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
            patch, st = patch.to(args.device), st.to(args.device)

            y = st_estimater(patch)
            loss_weight = CutOff_estimater(y["y"], target_gene_idx, aux_gene_idx)  # generate auxiliary labels

            st_estimater_optimizer.zero_grad()
            CutOff_estimater_optimizer.zero_grad()
         
            loss = loss_function(y["y"], st, np.array(slide_ids))
            train_loss = sum(loss_weight["w"] * loss) /sum(loss_weight["w"].cpu().detach().numpy())
       
            train_loss.backward()
            st_estimater_optimizer.step()

            gt_st.extend(st.cpu().detach().numpy()), pred_st.extend(y["y"].cpu().detach().numpy())
            losses.append(train_loss.item())
            slide_ids_list.extend(slide_ids)
            
        gt_st, pred_st, slide_ids_list = np.array(gt_st), np.array(pred_st), np.array(slide_ids_list)
        metrics = metric_func(gt_st, pred_st, slide_ids_list, gene_name_dict)

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] train loss: %.4f, @ mse: %.4f, pearsonr: %.4f, spearman: %.4f' %
                    (epoch+1, args.num_epochs, e_time-s_time, np.array(losses).mean(), metrics["mse"], metrics["pearsonr"], metrics["spearman"]))
        
        log_dict["train_mse"].append(metrics["mse"]), log_dict["train_pearsonr"].append(metrics["pearsonr"]), log_dict["train_spearman"].append(metrics["spearman"])
        log_dict["train_mse_loss"].append(np.array(losses).mean())
        

        ################# validation ####################
        s_time = time()
        st_estimater.eval()
        CutOff_estimater.eval()
        gt_st, pred_st, losses, slide_ids_list = [], [], [], []
        with torch.no_grad():
            for iteration, data in enumerate(tqdm(val_loader, leave=False)):
                patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
                patch, st = patch.to(args.device), st.to(args.device)

                y = st_estimater(patch)
                
                loss_prime = loss_function(y["y"][:, target_gene_idx], st[:, target_gene_idx], np.array(slide_ids))
                loss = sum(loss_prime)/len(target_gene_idx)
            
                gt_st.extend(st.cpu().detach().numpy()), pred_st.extend(y["y"].cpu().detach().numpy())
                losses.append(loss.item())
                slide_ids_list.extend(slide_ids)

        gt_st, pred_st, slide_ids_list = np.array(gt_st), np.array(pred_st), np.array(slide_ids_list)
        metrics = metric_func(gt_st, pred_st, slide_ids_list, gene_name_dict)

        log_dict["val_mse"].append(metrics["mse"]), log_dict["val_pearsonr"].append(metrics["pearsonr"]), log_dict["val_spearman"].append(metrics["spearman"])
        log_dict["val_mse_loss"].append(np.array(losses).mean())

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] val loss: %.4f, @ mse: %.4f, pearsonr: %.4f, spearman: %.4f' %
                     (epoch+1, args.num_epochs, e_time-s_time, log_dict["val_mse_loss"][-1], log_dict["val_mse"][-1], log_dict["val_pearsonr"][-1], log_dict["val_spearman"][-1]))
        
        ################## test ###################
        s_time = time()
        st_estimater.eval()
        CutOff_estimater.eval()
        gt_st, pred_st, losses, slide_ids_list = [], [], [], []
        with torch.no_grad():
            for iteration, data in enumerate(tqdm(test_loader, leave=False)):
                patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
                patch, st = patch.to(args.device), st.to(args.device)

                y = st_estimater(patch)

                gt_st.extend(st.cpu().detach().numpy()), pred_st.extend(y["y"].cpu().detach().numpy())
                slide_ids_list.extend(slide_ids)

        gt_st, pred_st, slide_ids_list = np.array(gt_st), np.array(pred_st), np.array(slide_ids_list)
        metrics = metric_func(gt_st, pred_st, slide_ids_list, gene_name_dict)
        
        log_dict["test_mse"].append(metrics["mse"]), log_dict["test_pearsonr"].append(metrics["pearsonr"]), log_dict["test_spearman"].append(metrics["spearman"])

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] , @ mse: %.4f, pearsonr: %.4f, spearman: %.4f' %
                    (epoch+1, args.num_epochs, e_time-s_time, log_dict["test_mse"][-1], log_dict["test_pearsonr"][-1], log_dict["test_spearman"][-1]))
        logging.info('===============================')

        if best_val_pearsonr < log_dict["val_pearsonr"][-1]:
            best_val_pearsonr = log_dict["val_pearsonr"][-1]
            cnt = 0
            best_epoch = epoch
            torch.save(st_estimater.state_dict(), ("%s/model/fold=%d_seed=%d-best_st_estimater.pkl") % (args.output_path, args.fold, args.seed))
        else:
            cnt += 1
            if args.patience == cnt:
                break

        logging.info('best epoch: %d , mse: %.4f, pearsonr: %.4f, spearman: %.4f' %
                            (best_epoch+1, log_dict["test_mse"][best_epoch], log_dict["test_pearsonr"][best_epoch], log_dict["test_spearman"][best_epoch]))

        make_loss_graph(args,log_dict['train_mse_loss'], log_dict['val_mse_loss'], "%s/loss_graph/fold=%d_seed=%d_loss-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_mse'], log_dict['val_mse'], log_dict['test_mse'], "%s/acc_graph/fold=%d_seed=%d_mse-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_pearsonr'], log_dict['val_pearsonr'], log_dict['test_pearsonr'], "%s/acc_graph/fold=%d_seed=%d_pearsonr-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_spearman'], log_dict['val_spearman'], log_dict['test_spearman'], "%s/acc_graph/fold=%d_seed=%d_spearman-graph.png" % (args.output_path, args.fold, args.seed))
        Make_CutOff_k_graph(args, log_dict["k"], "%s/task_wgt/fold=%d_seed=%d_thresh_hold_k.png" % (args.output_path, args.fold, args.seed))
        np.save("%s/log_dict/fold=%d_seed=%d_log" % (args.output_path, args.fold, args.seed) , log_dict)
    return

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

def meta_learning_converge_check(args, st_estimater, weight_estimater, weight_estimater_optimizer, train_eval_loader, val_loader, meta_cnt, best_meta_val_pearson, meta_val_pearson, best_roop, roop, task_weighted_pcc_loss, metric_func, make_bag_acc_graph, make_task_wgt_graph, make_task_row_wgt_graph, log_dict, epoch, eval_gene_idx, train_highly_gene_idx, train_low_gene_idx, gene_name_dict, _, converge = False):
    fix_seed(args.seed)
    st_estimater_copy = copy.deepcopy(st_estimater).to(args.device)
    st_estimater_copy_optimizer = torch.optim.Adam(st_estimater_copy.parameters(), lr=args.lr)
    for __ in range(1):
        st_estimater_copy.train()
        weight_estimater.eval()
        gt_st, pred_st, losses, slide_ids_list = [], [], [], []
        for iteration, data in enumerate(tqdm(train_eval_loader, leave=False)): #enumerate(tqdm(train_loader, leave=False)):
            patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
            patch, st = patch.to(args.device), st.to(args.device)

            y = st_estimater_copy(patch)
            loss_weight = weight_estimater(y["y"], eval_gene_idx, train_highly_gene_idx, train_low_gene_idx)  # generate auxiliary labels

            # reset optimizers with zero gradient
            st_estimater_copy_optimizer.zero_grad()
            weight_estimater_optimizer.zero_grad()
        
            loss = task_weighted_pcc_loss(y["y"], st, np.array(slide_ids))
            train_loss = sum(loss_weight["w"] * loss) /y["y"].shape[1]
            
            train_loss.backward()
            st_estimater_copy_optimizer.step()
        
            gt_st.extend(st.cpu().detach().numpy()), pred_st.extend(y["y"].cpu().detach().numpy())
            losses.append(train_loss.item())
            slide_ids_list.extend(slide_ids)

        gt_st, pred_st, slide_ids_list = np.array(gt_st), np.array(pred_st), np.array(slide_ids_list)
        metrics = metric_func(gt_st, pred_st, slide_ids_list, gene_name_dict) 
        print("mata learning converge check: Train pcc= %.7f" % metrics["pearsonr"])  
        
    st_estimater_copy.eval()
    gt_st, pred_st, losses, slide_ids_list = [], [], [], []
    with torch.no_grad():
        for iteration, data in enumerate(tqdm(val_loader, leave=False)):
            patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
            patch, st = patch.to(args.device), st.to(args.device)

            y = st_estimater_copy(patch)
            
            loss_prime = task_weighted_pcc_loss(y["y"][:, eval_gene_idx], st[:, eval_gene_idx], np.array(slide_ids))
            loss = sum(loss_prime*loss_weight["w"][eval_gene_idx])/len(eval_gene_idx)
        
            gt_st.extend(st.cpu().detach().numpy()), pred_st.extend(y["y"].cpu().detach().numpy())
            losses.append(loss.item())
            slide_ids_list.extend(slide_ids)

    gt_st, pred_st, slide_ids_list = np.array(gt_st), np.array(pred_st), np.array(slide_ids_list)
    metrics = metric_func(gt_st, pred_st, slide_ids_list, gene_name_dict) 
    # print("mata learning converge check: Val pcc= %.7f" % metrics["pearsonr"])  
    meta_val_pearson.append(metrics["pearsonr"])
    make_bag_acc_graph(args, meta_val_pearson, meta_val_pearson, meta_val_pearson, "%s/acc_graph/fold=%d_seed=%d_metalarning_pearson-graph.png" % (args.output_path, args.fold, args.seed)) 
    
    logging.info('[best roop: %d, current roop: %d, @ pearsonr: %.7f, ' % (best_roop, roop, meta_val_pearson[-1]))
    roop+=1

    if _==0:
        log_dict["task_wgt"] = loss_weight["w"].cpu().detach().numpy().reshape(1, -1)
        log_dict["task_row_wgt"] = loss_weight["row_w"].cpu().detach().numpy().reshape(1, -1)
    else:
        log_dict["task_wgt"] = np.concatenate((log_dict["task_wgt"], loss_weight["w"].cpu().detach().numpy().reshape(1, -1)))
        log_dict["task_row_wgt"] = np.concatenate((log_dict["task_row_wgt"], loss_weight["row_w"].cpu().detach().numpy().reshape(1, -1)))
        
    if _!=0 and _%5==0:
        make_task_wgt_line_graph(args, log_dict["task_wgt"][-1], all_gene_name, gene_name_dict["train_hvg_Sort"],  "%s/task_wgt/fold=%d_seed=%d_epoch=%d_itr=%d_task_wgt_line_graph.png" % (args.output_path, args.fold, args.seed, epoch, _))
    make_task_wgt_graph(args, log_dict["task_wgt"],  "%s/task_wgt/fold=%d_seed=%d_epoch=%d_task_wgt_graph.png" % (args.output_path, args.fold, args.seed, epoch))
    make_task_row_wgt_graph(args, log_dict["task_row_wgt"],  "%s/task_wgt/fold=%d_seed=%d_epoch=%d_task_row_wgt_graph.png" % (args.output_path, args.fold, args.seed, epoch))

    if best_meta_val_pearson < meta_val_pearson[-1]:
        best_meta_val_pearson = meta_val_pearson[-1]
        meta_cnt = 0
        best_roop = roop
        torch.save(weight_estimater.state_dict(), ("%s/model/fold=%d_seed=%d_epoch=%d-best_weight_estimater.pkl") % (args.output_path, args.fold, args.seed, epoch))
        return converge, meta_cnt, best_roop, best_meta_val_pearson, meta_val_pearson, best_roop, roop, log_dict
    else:
        meta_cnt += 1
        if 10 == meta_cnt:
            converge = True  
        return converge, meta_cnt, best_roop, best_meta_val_pearson, meta_val_pearson, best_roop, roop, log_dict
        
    

def train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function, metric_func, gene_name_dict):
    st_estimater, weight_estimater = model["st_estimater"], model["weight_estimater"]
    st_estimater, weight_estimater = st_estimater.to(args.device), weight_estimater.to(args.device) 
    st_estimater_optimizer, weight_estimater_optimizer = optimizer["st_estimater"], optimizer["weight_estimater"]
    
    pcc_loss, task_weighted_pcc_loss  = loss_function["prime"], loss_function["aux"]
    prime_gene_name, aux_gene_name, all_gene_name =  gene_name_dict["eval"], gene_name_dict["train"][np.where(~np.isin(gene_name_dict["train"], gene_name_dict["eval"]))[0]], gene_name_dict["train"]
    eval_gene_idx, aux_gene_idx = np.where(np.isin(all_gene_name, prime_gene_name))[0], np.where(np.isin(all_gene_name, aux_gene_name))[0]

    train_highly_gene_idx = np.where(np.isin(all_gene_name, gene_name_dict["train_highly_gene"]))[0]
    train_low_gene_idx = np.where(np.isin(all_gene_name, gene_name_dict["train_low_gene"]))[0]
    
    fix_seed(args.seed)
    log_dict = {"train_mse":[], "train_pearsonr":[], "train_spearman":[], "train_mse_loss":[], "train_weit_esstimater_loss":[], "train_weit_esstimater_loss_dist":[],
                "val_mse":[], "val_pearsonr":[], "val_spearman":[], "val_mse_loss":[],
                "test_mse":[], "test_pearsonr":[], "test_spearman":[], "test_mse_loss":[]}
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("%s/log_dict/fold=%d_seed=%d_training_setting.log" %  (args.output_path, args.fold, args.seed))
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
    logging.info(args)

    train_data_loader, meta_train_loader, meta_val_loader, train_eval_loader = train_loader["train"], train_loader["meta_train_sampler"], train_loader["meta_val_sampler"], train_loader["train_val_loader"]

    best_val_pearsonr = -100000000
    cnt = 0
    for epoch in range(args.num_epochs):
        if epoch%5==0:
            ############ meta learning ###################
            logging.info("####### meta learning step ######### ")
            st_estimater_copy = copy.deepcopy(st_estimater).to(args.device)
            # evaluating training data (meta-training step, update on theta_2)
            meta_val_pearson, loss_dist = [], []
            best_meta_val_pearson = -100000000
            meta_cnt, roop, best_roop = 0, 0, 0
            for _ in range(1000):
                st_estimater_copy.train()
                weight_estimater.train()
                gt_st, pred_st, losses, slide_ids_list = [], [], [], []
                for t_data, v_data in tqdm(zip(meta_train_loader, cycle(meta_val_loader)), total=len(meta_train_loader), leave=False):
                    inner_opt = torch.optim.SGD(st_estimater_copy.parameters(), lr=3e-2)
                    weight_estimater_optimizer.zero_grad()
                
                    with higher.innerloop_ctx(st_estimater_copy, inner_opt, copy_initial_weights=False, track_higher_grads=True) as (fnet, diffopt):
                        t_patch, t_st, t_slide_ids = t_data["patch"], t_data["st"], t_data["slide_ids"]
                        t_patch, t_st = t_patch.to(args.device), t_st.to(args.device)
                        logging.info("== Inner loop ==")
                        for k in range(5):
                            y = fnet(t_patch)
                            loss_weight = weight_estimater(y["y"], eval_gene_idx, train_highly_gene_idx, train_low_gene_idx)  # generate auxiliary labels
                            loss = task_weighted_pcc_loss(y["y"], t_st, np.array(t_slide_ids))
                            train_loss = sum(loss_weight["w"] * loss) /y["y"].shape[1]
                            diffopt.step(train_loss)
                            logging.info("loss: %.7f" % train_loss.item())
                        
                        #######  update weight model on train data
                        v_patch, v_st, v_slide_ids = v_data["patch"], v_data["st"], v_data["slide_ids"]
                        v_patch, v_st = v_patch.to(args.device), v_st.to(args.device)
                        y = fnet(v_patch)
                        
                        loss_prime = task_weighted_pcc_loss(y["y"][:, eval_gene_idx], v_st[:, eval_gene_idx], np.array(v_slide_ids))
                        val_loss = sum(loss_prime)/len(prime_gene_name)
                        val_loss.backward()
                        # logging.info("== outer loop: loss: %.7f" % val_loss.item())
                        
                    weight_estimater_optimizer.step()
                    gt_st.extend(v_st.cpu().detach().numpy()), pred_st.extend(y["y"].cpu().detach().numpy())
                    slide_ids_list.extend(v_slide_ids)

                #######  mata learning converge check             
                converge, meta_cnt, best_weight_estimater, best_meta_val_pearson, meta_val_pearson, best_roop, roop, log_dict = meta_learning_converge_check(args, st_estimater, weight_estimater, weight_estimater_optimizer, train_eval_loader, val_loader, meta_cnt, best_meta_val_pearson, meta_val_pearson, best_roop, roop, task_weighted_pcc_loss, metric_func, make_bag_acc_graph, make_task_wgt_graph, make_task_row_wgt_graph, log_dict, epoch, eval_gene_idx, train_highly_gene_idx, train_low_gene_idx, gene_name_dict, _)   
                if converge:
                    weight_estimater.load_state_dict(torch.load(("%s/model/fold=%d_seed=%d_epoch=%d-best_weight_estimater.pkl") % (args.output_path, args.fold, args.seed, epoch) ,map_location=args.device))
                    weight_estimater_optimizer = torch.optim.Adam(model["weight_estimater"].parameters(), lr=args.weight_estimater_lr)
                    break 
        

        ############ train ###################
        s_time = time()
        st_estimater.train()
        weight_estimater.eval()
        gt_s, pred_st, losses, slide_ids_list = [], [], [], []
        for iteration, data in enumerate(tqdm(train_data_loader, leave=False)): #enumerate(tqdm(train_loader, leave=False)):
            patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
            patch, st = patch.to(args.device), st.to(args.device)

            y = st_estimater(patch)
            loss_weight = weight_estimater(y["y"], eval_gene_idx, train_highly_gene_idx, train_low_gene_idx)  # generate auxiliary labels

            # reset optimizers with zero gradient
            st_estimater_optimizer.zero_grad()
            weight_estimater_optimizer.zero_grad()
         
            loss = task_weighted_pcc_loss(y["y"], st, np.array(slide_ids))
            train_loss = sum(loss_weight["w"] * loss) /y["y"].shape[1]
            
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
        weight_estimater.eval()
        gt_st, pred_st, losses, slide_ids_list = [], [], [], []
        with torch.no_grad():
            for iteration, data in enumerate(tqdm(val_loader, leave=False)):
                patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
                patch, st = patch.to(args.device), st.to(args.device)

                y = st_estimater(patch)
                
                loss_prime = task_weighted_pcc_loss(y["y"][:, eval_gene_idx], st[:, eval_gene_idx], np.array(slide_ids))
                loss = sum(loss_prime*loss_weight["w"][eval_gene_idx])/len(eval_gene_idx)
            
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
        
        if args.is_test == True:
            ################## test ###################
            s_time = time()
            st_estimater.eval()
            weight_estimater.eval()
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


        np.save("%s/log_dict/fold=%d_seed=%d_log" % (args.output_path, args.fold, args.seed) , log_dict)
    return

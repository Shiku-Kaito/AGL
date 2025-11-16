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

def train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function, metric_func, gene_name_dict):
    st_estimater, weight_estimater = model["st_estimater"], model["weight_estimater"]
    st_estimater, weight_estimater = st_estimater.to(args.device), weight_estimater.to(args.device) 
    st_estimater_optimizer, weight_estimater_optimizer = optimizer["st_estimater"], optimizer["weight_estimater"]
    
    pcc_loss, task_weighted_pcc_loss  = loss_function["prime"], loss_function["aux"]
    prime_gene_name, aux_gene_name, all_gene_name =  gene_name_dict["eval"], gene_name_dict["train"][np.where(~np.isin(gene_name_dict["train"], gene_name_dict["eval"]))[0]], gene_name_dict["train"]
    eval_gene_idx, aux_gene_idx = np.where(np.isin(all_gene_name, prime_gene_name))[0], np.where(np.isin(all_gene_name, aux_gene_name))[0]
    
    fix_seed(args.seed)
    log_dict = {"train_mse":[], "train_pearsonr":[], "train_spearman":[], "train_mse_loss":[], "train_weit_esstimater_loss":[], "train_weit_esstimater_loss_dist":[],
                "val_mse":[], "val_pearsonr":[], "val_spearman":[], "val_mse_loss":[],
                "test_mse":[], "test_pearsonr":[], "test_spearman":[], "test_mse_loss":[]}
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("%s/log_dict/fold=%d_seed=%d_training_setting.log" %  (args.output_path, args.fold, args.seed))
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
    logging.info(args)

    train_data_loader, train_meta_loader = train_loader["train"], train_loader["meta"]

    best_val_pearsonr = -100000000
    cnt = 0
    # weight_estimater_lr = args.weight_estimater_lr
    for epoch in range(args.num_epochs):
        # if (epoch + 1) % 10 == 0:
        #     weight_estimater_lr = weight_estimater_lr * 0.5

        ############ train ###################
        s_time = time()
        st_estimater.train()
        for _ in range(5):
            gt_st, pred_st, losses, slide_ids_list = [], [], [], []
            for iteration, data in enumerate(tqdm(train_data_loader, leave=False)): #enumerate(tqdm(train_loader, leave=False)):
                patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
                patch, st = patch.to(args.device), st.to(args.device)

                y = st_estimater(patch)
                loss_weight = weight_estimater(patch)  # generate auxiliary labels

                # reset optimizers with zero gradient
                st_estimater_optimizer.zero_grad()
                weight_estimater_optimizer.zero_grad()

                # if epoch > 10:
                loss_prime = task_weighted_pcc_loss(y["y"][:, eval_gene_idx], st[:, eval_gene_idx], loss_weight["w"][-len(prime_gene_name):], np.array(slide_ids))
                loss_aux = task_weighted_pcc_loss(y["y"][:, aux_gene_idx], st[:, aux_gene_idx], loss_weight["w"][:-len(prime_gene_name)], np.array(slide_ids))
                train_loss = 1 - ((loss_prime+loss_aux)/y["y"].shape[1])
                    
                # else:    
                #     ones_tensor = torch.ones(y["y"].shape[1], device=patch.device)  # yと同じデバイスに作成
                #     loss_weight = ones_tensor / ones_tensor.sum(dim=0, keepdim=True)
                #     loss_weight = loss_weight*10000
                #     train_loss = 1 - (task_weighted_pcc_loss(y["y"], st, loss_weight, np.array(slide_ids))/y["y"].shape[1])
                
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


        # if epoch > 10:
        # evaluating training data (meta-training step, update on theta_2)
        weight_estimater.train()
        losses, loss_dist = [], []
        for iteration, data in enumerate(tqdm(train_meta_loader, leave=False)): #enumerate(tqdm(train_loader, leave=False)):
            patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
            patch, st = patch.to(args.device), st.to(args.device)

            y = st_estimater(patch)
            loss_weight = weight_estimater(patch)  # generate auxiliary labels

            # reset optimizers with zero gradient
            st_estimater_optimizer.zero_grad()
            weight_estimater_optimizer.zero_grad()
            
            loss_prime = task_weighted_pcc_loss(y["y"][:, eval_gene_idx], st[:, eval_gene_idx], loss_weight["w"][-len(prime_gene_name):], np.array(slide_ids))
            loss_aux = task_weighted_pcc_loss(y["y"][:, aux_gene_idx], st[:, aux_gene_idx], loss_weight["w"][:-len(prime_gene_name)], np.array(slide_ids))
            train_loss = 1 - ((loss_prime+loss_aux)/y["y"].shape[1])
            pre_loss_prime = (1 - (loss_prime/len(prime_gene_name))).item()

            # current theta_1
            fast_weights = OrderedDict((name, param) for (name, param) in st_estimater.named_parameters())

            # create_graph flag for computing second-derivative
            grads = torch.autograd.grad(train_loss, st_estimater.parameters(), create_graph=True)
            data = [p.data for p in list(st_estimater.parameters())]

            # compute theta_1^+ by applying sgd on multi-task loss
            fast_weights = OrderedDict((name, param - args.lr * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))

            # compute primary loss with the updated thetat_1^+
            y = stateless.functional_call(st_estimater, fast_weights, (patch,))
            loss_prime = 1 - ((task_weighted_pcc_loss(y["y"][:, eval_gene_idx], st[:, eval_gene_idx], loss_weight["w"][-len(prime_gene_name):], np.array(slide_ids)))/len(prime_gene_name))

            loss_prime.backward()
            
            weight_estimater_optimizer.step()

                
            losses.append(loss_prime.item()), loss_dist.append(pre_loss_prime - loss_prime.item())

        log_dict["train_weit_esstimater_loss"].append(np.array(losses).mean())
        log_dict["train_weit_esstimater_loss_dist"].append(np.array(loss_dist).mean())


        ################# validation ####################
        s_time = time()
        st_estimater.eval()
        gt_st, pred_st, losses, slide_ids_list = [], [], [], []
        with torch.no_grad():
            for iteration, data in enumerate(tqdm(val_loader, leave=False)):
                patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
                patch, st = patch.to(args.device), st.to(args.device)

                y = st_estimater(patch)
                
                loss = 1 - (pcc_loss(y["y"][:, eval_gene_idx], st[:, eval_gene_idx], np.array(slide_ids))/len(prime_gene_name))
            
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

        if epoch==0:
            log_dict["task_wgt"] = loss_weight["w"].cpu().detach().numpy().reshape(1, -1)
        else:
            log_dict["task_wgt"] = np.concatenate((log_dict["task_wgt"], loss_weight["w"].cpu().detach().numpy().reshape(1, -1)))

        if best_val_pearsonr < log_dict["val_pearsonr"][-1]:
            best_val_pearsonr = log_dict["val_pearsonr"][-1]
            cnt = 0
            best_epoch = epoch
            torch.save(st_estimater.state_dict(), ("%s/model/fold=%d_seed=%d-best_st_estimater.pkl") % (args.output_path, args.fold, args.seed))
            torch.save(weight_estimater.state_dict(), ("%s/model/fold=%d_seed=%d-best_weight_estimater.pkl") % (args.output_path, args.fold, args.seed))
            
        else:
            cnt += 1
            if args.patience == cnt:
                break

        logging.info('best epoch: %d , mse: %.4f, pearsonr: %.4f, spearman: %.4f' %
                        (best_epoch+1, log_dict["test_mse"][best_epoch], log_dict["test_pearsonr"][best_epoch], log_dict["test_spearman"][best_epoch]))

        make_loss_graph(args,log_dict['train_weit_esstimater_loss_dist'], log_dict['train_weit_esstimater_loss_dist'], "%s/loss_graph/fold=%d_seed=%d_weight_esstimater_loss_dist-graph.png" % (args.output_path, args.fold, args.seed))
        make_loss_graph(args,log_dict['train_weit_esstimater_loss'], log_dict['train_weit_esstimater_loss'], "%s/loss_graph/fold=%d_seed=%d_weight_esstimater_loss-graph.png" % (args.output_path, args.fold, args.seed))
        make_loss_graph(args,log_dict['train_mse_loss'], log_dict['val_mse_loss'], "%s/loss_graph/fold=%d_seed=%d_loss-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_mse'], log_dict['val_mse'], log_dict['test_mse'], "%s/acc_graph/fold=%d_seed=%d_mse-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_pearsonr'], log_dict['val_pearsonr'], log_dict['test_pearsonr'], "%s/acc_graph/fold=%d_seed=%d_pearsonr-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_spearman'], log_dict['val_spearman'], log_dict['test_spearman'], "%s/acc_graph/fold=%d_seed=%d_spearman-graph.png" % (args.output_path, args.fold, args.seed))

        make_task_wgt_graph(args, log_dict["task_wgt"], "%s/task_wgt/fold=%d_seed=%d_task_wgt_graph.png" % (args.output_path, args.fold, args.seed))

        np.save("%s/log_dict/fold=%d_seed=%d_log" % (args.output_path, args.fold, args.seed) , log_dict)
        
            
    return
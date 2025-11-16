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

def train_net(args, 
              model,
              optimizer,
              train_loader, 
              val_loader, 
              test_loader, 
              loss_function, 
              metric_func, 
              gene_name_dict):
    
    fix_seed(args.seed)
    log_dict = {"train_mse":[], "train_pearsonr":[], "train_spearman":[], "train_mse_loss":[],
                "val_mse":[], "val_pearsonr":[], "val_spearman":[], "val_mse_loss":[],
                "test_mse":[], "test_pearsonr":[], "test_spearman":[], "test_mse_loss":[]}
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("%s/log_dict/fold=%d_seed=%d_training_setting.log" %  (args.output_path, args.fold, args.seed))
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
    logging.info(args)

    best_val_pearsonr = -100000000
    cnt = 0
    for epoch in range(args.num_epochs):
        ############ train ###################
        s_time = time()
        model.train()
        gt_st, pred_st, losses, slide_ids_list = [], [], [], []
        for iteration, data in enumerate(tqdm(train_loader, leave=False)): #enumerate(tqdm(train_loader, leave=False)):
            patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
            patch, st = patch.to(args.device), st.to(args.device)

            y = model(patch)
            loss = loss_function(y["y"], st, np.array(slide_ids)).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            gt_st.extend(st.cpu().detach().numpy()), pred_st.extend(y["y"].cpu().detach().numpy())
            losses.append(loss.item())
            slide_ids_list.extend(slide_ids)

        gt_st, pred_st, slide_ids_list = np.array(gt_st), np.array(pred_st), np.array(slide_ids_list)
        metrics = metric_func(gt_st, pred_st, slide_ids_list, gene_name_dict)

        log_dict["train_mse"].append(metrics["mse"]), log_dict["train_pearsonr"].append(metrics["pearsonr"]), log_dict["train_spearman"].append(metrics["spearman"])
        log_dict["train_mse_loss"].append(np.array(losses).mean())

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] train loss: %.4f, @ mse: %.4f, pearsonr: %.4f, spearman: %.4f' %
                     (epoch+1, args.num_epochs, e_time-s_time, log_dict["train_mse_loss"][-1], log_dict["train_mse"][-1], log_dict["train_pearsonr"][-1], log_dict["train_spearman"][-1]))
        
        ################# validation ####################
        s_time = time()
        model.eval()
        gt_st, pred_st, losses, slide_ids_list = [], [], [], []
        with torch.no_grad():
            for iteration, data in enumerate(tqdm(val_loader, leave=False)):
                patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
                patch, st = patch.to(args.device), st.to(args.device)

                y = model(patch)
                loss = loss_function(y["y"], st, np.array(slide_ids)).mean()

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
        log_dict["test_mse"].append(metrics["mse"]), log_dict["test_pearsonr"].append(metrics["pearsonr"]), log_dict["test_spearman"].append(metrics["spearman"])

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] , @ mse: %.4f, pearsonr: %.4f, spearman: %.4f' %
                    (epoch+1, args.num_epochs, e_time-s_time, log_dict["test_mse"][-1], log_dict["test_pearsonr"][-1], log_dict["test_spearman"][-1]))
        logging.info('===============================')

        if best_val_pearsonr < log_dict["val_pearsonr"][-1]:
            best_val_pearsonr = log_dict["val_pearsonr"][-1]
            cnt = 0
            best_epoch = epoch
            torch.save(model.state_dict(), ("%s/model/fold=%d_seed=%d-best_model.pkl") % (args.output_path, args.fold, args.seed))

        else:
            cnt += 1
            if args.patience == cnt:
                break

        logging.info('best epoch: %d , mse: %.4f, pearsonr: %.4f, spearman: %.4f' %
                        (best_epoch+1, log_dict["test_mse"][best_epoch], log_dict["test_pearsonr"][best_epoch], log_dict["test_spearman"][best_epoch]))

        Make_loss_graph(args, log_dict['train_mse_loss'], log_dict['val_mse_loss'], 
                        "%s/loss_graph/fold=%d_seed=%d_loss-graph.png" % (args.output_path, args.fold, args.seed))
        Make_acc_graph(args, log_dict['train_mse'], log_dict['val_mse'], log_dict['test_mse'], 
                       "%s/acc_graph/fold=%d_seed=%d_mse-graph.png" % (args.output_path, args.fold, args.seed))
        Make_acc_graph(args, log_dict['train_pearsonr'], log_dict['val_pearsonr'], log_dict['test_pearsonr'], 
                       "%s/acc_graph/fold=%d_seed=%d_pearsonr-graph.png" % (args.output_path, args.fold, args.seed))
        Make_acc_graph(args, log_dict['train_spearman'], log_dict['val_spearman'], log_dict['test_spearman'], 
                       "%s/acc_graph/fold=%d_seed=%d_spearman-graph.png" % (args.output_path, args.fold, args.seed))
        np.save("%s/log_dict/fold=%d_seed=%d_log" % (args.output_path, args.fold, args.seed) , log_dict)
    return



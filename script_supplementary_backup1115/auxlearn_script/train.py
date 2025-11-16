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
from auxlearn_script.auxilearn.hypernet import MonoHyperNet, MonoLinearHyperNet, MonoNonlinearHyperNet
from auxlearn_script.auxilearn.optim import MetaOptimizer
import torch.optim as optim

# ======
# params
# ======
meta_lr = 1e-3
meta_wd = 1e-4
hypergrad_every = 50

# ==============
# hypergrad step
# ==============
def hyperstep(args, st_estimater, meta_val_loader, train_loader, auxiliary_net, meta_optimizer, loss_function, eval_gene_idx):
    meta_val_loss = .0
    for n_val_step, data in enumerate(meta_val_loader):
        # if n_val_step < args.n_meta_loss_accum:
        if n_val_step < 4:
            patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
            patch, st = patch.to(args.device), st.to(args.device)

            y = st_estimater(patch)

            # loss = loss_function(y["y"], st, np.array(slide_ids))

            # meta_val_loss += loss[:, 0].mean(0)
            meta_val_loss += loss_function(y["y"][:, eval_gene_idx], st[:, eval_gene_idx], np.array(slide_ids)).mean()

    # inner_loop_end_train_loss, e.g. dL_train/dw
    total_meta_train_loss = 0.
    for n_train_step, data in enumerate(train_loader):
        # if n_train_step < args.n_meta_loss_accum:
        if n_train_step < 4:
            patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
            patch, st = patch.to(args.device), st.to(args.device)

            y = st_estimater(patch)

            loss = loss_function(y["y"], st, np.array(slide_ids))
            loss = loss.unsqueeze(0)

            meta_train_loss = auxiliary_net(loss)
            total_meta_train_loss += meta_train_loss

    # hyperpatam step
    curr_hypergrads = meta_optimizer.step(
        val_loss=meta_val_loss,
        train_loss=total_meta_train_loss,
        aux_params=list(auxiliary_net.parameters()),
        parameters=list(st_estimater.parameters()),
        return_grads=True
    )

    return curr_hypergrads



def train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function, metric_func, gene_name_dict):
    fix_seed(args.seed)
    log_dict = {"train_mse":[], "train_pearsonr":[], "train_spearman":[], "train_mse_loss":[],
                "val_mse":[], "val_pearsonr":[], "val_spearman":[], "val_mse_loss":[],
                "test_mse":[], "test_pearsonr":[], "test_spearman":[], "test_mse_loss":[]}
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("%s/log_dict/fold=%d_seed=%d_training_setting.log" %  (args.output_path, args.fold, args.seed))
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
    logging.info(args)
    
    prime_gene_name, aux_gene_name, all_gene_name =  gene_name_dict["eval"], gene_name_dict["train"][np.where(~np.isin(gene_name_dict["train"], gene_name_dict["eval"]))[0]], gene_name_dict["train"]
    eval_gene_idx, aux_gene_idx = np.where(np.isin(all_gene_name, prime_gene_name))[0], np.where(np.isin(all_gene_name, aux_gene_name))[0]
    
    # ===============
    # auxiliary model
    # ===============
    auxiliary_net = MonoNonlinearHyperNet(main_task=eval_gene_idx, input_dim=len(gene_name_dict["train"]), hidden_sizes=64).to(args.device)
        
    prim_optimizer = optimizer
    
    meta_opt = optim.SGD(
    auxiliary_net.parameters(),
    lr=meta_lr,
    momentum=.9,
    weight_decay=meta_wd
)

    meta_optimizer = MetaOptimizer(
        meta_optimizer=meta_opt, hpo_lr=1e-4, truncate_iter=3, max_grad_norm=25
    )

    step = 0
    best_val_pearsonr = -100000000
    cnt = 0
    for epoch in range(args.num_epochs):
        ############ train ###################
        s_time = time()
        model.train()
        gt_st, pred_st, losses, slide_ids_list = [], [], [], []
        for iteration, data in enumerate(tqdm(train_loader, leave=False)): #enumerate(tqdm(train_loader, leave=False)):
            step += 1
            patch, st, slide_ids = data["patch"], data["st"], data["slide_ids"]
            patch, st = patch.to(args.device), st.to(args.device)

            prim_optimizer.zero_grad()
            
            y = model(patch)
            loss = loss_function(y["y"], st, np.array(slide_ids))
            loss = loss.unsqueeze(0)
            loss = auxiliary_net(loss)
            
            loss.backward()
            prim_optimizer.step()

            # hyperparams step
            if step % hypergrad_every == 0:
                logging.info("Aux net update")
                curr_hypergrads = hyperstep(args, model, val_loader, train_loader, auxiliary_net, meta_optimizer, loss_function, eval_gene_idx)

                if isinstance(auxiliary_net, MonoHyperNet):
                    # monotonic network
                    auxiliary_net.clamp()
                    
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
        
        if args.is_test == True:
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

        make_loss_graph(args,log_dict['train_mse_loss'], log_dict['val_mse_loss'], "%s/loss_graph/fold=%d_seed=%d_loss-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_mse'], log_dict['val_mse'], log_dict['test_mse'], "%s/acc_graph/fold=%d_seed=%d_mse-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_pearsonr'], log_dict['val_pearsonr'], log_dict['test_pearsonr'], "%s/acc_graph/fold=%d_seed=%d_pearsonr-graph.png" % (args.output_path, args.fold, args.seed))
        make_bag_acc_graph(args, log_dict['train_spearman'], log_dict['val_spearman'], log_dict['test_spearman'], "%s/acc_graph/fold=%d_seed=%d_spearman-graph.png" % (args.output_path, args.fold, args.seed))

        np.save("%s/log_dict/fold=%d_seed=%d_log" % (args.output_path, args.fold, args.seed) , log_dict)
    return



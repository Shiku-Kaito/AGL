import argparse
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from utils import *
from dataloader  import *

from resnet_STnet_script.train import train_net as resnet_STnet_train_net
from resnet_STnet_script.model import resnet_ST_net
from resnet_STnet_script.eval import eval_net as resnet_STnet_eval_net

from threshIndexOptimize_2stageLearning_script.train import train_net as threshIndexOptimize_2stageLearning_script_train_net
from threshIndexOptimize_2stageLearning_script.eval import eval_net as threshIndexOptimize_2stageLearning_script_eval_net

from Uncertainty_script.train import train_net as Uncertainty_train_net
from Uncertainty_script.eval import eval_net as Uncertainty_eval_net
from Uncertainty_script.uncetainty_loss import MultiTaskLossWrapper

from losses import *

from auxlearn_script.train import train_net as auxlearn_train_net

from amal_script.train import train_net as AMAL_train_net
from amal_script.eval import eval_net as AMAL_eval_net
from amal_script.weight_estimater_model import MAXL_weight_estimater as amal_gene_mask_estimater

from Model2Step2Learning_threshOptimize_script.weight_estimater_model import MAXL_weight_estimater as Model2Step2Learning_threshOptimize_gene_mask_estimater
from threshIndexOptimize_2stageLearning_script.model import MAXL_ST_estimater as MAXL_task_weight_ST_estimater

def get_module(args):
    if args.module ==  "resnet_ST_net":
        args.mode = "resnet_ST_net"
        train_loader, val_loader, test_loader, gene_name_dict, all_gene_num = hest_1slide_load(args) 

        # Model
        model = resnet_ST_net(args, all_gene_num)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss
        if args.loss == 'mse':
            loss_function = MSE_loss()
        elif args.loss == 'pcc':
            loss_function = PearsonCorrCoef_loss
            
        metric_func = culc_metric
    
        # Train net
        train_net = resnet_STnet_train_net
        eval_net = resnet_STnet_eval_net

    elif args.module ==  "Uncertainty":
        args.mode = "Uncertainty"
        # Dataloader
        train_loader, val_loader, test_loader, gene_name_dict, all_gene_num = hest_1slide_load(args) 

        # Model
        model = None
        
        loss_function = MultiTaskLossWrapper(all_gene_num, PearsonCorrCoef_loss)
        loss_function = loss_function.to(args.device)
        optimizer = torch.optim.Adam(loss_function.parameters(), lr=args.lr)
        # Loss 
        metric_func = culc_metric
    
        # Train net
        train_net = Uncertainty_train_net
        eval_net = Uncertainty_eval_net

    elif args.module ==  "auxlearn":
        args.mode = "auxlearn"
        # Dataloader
        train_loader, val_loader, test_loader, gene_name_dict, all_gene_num = hest_1slide_load(args) 

        # Model
        model = resnet_ST_net(args, all_gene_num)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # Loss 
        # Loss
        if args.loss == 'mse':
            loss_function = MSE_loss()
        elif args.loss == 'pcc':
            loss_function = PearsonCorrCoef_loss
            
        metric_func = culc_metric
    
        # Train net
        train_net = auxlearn_train_net
        eval_net = resnet_STnet_eval_net

    elif args.module ==  "AMAL":
        args.mode = "AMAL"
        # Dataloader
        train_loader, val_loader, test_loader, gene_name_dict, all_gene_num = hest_1slide_2stage_meta_load(args) 

        # Model
        model = {"st_estimater": MAXL_task_weight_ST_estimater(args, all_gene_num).to(args.device), "weight_estimater": amal_gene_mask_estimater(args, train_highly_gene=gene_name_dict["train_highly_gene"], train_low_gene=gene_name_dict["train_low_gene"]).to(args.device)}
        optimizer = {"st_estimater": torch.optim.Adam(model["st_estimater"].parameters(), lr=args.lr), "weight_estimater": torch.optim.Adam(model["weight_estimater"].parameters(), lr=args.weight_estimater_lr)}
        # Loss
        if args.loss == 'mse':
            loss_function = {"prime": MSE_loss(), "aux": Weighted_MSE_loss()}
        elif args.loss == 'pcc':
            loss_function = {"prime": PearsonCorrCoef_loss, "aux": PearsonCorrCoef_loss}
        
        metric_func = culc_metric
        # Train net
        train_net = AMAL_train_net
        eval_net = AMAL_eval_net



    elif args.module ==  "threshIndexOptimize_2stageLearning":
        args.mode = "threshIndexOptimize_2stageLearning_Threshlr=%.8f_STmodel_lr=%.5f_sumplingNum=%d_iniWgt=0.0_TempSoftmax_temper=%.3f" % (args.weight_estimater_lr, args.lr, args.sampling_num, args.temper)
        
        # Dataloader
        train_loader, val_loader, test_loader, gene_name_dict, all_gene_num = hest_1slide_2stage_meta_load(args) 

        # Model
        model = {"st_estimater": MAXL_task_weight_ST_estimater(args, all_gene_num).to(args.device), "weight_estimater": Model2Step2Learning_threshOptimize_gene_mask_estimater(args, train_highly_gene=gene_name_dict["train_highly_gene"], hvg_score=gene_name_dict["hvg_score"], all_gene_name=gene_name_dict["train"])}
        optimizer = {"st_estimater": torch.optim.Adam(model["st_estimater"].parameters(), lr=args.lr), "weight_estimater": torch.optim.Adam(model["weight_estimater"].parameters(), lr=args.weight_estimater_lr)}

        # Loss
        if args.loss == 'mse':
            loss_function = {"prime": MSE_loss(), "aux": Weighted_MSE_loss()}
        elif args.loss == 'pcc':
            loss_function = {"prime": PearsonCorrCoef_loss, "aux": PearsonCorrCoef_loss}
        
        metric_func = culc_metric
        # Train net
        train_net = threshIndexOptimize_2stageLearning_script_train_net
        eval_net = threshIndexOptimize_2stageLearning_script_eval_net

        
    else:
        print("Module ERROR!!!!!")

    return train_net, eval_net, model, optimizer, loss_function, train_loader, val_loader, test_loader, metric_func, gene_name_dict
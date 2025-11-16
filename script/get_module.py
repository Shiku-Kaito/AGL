import argparse
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from utils import *
from dataloader  import *

from STnet_script.train import train_net as STnet_train_net
from STnet_script.model import ST_net
from STnet_script.eval import eval_net as STnet_eval_net

from DkGSB_script.train import train_net as DkGSB_train_net
from DkGSB_script.eval import eval_net as DkGSB_eval_net
from DkGSB_script.model import ST_estimater
from DkGSB_script.cuttoff_estimater import CutOff_estimater

from losses import *

def get_module(args):
    if args.module ==  "PGL":
        args.mode = "PGL"
        train_loader, val_loader, test_loader, gene_name_dict, all_gene_num = load_data(args) 
        # Model
        model = ST_net(args, all_gene_num)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.alpha)
        # Loss
        loss_function = PearsonCorrCoef_loss
        # Metric
        metric_func = culc_metric
        # Train / evalation net
        train_net = STnet_train_net
        eval_net = STnet_eval_net

    elif args.module ==  "AGL":
        args.mode = "AGL"
        train_loader, val_loader, test_loader, gene_name_dict, all_gene_num = load_data(args) 
        # Model
        model = ST_net(args, all_gene_num)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.alpha)
        # Loss
        loss_function = PearsonCorrCoef_loss
        # Metric
        metric_func = culc_metric
        # Train / evalation net
        train_net = STnet_train_net
        eval_net = STnet_eval_net

    elif args.module ==  "AGL+DkGSB":
        args.mode = "AGL+DkGSB_alpha=%.5f_beta=%.5f_H=%d_tau=%.3f_sumplingNum=%d" % (args.alpha, args.beta, args.H, args.tau, args.bilevel_sampling_num)
        train_loader, val_loader, test_loader, gene_name_dict, all_gene_num, hvg_score = load_meta_learning_data(args) 
        # Model
        model = {"st_estimater": ST_estimater(args, all_gene_num).to(args.device), "CutOff_estimater": CutOff_estimater(args, aux_gene=gene_name_dict["aux"], hvg_score=hvg_score, all_gene_name=gene_name_dict["all"])}
        optimizer = {"st_estimater": torch.optim.Adam(model["st_estimater"].parameters(), lr=args.alpha), "CutOff_estimater": torch.optim.Adam(model["CutOff_estimater"].parameters(), lr=args.beta)}
        # Loss
        loss_function = PearsonCorrCoef_loss
        # Metric
        metric_func = culc_metric
        # Train / evalation net
        train_net = DkGSB_train_net
        eval_net = DkGSB_eval_net
        
    else:
        print("Module ERROR!!!!!")

    return train_net, eval_net, model, optimizer, loss_function, train_loader, val_loader, test_loader, metric_func, gene_name_dict
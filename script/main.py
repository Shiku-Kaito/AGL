import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import json
import logging
from utils import *
from get_module import get_module
import torch.nn.functional as F

def main(args):
    fix_seed(args.seed) 
    train_net, eval_net, model, optimizer, loss_function, train_loader, val_loader, test_loader, metric_func, gene_name_dict = get_module(args)
    
    args.output_path += "%s/%s/%s/" % (args.output_path,  args.dataset, args.mode)
    make_folder(args)

    if args.is_evaluation == False:
        train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function, metric_func, gene_name_dict)
        return
    else:   
        if args.module=="AGL+DkGSB":
            model = model["st_estimater"]
            model.load_state_dict(torch.load(("%s/model/fold=%d_seed=%d-best_st_estimater.pkl") % (args.output_path, args.fold, args.seed) ,map_location=args.device))
        else: 
            model.load_state_dict(torch.load(("%s/model/fold=%d_seed=%d-best_model.pkl") % (args.output_path, args.fold, args.seed) ,map_location=args.device))

        result_dict = eval_net(args, model, test_loader, metric_func, gene_name_dict)   
             
    return result_dict

if __name__ == '__main__':
    results_dict = {"mse":[], 
                    "pearsonr":[], 
                    "spearman":[], 
                    "k":[]}
    
    for fold in range(5):
        parser = argparse.ArgumentParser()
        # Data setting
        parser.add_argument('--fold', default=fold,
                            type=int, help='fold number')
        parser.add_argument('--dataset', default='TENX152',
                            type=str, help='TENX89 or TENX152 or TENX65')
        parser.add_argument('--target_genes_num', default=50,
                            type=int, help='fold number')
        # Training Setup
        parser.add_argument('--num_epochs', default=1000, type=int,
                            help='number of epochs for training.')
        parser.add_argument('--patience', default=20,
                            type=int, help='patience of early stopping')
        parser.add_argument('--device', default='cuda:0',
                            type=str, help='device')
        parser.add_argument('--batch_size', default=128,
                            type=int, help='batch size for training.')
        parser.add_argument('--seed', default=0,
                            type=int, help='seed value')
        parser.add_argument('--num_workers', default=0, type=int,
                            help='number of workers for training.')
        parser.add_argument('--alpha', default=3e-5,
                            type=float, help='learning rate')      
        parser.add_argument('--is_evaluation', default=0,
                            type=int, help='1 or 0')                     
        # Module Selection
        parser.add_argument('--module',default='PGL', 
                            type=str, help="PGL or AGL or AGL+DkGSB")
        parser.add_argument('--mode',default='',    # don't write!
                            type=str, help="")                        
        parser.add_argument('--output_path',
                            default='./result/', type=str, help="output file name")
        parser.add_argument('--input_path',
                            default='./', type=str, help="output file name")
        
        ############## task weight estimater ####################################
        parser.add_argument('--beta', default=3e-3, type=float, help='learning rate')   
        parser.add_argument('--bilevel_sampling_num', default=64, type=int, help='') 
        parser.add_argument('--tau', default=0.01, type=float, help='random or top')
        parser.add_argument('--H', default=12, type=int, help='random or top')
        args = parser.parse_args()
        
        if args.is_evaluation == False:
            main(args)
        else:
            result_dict = main(args)
            results_dict["mse"].append(result_dict["mse"]), results_dict["pearsonr"].append(result_dict["pearsonr"]), results_dict["spearman"].append(result_dict["spearman"])
            print("############ fold=%d ###########" % args.fold)
            print("@ MSE:%.5f, pearsonr:%.5f, spearman:%.5f" % (float(result_dict["mse"]), float(result_dict["pearsonr"]), float(result_dict["spearman"])))

    if args.is_evaluation == True:
        print("5-fold cross-validation")
        print("@ MSE:%.5f±%.5f, pearsonr:%.5f±%.5f, spearman:%.5f±%.5f" % ((np.array(results_dict["mse"]).mean()), np.std(np.array(results_dict["mse"])), (np.array(results_dict["pearsonr"]).mean()), np.std(np.array(results_dict["pearsonr"])), (np.array(results_dict["spearman"]).mean()), np.std(np.array(results_dict["spearman"]))))

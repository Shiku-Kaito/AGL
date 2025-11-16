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
        if args.module=="threshIndexOptimize_2stageLearning"or args.module=="AMAL":
            model = model["st_estimater"]
            model.load_state_dict(torch.load(("%s/model/fold=%d_seed=%d-best_st_estimater.pkl") % (args.output_path, args.fold, args.seed) ,map_location=args.device))
        elif args.module == "Uncertainty":
            loss_function.load_state_dict(torch.load(("%s/model/fold=%d_seed=%d-best_model.pkl") % (args.output_path, args.fold, args.seed) ,map_location=args.device))
        else: 
            model.load_state_dict(torch.load(("%s/model/fold=%d_seed=%d-best_model.pkl") % (args.output_path, args.fold, args.seed) ,map_location=args.device))

        if args.module == "Uncertainty":
            result_dict = eval_net(args, loss_function, test_loader, metric_func, gene_name_dict)  
        else:
            result_dict = eval_net(args, model, test_loader, metric_func, gene_name_dict)   
             
    return result_dict

if __name__ == '__main__':
    results_dict = {"mse":[], "pearsonr":[], "spearman":[], "pearsonr_eval_all_gene": [], "pearsonr_eval_all_gene_name":[], "k":[]}
    for fold in range(5):
    #    for seed in range(1):
        parser = argparse.ArgumentParser()
        # Data selectiion
        parser.add_argument('--fold', default=fold,
                            type=int, help='fold number')
        parser.add_argument('--dataset', default='TENX152',
                            type=str, help='NCBI783 or TENX94')
        parser.add_argument('--img_or_token_feat_or_feat', #書き換え
                            default="img", type=str, help="img or feat")
        parser.add_argument('--eval_highly_variable_top_genes', default=50,
                            type=int, help='fold number')
        parser.add_argument('--data_type', #書き換え
                            default="5-fold", type=str, help="5-fold_in_test_balanced or 5-fold_in_test_balanced_time_order")
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
        parser.add_argument('--is_test', default=1,
                            type=int, help='1 or 0')           
        parser.add_argument('--is_evaluation', default=0,
                            type=int, help='1 or 0')         
        # loss Selection
        parser.add_argument('--loss',default='pcc', 
                            type=str, help="mse or pcc")               
        # Module Selection
        parser.add_argument('--module',default='AGL+DkGSB', 
                            type=str, help="resnet_ST_net or Model2Step2Learning or Uncertainty or auxlearn or AMAL")
        parser.add_argument('--mode',default='',    # don't write!
                            type=str, help="")                        
        # Save Path
        parser.add_argument('--output_path',
                            default='./result/', type=str, help="output file name")
        # input path
        parser.add_argument('--input_path',
                            default='/media/user/HD-QHAU3/ST_estimation_data/', type=str, help="output file name")
        
        ############# train highly_variable top_genes  ###################
        parser.add_argument('--gene_select_type', default='top', type=str, help='random or top')
        parser.add_argument('--train_highly_variable_top_genes', default=0, type=int, help='fold number')

        ############# analysis setup ###################
        parser.add_argument('--ana_exp_eval_one_gene', # analysis exp: main gene num=1
                            default=1, type=int, help="")
        parser.add_argument('--toy_gene', # 
                            default=0, type=int, help="")

        ############## task weight estimater ####################################
        parser.add_argument('--beta', default=3e-3,
                            type=float, help='learning rate')   
        parser.add_argument('--bilevel_sampling_num', default=64, type=int, help='') 
        parser.add_argument('--iter_meta_val_update', default=1, type=int, help='') 
        parser.add_argument('--is_task_sparse_loss', default=1, type=int, help='') 
        parser.add_argument('--task_sparse_loss_weight', default=0.000001, type=float, help='') 
        parser.add_argument('--best_highlyvariable_geneNum', default=0, type=int, help='random or top')
        parser.add_argument('--reg_r', default=1.0, type=float, help='random or top')
        parser.add_argument('--tau', default=0.01, type=float, help='random or top')
        parser.add_argument('--dropout_rate', default=0.2, type=float, help='random or top')
        parser.add_argument('--meta_pacience', default=50, type=int, help='random or top')
        parser.add_argument('--H', default=2, type=int, help='random or top')

        args = parser.parse_args()

        if args.train_highly_variable_top_genes == 0:
            args.train_highly_variable_top_genes = None

        if args.is_evaluation == False:
            main(args)

        else:
            result_dict = main(args)
            results_dict["mse"].append(result_dict["mse"]), results_dict["pearsonr"].append(result_dict["pearsonr"]), results_dict["spearman"].append(result_dict["spearman"])
            results_dict["pearsonr_eval_all_gene"].append(result_dict["pearsonr_eval_all_gene"]), results_dict["pearsonr_eval_all_gene_name"].append(result_dict["pearsonr_eval_all_gene_name"])
            print("############ fold=%d ###########" % args.fold)
            print("@ MSE:%.5f, pearsonr:%.5f, spearman:%.5f" % (float(result_dict["mse"]), float(result_dict["pearsonr"]), float(result_dict["spearman"])))


    
    if args.is_evaluation == True:
        print("5-fold cross-validation")
        print("@ MSE:%.5f±%.5f, pearsonr:%.5f±%.5f, spearman:%.5f±%.5f" % ((np.array(results_dict["mse"]).mean()), np.std(np.array(results_dict["mse"])), (np.array(results_dict["pearsonr"]).mean()), np.std(np.array(results_dict["pearsonr"])), (np.array(results_dict["spearman"]).mean()), np.std(np.array(results_dict["spearman"]))))

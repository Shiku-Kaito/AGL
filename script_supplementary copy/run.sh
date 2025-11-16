# run ``AGL+DkGSB''
python ./script/main.py --module "resnet_ST_net" --dataset 'TENX152'  --best_highlyvariable_geneNum 0 --train_highly_variable_top_genes 0  --lr 0.00003 --temper 0.01 --weight_estimater_lr 0.003 --is_evaluation 0 --device "cuda:0"
python ./script/main.py --module "threshIndexOptimize_2stageLearning" --dataset 'TENX152'  --best_highlyvariable_geneNum 0 --train_highly_variable_top_genes 0  --lr 0.00003 --temper 0.01 --weight_estimater_lr 0.003 --is_evaluation 0 --device "cuda:0"


Please follow the steps below when conducting experiments using our code.

■ Requirement
    To set up their environment, please run:
    (we recommend to use Anaconda for installation.)
    $ conda env create -n hest -f hest.yml
    $ conda activate hest

■ Download dataset
    Please Hest-1k dataset here
    https://github.com/mahmoodlab/HEST

■ Training & Test for Prior Knowledge-Based Differentiable Top-k Gene Selection via Bi-level Optimization
    You can run Selective Prior Knowledge-Based Differentiable Top-k Gene Selection via Bi-level Optimization (DkGSB) code.
    If you want to train DkGSB, please run following command. 5 fold training is automatically done in our code.
    $ bash ./script_supplementary/run.sh

    If you want to evaluate DkGSB, please run following command. 5 fold trainevaluation is automatically done in our code.
    $ python ./script/main.py --module "threshIndexOptimize_2stageLearning" --dataset 'TENX152'  --best_highlyvariable_geneNum 0 --train_highly_variable_top_genes 0  --lr 0.00003 --temper 0.01 --weight_estimater_lr 0.003 --is_evaluation 1 --device "cuda:0"


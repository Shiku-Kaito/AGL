# Auxiliary Gene Learning 🧬
"Auxiliary Gene Learning: Spatial Gene Expression Estimation by Auxiliary Gene Selection", in **AAAI2026** 🎉🎉!

Shikui Kaito, Kazuya Nishimura, Shinnosuke Matsuo, Yasuhiro Kojima, Ryoma Bise

![Alt Text](./Overview_AGL.png)



## 📑Abstract
*Developing neural network models to estimate spatial gene expression from pathological images is important for overcoming the high observational costs associated with spatial gene expression data.
In prior studies, only a small subset of highly variable genes has been used for expression estimation, despite tens of thousands of genes being observed, in order to enable evaluation that mitigates the impact of observational noise. Genes outside this subset have been excluded from the training process as well. However, since there are likely co-expression relationships between genes, low-expression genes may still contribute to the estimation of the evaluation target.
In this paper, we propose **Auxiliary Gene Learning** (AGL) that utilizes the benefit of the ignored genes by reformulating their expression estimation as auxiliary tasks and training them jointly with the primary tasks. To effectively leverage auxiliary genes, we must select a subset of auxiliary genes that positively influence the prediction of the evaluation genes.
However, this is a challenging optimization problem due to the vast number of possible combinations.
To overcome this challenge, we propose **Prior-Knowledge-Based Differentiable Top-$k$ Gene Selection via Bi-level Optimization** (DkGSB), a method that ranks genes by leveraging prior knowledge and relaxes the combinatorial selection problem into a differentiable top-$k$ selection problem.
The experiments demonstrate the effectiveness of incorporating auxiliary genes into the learning process and show that the proposed method outperforms conventional auxiliary task learning approaches.
*

## ⬇️ Installation
To set up their environment, please run:  
(we recommend to use [Anaconda](https://www.anaconda.com/) for installation.)
```
conda env create -n hest -f hest.yml
conda activate hest
```

## 🌐 Data Preparation
You can download and preprocess the required datasets from Hest-1K using the following commands.
```
python ./script/preprocess/load_hest1k.py
python ./script/preprocess/preprocess_hest1k.py
```

Split the data into 5-fold cross-validation.
```
python ./script/preprocess/split_to_5-fold_cv.py
```

## 🚀 Training the DkGSB
You can train our proposed **Prior-Knowledge-Based Differentiable Top-$k$ Gene Selection via Bi-level Optimization** (DkGSB) using the following command.
```
python ./script/main.py --module "AGL+DkGSB" --dataset 'TENX152'
```

## 🔥 Training the comparative methods
If you would like to run the comparative experiments, please execute the following commands:

1. **AGL** — Training with **all auxiliary genes**
```
python ./script/main.py --module "AGL" --dataset 'TENX152'
```
2. **Primary-only** — Training with **only the primary genes**
```
python ./script/main.py --module "PGL" --dataset 'TENX152'
```

## 📊 Evaluation of Each Method
You can evaluate the following three approaches:

1. **DkGSB (Prior-Knowledge-Based Differentiable Top-$k$ Gene Selection via Bi-level Optimization)**  
```
python ./script/main.py --module "AGL+DkGSB" --dataset 'TENX152' --is_evaluation 1
```

   
2. **AGL (All Auxiliary Genes)**  
```
python ./script/main.py --module "AGL" --dataset 'TENX152' --is_evaluation 1
```

3. **Primary-only**  
```
python ./script/main.py --module "PGL" --dataset 'TENX152' --is_evaluation 1
```


## 🔍 Citation
If you find this repository helpful, please consider citing:
```
Comming soon !
```

# ✏️ Author
@ Shiku Kaito  
・ Contact: kaito.shiku@human.ait.kyushu-u.ac.jp

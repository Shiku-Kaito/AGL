import pandas as pd
from huggingface_hub import login
import matplotlib.pyplot as plt
import scanpy as sc
from hest import iter_hest
import h5py
import anndata as ad
import numpy as np
import argparse

def main(args):
    for gene_num in args.gene_num:
        slide_count = 0
        for st in iter_hest("./data/org/hest_data/", id_list=args.slide_ids):
            slide_id = args.slide_ids[slide_count]
            # ST (adata):
            adata = st.adata
            # mitochondrial genes, "MT-" for human, "Mt-" for mouse
            adata.var["mt"] = adata.var_names.str.startswith("MT-")
            # ribosomal genes
            adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
            # hemoglobin genes
            adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
            
            sc.pp.filter_cells(adata, min_genes=200) 
            sc.pp.filter_genes(adata, min_cells=3) 

            adata.layers["counts"] = adata.X.copy()
            adata.X = adata.X.astype("float32")
            sc.pp.log1p(adata)  

            if gene_num==50:
                sc.pp.highly_variable_genes(adata, n_top_genes=agene_num)            
                top_genes_var = adata[:, adata.var['highly_variable']]
            elif gene_num=="all":
                top_genes_var = adata

            ############# matching gene and patch
            top_genes_var_oneslide = top_genes_var
            print("top gene name:", top_genes_var.var.index)
            
            with h5py.File("./data/org/hest_data/patches/%s.h5" % slide_id, "r") as h5file:
                print(list(h5file.keys()))
                patch_list = h5file["img"]
                patch_list = patch_list[:]

                coords = h5file["coords"]
                coords = coords[:]

                barcode = h5file["barcode"]
                barcode = barcode[:]
                
            code_list = []
            for bar in barcode:
                code_list.append("%s-%s" % (bar[0].decode("utf-8"), slide_id) )

            top_genes_var_oneslide.obs["patch_index"] =  np.nan
            for code in top_genes_var_oneslide.obs_names:
                try:
                    idx = code_list.index("%s-%s" % (code, slide_id_list[slide_count]))
                    top_genes_var_oneslide.obs.loc[code, "patch_index"] = int(idx)
                except:
                    pass

            matched_top_genes_var_oneslide = top_genes_var_oneslide[top_genes_var_oneslide.obs["patch_index"].notnull()]
            if gene_num==50:
                matched_top_genes_var_oneslide.write('./data/preprocessed_data/hest_1slide/%s_top50_matching_patch_adata.h5ad' % (slide_id))
            elif gene_num=="all":
                matched_top_genes_var_oneslide.write('./data/preprocessed_data/hest_1slide/%s_topall_matching_patch_adata.h5ad' % slide_id)
                
            slide_count+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_ids', default=["TENX65", "TENX89", "TENX152"],
                        type=int, help='fold number')
    parser.add_argument('--gene_num', default=[50, "all"],
                        type=int, help='fold number')
    args = parser.parse_args()  
    
    os.makedirs("./data/preprocessed_data/hest_1slide/", exist_ok=True)
    main(args)

import scanpy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
import argparse

def main(args):
    # Filter for directories matching the specified pattern
    filter_pattern = 'ABN*KQNG*'
    abn_directories = [d for d in glob.glob(filter_pattern) if os.path.isdir(d)]
    print(f"Found directories to process: {len(abn_directories)} directories.")
    print(f"Directory filter pattern: '{filter_pattern}'")

    # --- Pass 1: Aggregate all raw gene expression data and directory identities ---
    print("\n--- Pass 1: Aggregating raw gene expression data across directories ---")
    
    all_raw_data = []
    all_directory_labels = []

    for directory in abn_directories:
        h5ad_file = os.path.join(directory, 'processed_data_qc_only.h5ad')
        if not os.path.exists(h5ad_file):
            print(f"Warning: File not found for directory {directory}. Skipping.")
            continue
            
        adata = sc.read_h5ad(h5ad_file)
        
        if adata.raw is None:
            print(f"Warning: No .raw attribute found for directory {directory}. Skipping.")
            continue
            
        raw_expression = adata.raw.X
        all_raw_data.append(raw_expression)
        all_directory_labels.extend([directory] * raw_expression.shape[0])

    if not all_raw_data:
        print("No raw gene expression data found across the filtered directories. Exiting.")
        return

    # --- Pass 2: Compute and plot a global UMAP for the raw gene expression ---
    print("\n--- Pass 2: Computing and plotting global UMAP ---")

    # Combine all data into one AnnData object
    combined_data = np.vstack(all_raw_data)
    
    print(f"Total number of cells being processed: {combined_data.shape[0]}")
    
    adata_global = sc.AnnData(combined_data)
    adata_global.obs['directory'] = pd.Categorical(all_directory_labels)

    # Compute UMAP
    print("Computing neighbors and UMAP...")
    sc.pp.neighbors(adata_global, n_neighbors=15, use_rep='X')
    sc.tl.umap(adata_global)
    print("UMAP computation complete.")

    # --- Generate and Save Plot ---
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sc.pl.umap(
        adata_global,
        color='directory',
        ax=ax,
        show=False,
        title='Global UMAP of Raw Gene Expression',
        s=5 # smaller dot size for clarity
    )
    
    plt.tight_layout()

    plot_path = os.path.join('.', 'global_umap_raw_gene_expression.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved global UMAP plot to: {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a global UMAP of raw gene expression from select datasets.")
    args = parser.parse_args()
    main(args) 
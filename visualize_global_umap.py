import scanpy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
import argparse

def main(args):
    abn_directories = [d for d in glob.glob('ABN*KQNG*') if os.path.isdir(d)]
    print(f"Found directories to process: {len(abn_directories)} directories.")
    print(f"Directory filter pattern: 'ABN*area18*'")

    # --- Pass 1: Aggregate all latent data and directory identities ---
    print("\n--- Pass 1: Aggregating all latent data across directories ---")
    
    all_latent_data = {} # Key: latent_key, Value: list of np.arrays
    all_directory_labels = {} # Key: latent_key, Value: list of directory names

    for directory in abn_directories:
        h5ad_file = os.path.join(directory, args.input_file)
        if not os.path.exists(h5ad_file):
            continue
            
        adata = sc.read_h5ad(h5ad_file)
        for key in adata.obsm.keys():
            if 'latent' in key:
                if key not in all_latent_data:
                    all_latent_data[key] = []
                    all_directory_labels[key] = []
                
                latent_vars = adata.obsm[key]
                all_latent_data[key].append(latent_vars)
                # Create a label for each cell, indicating its source directory
                all_directory_labels[key].extend([directory] * latent_vars.shape[0])

    if not all_latent_data:
        print("No latent data found across all directories. Exiting.")
        return

    # --- Pass 2: Compute and plot a global UMAP for each latent key ---
    print("\n--- Pass 2: Computing and plotting global UMAPs ---")

    for latent_key, data_list in all_latent_data.items():
        print(f"\nProcessing global UMAP for: {latent_key}")
        
        # Combine all data for this latent key into one AnnData object
        combined_data = np.vstack(data_list)
        directory_labels = all_directory_labels[latent_key]
        
        print(f"Total number of cells being processed: {combined_data.shape[0]}")
        
        adata_global = sc.AnnData(combined_data)
        adata_global.obs['directory'] = pd.Categorical(directory_labels)

        # Compute UMAP
        print("Computing neighbors and UMAP...")
        sc.pp.neighbors(adata_global, n_neighbors=15, use_rep='X')
        sc.tl.umap(adata_global)
        print("UMAP computation complete.")

        # --- Generate and Save Plot ---
        output_suffix = latent_key.replace('X_', '').replace('_latents', '')
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sc.pl.umap(
            adata_global,
            color='directory',
            ax=ax,
            show=False,
            title=f'Global UMAP of {output_suffix}',
            s=5 # smaller dot size for clarity
        )
        
        plt.tight_layout()

        plot_path = os.path.join('.', f'global_umap_{output_suffix}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved global UMAP plot to: {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a global UMAP of latent variables from all datasets.")
    parser.add_argument(
        '--input_file',
        type=str,
        default='processed_data_latents.h5ad',
        help="Name of the H5AD file to use from each directory (e.g., 'processed_data_inference_latents.h5ad')."
    )
    args = parser.parse_args()
    main(args) 
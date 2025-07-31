import pandas as pd
import scanpy as sc
import os
import glob
import argparse

def main(args):
    # Find all directories starting with 'ABN' that are actual directories
    abn_directories = [d for d in glob.glob('ABN*') if os.path.isdir(d)]
    print(f"Found directories to process: {abn_directories}")

    for directory in abn_directories:
        print(f"\n--- Processing Directory: {directory} ---\n")

        # --- 1. Configuration ---
        CSV_PATH = os.path.join(directory, 'merged_metadata_and_partitions.csv')
        
        if args.compute_umap:
            OUTPUT_H5AD_PATH = os.path.join(directory, 'processed_data_umap.h5ad')
        else:
            OUTPUT_H5AD_PATH = os.path.join(directory, 'processed_data_qc_only.h5ad')

        if not os.path.exists(CSV_PATH):
            print(f"Skipping {directory}, 'merged_metadata_and_partitions.csv' not found.")
            continue

        # --- 2. Load Data ---
        print(f"Loading data from {CSV_PATH}...")
        raw_data = pd.read_csv(CSV_PATH, index_col=0)
        raw_data.index = raw_data.index.map(str)
        print("Converted cell index to string type.")

        # --- 3. Replicate R Script's Quality Control ---
        print("Performing quality control...")
        data_transposed = raw_data.T
        gene_rows = data_transposed.index[0:471]
        total_rna_per_cell = data_transposed.loc[gene_rows].sum(axis=0)
        rna_qc_mask = total_rna_per_cell > 10
        volume_feature_name = data_transposed.index[551]
        print(f"Using feature '{volume_feature_name}' for volume QC.")
        volume_per_cell = data_transposed.loc[volume_feature_name]
        volume_qc_mask = volume_per_cell > 50
        passing_qc_cells_mask = rna_qc_mask & volume_qc_mask
        data_qc = raw_data[passing_qc_cells_mask]
        print(f"QC complete. Kept {data_qc.shape[0]} out of {raw_data.shape[0]} cells.")

        # --- 4. Prepare Data for Scanpy ---
        gene_columns = raw_data.columns[0:471]
        adata = sc.AnnData(data_qc[gene_columns].copy())
        metadata_columns = raw_data.columns[471:]
        adata.obs = data_qc[metadata_columns].copy()
        x_coord_col_name = raw_data.columns[552]
        y_coord_col_name = raw_data.columns[553]
        adata.uns['spatial_coords'] = {'x_col': x_coord_col_name, 'y_col': y_coord_col_name}
        print(f"Stored coordinate column names in adata.uns['spatial_coords']")

        # --- 5. Log-Normalize Data ---
        print("Log-normalizing data...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Store the log-normalized data with all genes in the .raw attribute
        # This is critical for the GAT script
        adata.raw = adata

        if args.compute_umap:
            # --- 6a. Compute UMAP on Raw Log-Expression with Cosine Metric (No PCA) ---
            print("Computing UMAP on raw log-expression for all genes (cosine metric)...")
            sc.pp.neighbors(adata, n_neighbors=15, use_rep='X', metric='cosine', key_added='neighbors_raw')
            sc.tl.umap(adata, neighbors_key='neighbors_raw')
            adata.obsm['X_umap_raw'] = adata.obsm['X_umap'].copy()
            print("Raw UMAP computed and stored in adata.obsm['X_umap_raw'].")

            # --- 6b. Compute Standard UMAP (with PCA) ---
            print("Computing standard UMAP (highly variable genes, PCA)...")
            sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            
            # Filter for highly variable genes for the PCA-based UMAP.
            # Important: this filtering happens *after* the full data was stored in adata.raw
            adata_hvg = adata[:, adata.var.highly_variable].copy()
            sc.pp.scale(adata_hvg, max_value=10)
            sc.tl.pca(adata_hvg, svd_solver='arpack')
            sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=30)
            sc.tl.umap(adata_hvg)
            adata.obsm['X_umap_pca'] = adata_hvg.obsm['X_umap'].copy()
            print("Standard UMAP computed and stored in adata.obsm['X_umap_pca'].")
            
            # Clean up temporary keys
            del adata.obsm['X_umap']
            if 'umap' in adata.uns:
                del adata.uns['umap']
            if 'neighbors_raw' in adata.uns:
                del adata.uns['neighbors_raw']
        
        # --- 7. Save ---
        print(f"\nSaving processed data to {OUTPUT_H5AD_PATH}...")
        adata.write(OUTPUT_H5AD_PATH)
        print("Script finished successfully!")

        print(f"\nYou can now load the file '{OUTPUT_H5AD_PATH}' to explore the data.")
        print("For example:")
        print("import scanpy as sc")
        print(f"adata = sc.read_h5ad('{OUTPUT_H5AD_PATH}')")
        if args.compute_umap:
            print(f"sc.pl.embedding(adata, basis='X_umap_pca', color='{gene_columns[0]}')")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform QC and optionally UMAP on spatial data.")
    parser.add_argument(
        '--compute_umap',
        action='store_true',
        help="If set, compute and store UMAP embeddings in the output file."
    )
    args = parser.parse_args()
    main(args)
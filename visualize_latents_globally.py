import scanpy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import glob
import argparse

def main(args):
    abn_directories = [d for d in glob.glob('ABN*') if os.path.isdir(d)]
    print(f"Found directories to process: {len(abn_directories)} directories.")

    # --- Pass 1: Aggregate all latent data to find a global transformation ---
    print("\n--- Pass 1: Aggregating data for global PCA and color scaling ---")
    
    all_latent_data = {} # Key: latent_key, Value: list of np.arrays
    
    for directory in abn_directories:
        h5ad_file = os.path.join(directory, args.input_file)
        if not os.path.exists(h5ad_file):
            continue
            
        adata = sc.read_h5ad(h5ad_file)
        for key in adata.obsm.keys():
            if 'latent' in key:
                if key not in all_latent_data:
                    all_latent_data[key] = []
                all_latent_data[key].append(adata.obsm[key])

    if not all_latent_data:
        print("No latent data found across all directories. Exiting.")
        return

    # --- Compute Global Transformations ---
    print("\nComputing global PCA and color scales...")
    global_pcas = {}
    global_color_scales = {}

    for latent_key, data_list in all_latent_data.items():
        # Fit PCA on the combined data for this key
        combined_data = np.vstack(data_list)
        pca = PCA(n_components=3)
        pca.fit(combined_data)
        global_pcas[latent_key] = pca
        print(f"Global PCA for '{latent_key}': Explained variance = {pca.explained_variance_ratio_ * 100}")

        # Get global color scale by transforming all data and finding percentiles
        all_transformed_data = pca.transform(combined_data)
        scales = []
        for i in range(3):
            p_low, p_high = np.percentile(all_transformed_data[:, i], [2, 98])
            scales.append({'low': p_low, 'high': p_high})
        global_color_scales[latent_key] = scales

    # --- Pass 2: Generate plots using the global transformations ---
    print("\n--- Pass 2: Generating plots with global transformations ---")
    
    for directory in abn_directories:
        print(f"\n--- Plotting for Directory: {directory} ---")
        h5ad_file = os.path.join(directory, args.input_file)
        if not os.path.exists(h5ad_file):
            print(f"File not found: {h5ad_file}. Skipping.")
            continue
            
        adata = sc.read_h5ad(h5ad_file)
        
        for latent_key in (key for key in adata.obsm.keys() if 'latent' in key):
            if latent_key not in global_pcas:
                continue

            # Apply the global PCA and color scaling
            pca = global_pcas[latent_key]
            scales = global_color_scales[latent_key]
            
            latent_variables = adata.obsm[latent_key]
            latent_pca = pca.transform(latent_variables)
            
            rgb_colors = np.zeros_like(latent_pca)
            for i in range(3):
                channel_clipped = np.clip(latent_pca[:, i], scales[i]['low'], scales[i]['high'])
                rgb_colors[:, i] = minmax_scale(channel_clipped, feature_range=(0, 1))

            # Get coordinates for plotting
            x_col = adata.uns['spatial_coords']['x_col']
            y_col = adata.uns['spatial_coords']['y_col']
            coords = adata.obs[[x_col, y_col]].to_numpy()
            
            # --- Generate and Save Plot ---
            output_suffix = latent_key.replace('X_', '').replace('_latents', '')
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(coords[:, 0], coords[:, 1], s=2, c=rgb_colors, edgecolors='none')
            ax.set_title(f'Global PCA of {output_suffix} in {directory}')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()

            plot_path = os.path.join(directory, f'global_spatial_pca_rgb_{output_suffix}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='black')
            plt.close(fig)
            print(f"Saved globally consistent plot to: {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate globally consistent visualizations of latent variables.")
    parser.add_argument(
        '--input_file',
        type=str,
        default='processed_data_latents.h5ad',
        help="Name of the H5AD file to visualize (e.g., 'processed_data_inference_latents.h5ad')."
    )
    args = parser.parse_args()
    main(args) 
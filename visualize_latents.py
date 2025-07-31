import scanpy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import glob
import argparse

def generate_and_save_plot(h5ad_path, latent_key, output_suffix, output_dir):
    """
    Loads an AnnData object, computes PCA on specified latent variables,
    and saves a spatial RGB plot.
    """
    # --- 1. Load Data ---
    print(f"--- Processing: {output_suffix} ---")
    if not os.path.exists(h5ad_path):
        print(f"File not found: {h5ad_path}. Skipping.\n")
        return

    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded data from {h5ad_path}")

    # --- 2. Verify Data ---
    if latent_key not in adata.obsm:
        print(f"Error: Latent variables '{latent_key}' not found. Skipping.\n")
        return
    if 'spatial_coords' not in adata.uns:
        print("Error: Spatial coordinate information not found. Skipping.\n")
        return

    latent_variables = adata.obsm[latent_key]
    x_col = adata.uns['spatial_coords']['x_col']
    y_col = adata.uns['spatial_coords']['y_col']
    coords = adata.obs[[x_col, y_col]].to_numpy()

    # --- 3. Compute PCA ---
    print("Computing PCA on latent variables...")
    pca = PCA(n_components=3)
    latent_pca = pca.fit_transform(latent_variables)
    print(f"Explained variance (PC1-3): {pca.explained_variance_ratio_ * 100}")

    # --- 4. Create RGB Colors with Contrast Stretching ---
    print("Mapping PCA components to RGB colors with contrast stretching...")
    pc_data = latent_pca[:, :3]
    rgb_colors = np.zeros_like(pc_data)

    # Process each channel (R, G, B) independently for robust contrast stretching
    for i in range(3):
        channel_data = pc_data[:, i]
        # Clip the data to the 2nd and 98th percentiles to remove outliers
        p_low, p_high = np.percentile(channel_data, [2, 98])
        channel_clipped = np.clip(channel_data, p_low, p_high)
        # Scale the clipped data to the [0, 1] range to create the color channel
        rgb_colors[:, i] = minmax_scale(channel_clipped, feature_range=(0, 1))

    # --- 5. Generate and Save Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        coords[:, 0], coords[:, 1], s=2, c=rgb_colors, edgecolors='none'
    )
    ax.set_title(f'Spatial Map of Latent PCA ({output_suffix})')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()

    # Save the figure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_path = os.path.join(output_dir, f'spatial_latent_pca_rgb2_{output_suffix}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    
    print(f"Successfully saved RGB visualization to: {plot_path}\n")


def main(args):
    abn_directories = [d for d in glob.glob('ABN*') if os.path.isdir(d)]
    print(f"Found directories to process: {abn_directories}")

    for directory in abn_directories:
        print(f"\n--- Visualizing Directory: {directory} ---\n")
        
        h5ad_file_to_visualize = os.path.join(directory, args.input_file)
        
        if not os.path.exists(h5ad_file_to_visualize):
            print(f"File not found: {h5ad_file_to_visualize}. Skipping directory.\n")
            continue

        # Load the file once to inspect its keys
        adata = sc.read_h5ad(h5ad_file_to_visualize)
        
        # Find all keys in .obsm that appear to contain latent variables
        latent_keys_to_process = [key for key in adata.obsm.keys() if 'latent' in key]
        
        if not latent_keys_to_process:
            print(f"No latent variable keys found in {h5ad_file_to_visualize}. Skipping.\n")
            continue
            
        print(f"Found latent keys to visualize: {latent_keys_to_process}")

        # --- Run visualization for each found latent key ---
        for key in latent_keys_to_process:
            # Derive the output filename suffix from the latent key
            output_suffix = key.replace('X_', '').replace('_latents', '')
            
            generate_and_save_plot(
                h5ad_path=h5ad_file_to_visualize,
                latent_key=key,
                output_suffix=output_suffix,
                output_dir=directory
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automatically visualize all latent variables found in an H5AD file.")
    parser.add_argument(
        '--input_file',
        type=str,
        default='processed_data_latents.h5ad',
        help="Name of the H5AD file to visualize (e.g., 'processed_data_inference_latents.h5ad')."
    )
    args = parser.parse_args()
    main(args) 
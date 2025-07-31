import scanpy as sc
import squidpy as sq
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
import argparse
import glob

# --- 1. Hybrid GNN / Autoencoder Model Definition (Same as training script) ---

class HybridAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, latent_dim=32, model_type='GAT', heads=2):
        super().__init__()
        self.model_type = model_type.upper()

        if self.model_type == 'GAT':
            # --- Graph-based (GAT) layers ---
            self.encoder_conv1 = GATConv(in_channels, in_channels * 2, heads=heads)
            self.encoder_conv2 = GATConv(in_channels * 2 * heads, latent_dim, heads=1)
            self.decoder_conv1 = GATConv(latent_dim, in_channels * 2, heads=heads)
            self.decoder_conv2 = GATConv(in_channels * 2 * heads, in_channels, heads=1)
        elif self.model_type == 'GCN':
            # --- Graph-based (GCN) layers ---
            self.encoder_conv1 = GCNConv(in_channels, in_channels * 2)
            self.encoder_conv2 = GCNConv(in_channels * 2, latent_dim)
            self.decoder_conv1 = GCNConv(latent_dim, in_channels * 2)
            self.decoder_conv2 = GCNConv(in_channels * 2, in_channels)
        else:
            raise ValueError("Unsupported model_type. Choose 'GAT' or 'GCN'.")

        # --- Standard (Linear) layers for non-spatial autoencoder ---
        self.encoder_lin1 = torch.nn.Linear(in_channels, in_channels * 2)
        self.encoder_lin2 = torch.nn.Linear(in_channels * 2, latent_dim)
        self.decoder_lin1 = torch.nn.Linear(latent_dim, in_channels * 2)
        self.decoder_lin2 = torch.nn.Linear(in_channels * 2, in_channels)

    def encode(self, x, edge_index, use_neighbors):
        if use_neighbors:
            x = F.relu(self.encoder_conv1(x, edge_index))
            return self.encoder_conv2(x, edge_index)
        else:
            x = F.relu(self.encoder_lin1(x))
            return self.encoder_lin2(x)

    def decode(self, z, edge_index, use_neighbors):
        if use_neighbors:
            z = F.relu(self.decoder_conv1(z, edge_index))
            return self.decoder_conv2(z, edge_index)
        else:
            z = F.relu(self.decoder_lin1(z))
            return self.decoder_lin2(z)

def main(args):
    # --- Find all ABN directories ---
    abn_directories = [d for d in glob.glob('ABN*') if os.path.isdir(d)]
    if not abn_directories:
        print("No 'ABN*' directories found to process.")
        return
    print(f"Found directories to process: {abn_directories}")

    # --- 1. Model Initialization ---
    print(f"Initializing a {args.model_type} model for inference...")
    
    # We still need to know the number of input channels.
    first_dir_h5ad = os.path.join(abn_directories[0], 'processed_data_qc_only.h5ad')
    if not os.path.exists(first_dir_h5ad):
        print(f"Error: Cannot initialize model. 'processed_data_qc_only.h5ad' not found in first directory: {abn_directories[0]}")
        return
        
    temp_adata = sc.read_h5ad(first_dir_h5ad)
    in_channels = temp_adata.raw.n_vars
    del temp_adata

    LATENT_DIM = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HybridAutoencoder(in_channels, latent_dim=LATENT_DIM, model_type=args.model_type, heads=2).to(device)
    
    # --- Load the Pre-trained Model Weights ---
    if not os.path.exists(args.load_model_path):
        print(f"Error: Model path '{args.load_model_path}' not found. Cannot run inference.")
        return
    model.load_state_dict(torch.load(args.load_model_path, map_location=device))
    model.eval() # Set the model to evaluation mode
    print(f"Successfully loaded pre-trained model from: {args.load_model_path}")
    print(f"Model is on device: {device} with {in_channels} input features.")

    # --- 2. Loop Through Directories to Run Inference ---
    K_NEIGHBORS = 10

    for directory in abn_directories:
        print(f"\n--- Running Inference on Directory: {directory} ---")
        
        H5AD_PATH = os.path.join(directory, 'processed_data_qc_only.h5ad')
        if not os.path.exists(H5AD_PATH):
            print(f"Skipping {directory}, 'processed_data_qc_only.h5ad' not found.")
            continue

        print(f"Loading data from: {H5AD_PATH}")
        adata = sc.read_h5ad(H5AD_PATH)
        adata_spatial = adata.raw.to_adata()

        x_col = adata.uns['spatial_coords']['x_col']
        y_col = adata.uns['spatial_coords']['y_col']
        adata_spatial.obsm['spatial'] = adata.obs[[x_col, y_col]].to_numpy()
        
        sq.gr.spatial_neighbors(adata_spatial, coord_type="generic", n_neighs=K_NEIGHBORS)
        
        features = torch.tensor(adata_spatial.X, dtype=torch.float32)
        adj_matrix = adata_spatial.obsp['spatial_connectivities']
        edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
        graph_data = Data(x=features, edge_index=edge_index).to(device)

        # --- Run Inference and Extract Latent Variables ---
        print("Extracting latent variables...")
        with torch.no_grad():
            latent_variables = model.encode(graph_data.x, graph_data.edge_index, use_neighbors=args.use_neighbors)
        
        # Define a new key for inference results to avoid overwriting trained ones
        latent_key = f'X_{args.model_type.lower()}_inference_latents'
        
        output_path = os.path.join(directory, 'processed_data_inference_latents.h5ad')
            
        adata.obsm[latent_key] = latent_variables.cpu().numpy()
        adata.write(output_path)
        print(f"Saved inference latent variables for {directory} to '{output_path}'.")
        del adata

    print("\n--- All inference complete. ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with a pre-trained GNN model.")
    parser.add_argument(
        '--load_model_path',
        type=str,
        required=True,
        help="Path to the pre-trained model file (.pt) to use for inference."
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='GAT',
        choices=['GAT', 'GCN'],
        help="Type of GNN model architecture to use. Must match the loaded model. (default: GAT)"
    )
    parser.add_argument(
        '--use_neighbors',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="Whether to use neighbor information (GNN) or not (Autoencoder). (default: True)"
    )
    args = parser.parse_args()
    main(args) 
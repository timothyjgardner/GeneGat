import scanpy as sc
import squidpy as sq
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # Switched from GCNConv to GATConv
from torch_geometric.data import Data
import argparse
import glob

# --- 1. Hybrid GAT / Autoencoder Model Definition ---

class HybridAttentionAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, latent_dim=32, heads=2):
        super().__init__()
        
        # --- Graph-based (GAT) layers ---
        # The key difference is the use of GATConv and multi-head attention.
        # The output dimension of a multi-head GAT layer is heads * out_channels.
        self.encoder_conv1 = GATConv(in_channels, in_channels * 2, heads=heads)
        self.encoder_conv2 = GATConv(in_channels * 2 * heads, latent_dim, heads=1) # Final layer has 1 head
        
        self.decoder_conv1 = GATConv(latent_dim, in_channels * 2, heads=heads)
        self.decoder_conv2 = GATConv(in_channels * 2 * heads, in_channels, heads=1)
        
        # --- Standard (Linear) layers (identical to the GCN script) ---
        self.encoder_lin1 = torch.nn.Linear(in_channels, in_channels * 2)
        self.encoder_lin2 = torch.nn.Linear(in_channels * 2, latent_dim)
        self.decoder_lin1 = torch.nn.Linear(latent_dim, in_channels * 2)
        self.decoder_lin2 = torch.nn.Linear(in_channels * 2, in_channels)

    def encode(self, x, edge_index, use_neighbors):
        if use_neighbors:
            # GAT path: uses attention to weigh neighbors
            x = F.relu(self.encoder_conv1(x, edge_index))
            return self.encoder_conv2(x, edge_index)
        else:
            # Standard autoencoder path
            x = F.relu(self.encoder_lin1(x))
            return self.encoder_lin2(x)

    def decode(self, z, edge_index, use_neighbors):
        if use_neighbors:
            # GAT path
            z = F.relu(self.decoder_conv1(z, edge_index))
            return self.decoder_conv2(z, edge_index)
        else:
            # Standard autoencoder path
            z = F.relu(self.decoder_lin1(z))
            return self.decoder_lin2(z)

    def forward(self, x, edge_index, use_neighbors):
        z = self.encode(x, edge_index, use_neighbors)
        return self.decode(z, edge_index, use_neighbors)


def main(args):
    # --- Find all ABN directories ---
    abn_directories = [d for d in glob.glob('ABN*synthetic*') if os.path.isdir(d)]
    if not abn_directories:
        print("No 'ABN*' directories found to process.")
        return
    print(f"Found directories to process: {abn_directories}")

    # --- 1. One-time Model and Optimizer Initialization ---
    print("Initializing a single global model...")
    
    # Initialize the model using the first directory's qc_only file
    first_dir_h5ad = os.path.join(abn_directories[0], 'processed_data_qc_only.h5ad')
    if not os.path.exists(first_dir_h5ad):
        print(f"Error: Cannot initialize model. 'processed_data_qc_only.h5ad' not found in first directory: {abn_directories[0]}")
        return
        
    temp_adata = sc.read_h5ad(first_dir_h5ad)
    in_channels = temp_adata.raw.n_vars
    del temp_adata

    LATENT_DIM = 64
    LEARNING_RATE = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HybridAttentionAutoencoder(in_channels, latent_dim=LATENT_DIM, heads=2).to(device)
    
    # --- Load existing model parameters if a path is provided ---
    if args.load_model_path:
        if os.path.exists(args.load_model_path):
            model.load_state_dict(torch.load(args.load_model_path, map_location=device))
            print(f"Successfully loaded model parameters from: {args.load_model_path}")
        else:
            print(f"Warning: Model path '{args.load_model_path}' not found. Starting with a new model.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Model initialized on device: {device} with {in_channels} input features.")

    # --- 2. Loop Through Directories to Train the Model ---
    N_EPOCHS = 2
    K_NEIGHBORS = 10
    MASK_FRACTION = 0.2

    for directory in abn_directories:
        print(f"\n--- Continuously Training on Directory: {directory} ---")
        
        # Look for the specific qc_only file
        H5AD_PATH = os.path.join(directory, 'processed_data_qc_only.h5ad')
        if not os.path.exists(H5AD_PATH):
            print(f"Skipping {directory}, 'processed_data_qc_only.h5ad' not found.")
            continue

        # Load data for the current directory
        print(f"Loading data from: {H5AD_PATH}")
        adata = sc.read_h5ad(H5AD_PATH)
        # Use the .raw attribute which should always exist
        adata_spatial = adata.raw.to_adata()

        x_col = adata.uns['spatial_coords']['x_col']
        y_col = adata.uns['spatial_coords']['y_col']
        adata_spatial.obsm['spatial'] = adata.obs[[x_col, y_col]].to_numpy()
        
        sq.gr.spatial_neighbors(adata_spatial, coord_type="generic", n_neighs=K_NEIGHBORS)
        
        features = torch.tensor(adata_spatial.X, dtype=torch.float32)
        adj_matrix = adata_spatial.obsp['spatial_connectivities']
        edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
        graph_data = Data(x=features, edge_index=edge_index).to(device)

        # Train the *existing* model on the new data
        print(f"Starting training for {directory}...")
        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            
            # More memory-efficient masking
            mask = torch.rand(graph_data.x.shape) < MASK_FRACTION
            
            # Store original values and apply mask
            original_values = graph_data.x[mask].clone()
            graph_data.x[mask] = 0
            
            # Encode, then decode
            z = model.encode(graph_data.x, graph_data.edge_index, use_neighbors=args.use_neighbors)
            reconstructed_data = model.decode(z, graph_data.edge_index, use_neighbors=args.use_neighbors)
            
            # 1. Reconstruction Loss
            reconstruction_loss = F.mse_loss(reconstructed_data[mask], original_values)

            # 2. Decorrelation Loss (only if enabled and using neighbors)
            decorrelation_loss = 0.0
            if args.decorrelation_strength > 0 and args.use_neighbors:
                source_nodes, target_nodes = graph_data.edge_index
                z_source = z[source_nodes]
                z_target = z[target_nodes]
                
                # We penalize the mean cosine similarity between neighbors
                similarities = F.cosine_similarity(z_source, z_target, dim=1, eps=1e-8)
                decorrelation_loss = torch.mean(similarities)

            # Combine losses
            loss = reconstruction_loss + (args.decorrelation_strength * decorrelation_loss)
            
            loss.backward()
            optimizer.step()

            # Restore original values after step
            graph_data.x[mask] = original_values
            
            if epoch % 1 == 0:
                print(f'Epoch: {epoch:02d}, Loss: {loss.item():.4f}, Recon Loss: {reconstruction_loss.item():.4f}, Decorr Loss: {decorrelation_loss if isinstance(decorrelation_loss, float) else decorrelation_loss.item():.4f}')

        # Extract and save latent variables for THIS directory
        print("Extracting latent variables for this directory...")
        model.eval()
        with torch.no_grad():
            latent_variables = model.encode(graph_data.x, graph_data.edge_index, use_neighbors=args.use_neighbors)
        
        if args.use_neighbors:
            latent_key = 'X_gat_latent_neighbors'
        else:
            latent_key = 'X_gat_latent_no_neighbors'
        
        output_path = os.path.join(directory, 'processed_data_latents.h5ad')
            
        adata.obsm[latent_key] = latent_variables.cpu().numpy()
        adata.write(output_path)
        print(f"Saved latent variables for {directory} to '{output_path}'.")
        del adata

    # --- 3. Save the Final, Globally Trained Model ---
    print("\n--- All training complete. Saving final global model. ---")
    if args.use_neighbors:
        model_path = 'global_gat_model_with_neighbors.pt'
    else:
        model_path = 'global_gat_model_no_neighbors.pt'
        
    torch.save(model.state_dict(), model_path)
    print(f"Saved globally trained GAT model to '{model_path}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-supervised GAT for spatial transcriptomics.")
    parser.add_argument(
        '--use_neighbors',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="Whether to use neighbor information (GAT) or not (Autoencoder). (default: True)"
    )
    parser.add_argument(
        '--load_model_path',
        type=str,
        default=None,
        help="Path to a pre-trained model file (.pt) to continue training from."
    )
    parser.add_argument(
        '--decorrelation_strength',
        type=float,
        default=0.0,
        help="Strength of the decorrelation term (lambda). Set > 0 to enable. (default: 0.0)"
    )
    args = parser.parse_args()
    main(args) 
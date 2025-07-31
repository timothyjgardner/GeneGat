import anndata
import numpy as np
import pandas as pd
import os

def create_synthetic_data(output_path, template_file_path):
    """
    Creates a synthetic anndata file with random cell locations and gene expression values
    that statistically match the template data.

    This function uses an existing anndata file as a template to maintain the same
    number of cells and genes. The gene expression data is generated from a log-normal
    distribution to better mimic real gene expression data, and sparsity is introduced
    to match the original data's characteristics.

    Args:
        output_path (str): The path where the synthetic anndata file will be saved.
        template_file_path (str): The path to an existing anndata file to use as a
                                  template for dimensions and statistics.
    """
    # Load the template anndata file to get the dimensions and statistics
    template_adata = anndata.read_h5ad(template_file_path)
    n_obs, n_vars = template_adata.shape

    # Calculate statistics from the real data
    if hasattr(template_adata.X, 'toarray'):
        real_expression = template_adata.X.toarray()
    else:
        real_expression = template_adata.X
        
    mean_val = np.mean(real_expression[real_expression > 0])
    std_val = np.std(real_expression[real_expression > 0])
    sparsity = (np.sum(real_expression == 0) / real_expression.size)

    # Generate random data from a log-normal distribution
    # Parameters for the log-normal are derived from the mean and std of the log-transformed data
    log_data = np.log1p(real_expression[real_expression > 0])
    mu = np.mean(log_data)
    sigma = np.std(log_data)
    
    random_expression = np.random.lognormal(mean=mu, sigma=sigma, size=(n_obs, n_vars))
    
    # Introduce sparsity
    num_zeros = int(sparsity * random_expression.size)
    zero_indices = np.random.choice(random_expression.size, num_zeros, replace=False)
    random_expression.ravel()[zero_indices] = 0
    
    # Convert to float32 for memory efficiency
    random_expression = random_expression.astype(np.float32)

    # Create a new AnnData object
    adata = anndata.AnnData(random_expression,
                              obs=template_adata.obs.copy(),
                              var=template_adata.var.copy(),
                              uns=template_adata.uns.copy())

    # Set the .raw attribute, which the processing script expects
    adata.raw = adata

    # Generate random spatial coordinates
    min_x, max_x = template_adata.obs['center_x'].min(), template_adata.obs['center_x'].max()
    min_y, max_y = template_adata.obs['center_y'].min(), template_adata.obs['center_y'].max()

    random_x = np.random.uniform(min_x, max_x, n_obs)
    random_y = np.random.uniform(min_y, max_y, n_obs)
    
    adata.obs['center_x'] = random_x
    adata.obs['center_y'] = random_y

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the new anndata object
    adata.write(output_path)
    print(f"Synthetic data with realistic statistics saved to {output_path}")

if __name__ == '__main__':
    # Get the absolute path of the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    template_file = os.path.join(script_dir, 'ABN_57CJ_area17_region0/processed_data_qc_only.h5ad')
    output_file = os.path.join(script_dir, 'ABN_synthetic_data_region0/processed_data_qc_only.h5ad')
    
    if not os.path.exists(template_file):
        print(f"Error: Template file not found at {template_file}")
        print("Please ensure the path is correct and the file exists.")
    else:
        create_synthetic_data(output_file, template_file) 
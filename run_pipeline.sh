#!/bin/bash
#
# This script runs the full processing pipeline for the spatially-aware graph model.
# It starts with quality control, then trains a model, runs inference, and
# finally generates visualizations of the results.

# --- Configuration ---
# Set the model type you want to train and use. Options: 'GAT' or 'GCN'.
MODEL_TYPE="GAT"
# Set the decorrelation strength for training.
DECORR_STRENGTH=0.5
# Set the Python interpreter.
PYTHON_CMD="python3.10"

echo "--- Starting Full Processing Pipeline ---"

# --- Step 1: Quality Control ---
echo -e "\n[Step 1/4] Running Quality Control..."
$PYTHON_CMD qc_and_umap.py
if [ $? -ne 0 ]; then
    echo "Error during QC. Aborting."
    exit 1
fi
echo "Quality Control complete."

# --- Step 2: Training the Model ---
echo -e "\n[Step 2/4] Training the ${MODEL_TYPE} model..."
$PYTHON_CMD self_supervised_graph_model.py --model_type ${MODEL_TYPE} --decorrelation_strength ${DECORR_STRENGTH}
if [ $? -ne 0 ]; then
    echo "Error during model training. Aborting."
    exit 1
fi
echo "Model training complete."

# Define the expected model and results filenames based on the configuration
MODEL_PATH="global_${MODEL_TYPE,,}_model_with_neighbors.pt" # ,, converts to lowercase
INFERENCE_RESULTS_FILE="processed_data_inference_latents.h5ad"

# --- Step 3: Running Inference ---
echo -e "\n[Step 3/4] Running inference with the trained model..."
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Trained model file '${MODEL_PATH}' not found. Aborting."
    exit 1
fi
$PYTHON_CMD inference.py --load_model_path ${MODEL_PATH} --model_type ${MODEL_TYPE}
if [ $? -ne 0 ]; then
    echo "Error during inference. Aborting."
    exit 1
fi
echo "Inference complete."

# --- Step 4: Visualizing the Results ---
echo -e "\n[Step 4/4] Generating visualizations..."
# Option A: Standard, dataset-specific visualization
echo "Generating standard visualizations..."
$PYTHON_CMD visualize_latents.py --input_file ${INFERENCE_RESULTS_FILE}

# Option B: Globally consistent visualization
echo "Generating globally consistent visualizations..."
$PYTHON_CMD visualize_latents_globally.py --input_file ${INFERENCE_RESULTS_FILE}
echo "Visualizations complete."

echo -e "\n--- Full Processing Pipeline Finished Successfully ---" 
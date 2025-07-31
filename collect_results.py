import os
import glob
import shutil

def main():
    # --- Configuration ---
    source_filename = "global_spatial_pca_rgb_gat_inference.png"
    destination_dir = "RUN1"
    
    print(f"--- Starting Result Collection ---")
    print(f"Source filename: '{source_filename}'")
    print(f"Destination directory: '{destination_dir}'")

    # --- Create Destination Directory ---
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"Created destination directory: {destination_dir}")

    # --- Find and Process ABN Directories ---
    abn_directories = [d for d in glob.glob('ABN*') if os.path.isdir(d)]
    print(f"Found {len(abn_directories)} potential source directories.")
    
    files_copied = 0
    for directory in abn_directories:
        source_path = os.path.join(directory, source_filename)
        
        if os.path.exists(source_path):
            # Construct the new filename
            new_filename = f"{os.path.basename(directory)}_glob_gat.png"
            destination_path = os.path.join(destination_dir, new_filename)
            
            # Copy and rename the file
            shutil.copyfile(source_path, destination_path)
            print(f"Copied: {source_path} -> {destination_path}")
            files_copied += 1
        else:
            print(f"Skipping: Source file not found in {directory}")

    print(f"\n--- Collection Complete ---")
    print(f"Successfully copied {files_copied} files to '{destination_dir}'.")

if __name__ == '__main__':
    main() 
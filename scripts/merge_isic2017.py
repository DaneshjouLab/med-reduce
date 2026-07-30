import os
import shutil
import pandas as pd
from pathlib import Path

def merge_isic_ground_truth_part3(ground_truth_files, output_filename="merged_ground_truth_part3.csv"):    
    all_gt_dfs = []
    
    for split_name, file_path in ground_truth_files.items():
        file_path = Path(file_path) # Use Path for easier handling
        
        if file_path.exists():
            print(f"Reading {split_name} data from: {file_path.name}")
            try:
                df = pd.read_csv(file_path)
                
                df['original_split'] = split_name
                
                all_gt_dfs.append(df)
                print(f"  -> Successfully loaded {df.shape[0]} records.")
            except Exception as e:
                print(f"  -> ERROR reading {file_path.name}: {e}")
        else:
            print(f"  -> WARNING: File not found: {file_path}. Skipping.")


    if not all_gt_dfs:
        print("\nERROR: No ground truth files were successfully loaded. Aborting merge.")
        return pd.DataFrame()

    final_gt_df = pd.concat(all_gt_dfs, ignore_index=True)
    
    try:
        final_gt_df.to_csv(output_filename, index=False)
        print(f"\n--- Merge Complete ---")
        print(f"Total merged records: {final_gt_df.shape[0]}")
        print(f"Saved collective ground truth to: {output_filename}")
    except Exception as e:
        print(f"\nERROR saving merged CSV: {e}")
        
    return final_gt_df

ROOT_DIR = Path(os.environ.get("MR_DATA_ROOT", "/scratch/groups/roxanad/datasets")) / "isic/challenges/2017" 

OUTPUT_DIR = ROOT_DIR / "merged_isic_2017_data"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_CSV_FILE = OUTPUT_DIR / "merged_metadata.csv"
OUTPUT_GROUND_TRUTH = OUTPUT_DIR / "merged_ground_truth_part3.csv"

DATA_SETS = {
    "train": "ISIC-2017_Training_Data",
    "val": "ISIC-2017_Validation_Data",
    "test": "ISIC-2017_Test_v2_Data",
}

ground_truth_files = {
    "train": ROOT_DIR / "ISIC-2017_Training_Part3_GroundTruth.csv",
    "val": ROOT_DIR / "ISIC-2017_Validation_Part3_GroundTruth.csv",
    "test": ROOT_DIR / "ISIC-2017_Test_v2_Part3_GroundTruth.csv", 
}

print(f"Starting merge process. Creating output directory at: {OUTPUT_DIR}")

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

all_metadata_dfs = []


for split_name, folder_name in DATA_SETS.items():
    print(f"\n--- Processing {split_name.upper()} set ({folder_name}) ---")
    
    source_dir = ROOT_DIR / folder_name / folder_name
    
    if not source_dir.exists():
        print(f"WARNING: Source directory not found: {source_dir}. Skipping this set.")
        continue

    print("Copying images...")
    
    image_files = [f for f in source_dir.iterdir() if f.suffix in ['.jpg', '.png']]
    
    for img_path in image_files:
        try:
            shutil.copy2(img_path, OUTPUT_IMAGE_DIR / img_path.name)
        except Exception as e:
            print(f"ERROR copying {img_path.name}: {e}")
            
    print(f"Successfully copied {len(image_files)} images to {OUTPUT_IMAGE_DIR}.")

    print("Reading and consolidating metadata...")
    
    csv_file_name = f"{folder_name}_metadata.csv"
    csv_path = source_dir / csv_file_name

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            
            # Add a column to mark the original split
            df['original_split'] = split_name 
            
            all_metadata_dfs.append(df)
            print(f"Successfully loaded metadata from {csv_file_name}.")
            print(f"Metadata shape: {df.shape}")
            
        except Exception as e:
            print(f"ERROR reading {csv_file_name}: {e}")
    else:
        print(f"WARNING: Metadata CSV not found: {csv_path}. Skipping metadata for this set.")

if all_metadata_dfs:
    print("\n--- Finalizing Collective CSV ---")
    
    final_df = pd.concat(all_metadata_dfs, ignore_index=True)
    
    final_df.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print(f"Successfully merged {len(all_metadata_dfs)} CSVs.")
    print(f"Final merged CSV saved to: {OUTPUT_CSV_FILE}")
    print(f"Total number of rows in CSV: {final_df.shape[0]}")

    print("\n*** Starting Ground Truth Consolidation ***")
    merged_gt_df = merge_isic_ground_truth_part3(ground_truth_files, output_filename=OUTPUT_GROUND_TRUTH)
    
else:
    print("\n--- Process complete, but no metadata was successfully loaded. ---")

print("\n✨ Merge process finished! ✨")
import os
import shutil

# 1. SET YOUR PATHS
# Path to the GitHub repo you cloned (the one with the training_data)
SOURCE_PATH = r'C:\Users\Ankitha Hathwar\OneDrive\Documents\gitsnip\Snyptr\training_data' 
# Path to your project's seed folder
DEST_PATH = r'C:\Users\Ankitha Hathwar\OneDrive\Documents\Snypter\Shooting_Error_Analysis\Shooting_Dataset'

# 2. THE MAPPING
# This maps the names in that GitHub repo to YOUR C1-C7 folders
mapping = {
    "to and fro motion": "C1_to_and_fro",
    "frontsight_dip": "C2_frontsight_dip",
    "overtight_grip": "C3_over-tight_grip",
    "breath_control": "C4_breathe_control",
    "early_recoil": "C5_early_recoil",
    "stance_position": "C6_stance",
    "acute_angle_trigger": "C7_acute_angle"
}

def collect():
    count = 0
    # os.walk scans every subfolder automatically
    for root, dirs, files in os.walk(SOURCE_PATH):
        folder_name = os.path.basename(root)
        
        if folder_name in mapping:
            target_folder = mapping[folder_name]
            dest_dir = os.path.join(DEST_PATH, target_folder)
            
            print(f"Copying from {folder_name} to {target_folder}...")
            
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy2(os.path.join(root, file), os.path.join(dest_dir, file))
                    count += 1
                    
    print(f"Finished! Successfully moved {count} images into your Shooting_Dataset.")

if __name__ == "__main__":
    collect()
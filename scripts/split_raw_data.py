import splitfolders
import os

print("Starting the raw data split...")

# Where your raw photos are sitting right now
input_folder = r"C:\Users\Ankitha Hathwar\OneDrive\Documents\Snypter\Shooting_Error_Analysis\Shooting_Dataset"

# Where the cleanly split folders will be generated
output_folder = r"C:\Users\Ankitha Hathwar\OneDrive\Documents\Snypter\Shooting_Error_Analysis\Raw_Split_Dataset"

# This cleanly slices the data: 80% Training, 10% Validation, 10% Final Test
# seed=42 ensures that if you run this again, it splits the exact same way
splitfolders.ratio(
    input_folder, 
    output=output_folder, 
    seed=42, 
    ratio=(0.8, 0.1, 0.1), 
    group_prefix=None
)

print(f"Clean split complete! Your raw Train, Val, and Test folders are waiting in: {output_folder}")
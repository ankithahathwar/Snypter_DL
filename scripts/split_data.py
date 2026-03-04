import splitfolders
import os

# Use the dynamic path logic
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Now these paths work regardless of the folder name!
input_folder = os.path.join(BASE_DIR, 'Augmented_Data')
output_folder = os.path.join(BASE_DIR, 'Final_Training_Set')

splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.8, .1, .1))
print(f"Data split successfully in: {output_folder}")
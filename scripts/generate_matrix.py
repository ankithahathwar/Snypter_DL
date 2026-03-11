import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

print("Loading the trained baseline model")
# Make sure this matches the name of the model you saved after your 28% run!
model = tf.keras.models.load_model('comparative_baseline.keras') 

# --- CRITICAL STEP: NO SHUFFLING ---
# We must set shuffle=False, otherwise the AI's answers won't line up with the real labels
test_dir = r"C:\Users\Ankitha Hathwar\OneDrive\Documents\Snypter\Shooting_Error_Analysis\Raw_Split_Dataset\test"

print("Loading the honest test data")
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(128, 128),
    batch_size=32,
    shuffle=False 
)

class_names = test_ds.class_names

# --- MAKE PREDICTIONS ---
print("Taking the final exam to generate the matrix...")
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)

# Extract the true answers straight from the folders
true_classes = np.concatenate([y for x, y in test_ds], axis=0)

# --- BUILD & SAVE THE GRAPH ---
print("Drawing the Confusion Matrix...")
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10, 8))
# cmap='Blues' gives it that professional corporate look
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)

plt.title('True Baseline CNN: SCATT Error Confusion Matrix', fontsize=14)
plt.ylabel('Actual True Error', fontsize=12)
plt.xlabel('Model Predicted Error', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save it as a high-res image for your PowerPoint!
plt.savefig('raw_confusion_matrix.png', dpi=300)

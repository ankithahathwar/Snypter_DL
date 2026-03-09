import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. CLEAN SLATE ---
tf.keras.backend.clear_session()
print(" Memory wiped. Starting the official Comparative Baseline run.")

# --- 2. THE PATHS ---
# Pointing directly to your perfectly organized folders
train_dir = r"C:\Users\Ankitha Hathwar\OneDrive\Documents\Snypter\Shooting_Error_Analysis\Final_Training_Set\train"
val_dir = r"C:\Users\Ankitha Hathwar\OneDrive\Documents\Snypter\Shooting_Error_Analysis\Final_Training_Set\val"
test_dir = r"C:\Users\Ankitha Hathwar\OneDrive\Documents\Snypter\Shooting_Error_Analysis\Final_Training_Set\test"

# --- 3. THE ROCK-SOLID DATA LOADERS ---
# We use the built-in loader that guarantees labels and images never get mixed up
train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=(128, 128), batch_size=32)
val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=(128, 128), batch_size=32)
test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=(128, 128), batch_size=32)

# This physically prevents the CPU from starving for data (Fixes the crashing issues)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 4. THE BASELINE ARCHITECTURE ---
model = models.Sequential([
    # Automatically scales the pixels
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Keeping the shock absorber!
    layers.Dense(7) # 7 Shooting Error Classes
])

# --- 5. TRAINING ENGINE ---
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# --- 6. EXECUTION ---
print("\nCommencing Baseline Training...")
# Train on 'train', check progress on 'val'
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# --- 7. THE FINAL EXAM ---
print("\nRunning the ultimate evaluation on the test folder")
test_loss, test_acc = model.evaluate(test_ds)
print(f"OFFICIAL BASELINE TEST ACCURACY: {test_acc:.4f}")

model.save('comparative_baseline.keras')
print("Official baseline saved for the final report!")
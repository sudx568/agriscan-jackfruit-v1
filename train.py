import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os
import sys

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
DATASET_DIR = 'dataset'

# 1. Verify Folder Structure Before Starting
if not os.path.exists(DATASET_DIR):
    print(f"❌ ERROR: folder '{DATASET_DIR}' not found!")
    sys.exit()

for cls in ['Bad', 'Good']:
    path = os.path.join(DATASET_DIR, cls)
    if not os.path.exists(path):
        print(f"❌ ERROR: Subfolder '{path}' is missing!")
        sys.exit()

# 2. Data Augmentation & Loading
datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2, 
    rotation_range=20,
    horizontal_flip=True
)

print("--- Loading Images ---")
# Explicitly setting classes=['Bad', 'Good'] ensures 0=Bad, 1=Good
train_data = datagen.flow_from_directory(
    DATASET_DIR, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE,
    class_mode='binary', 
    classes=['Bad', 'Good'], 
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_DIR, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE,
    class_mode='binary', 
    classes=['Bad', 'Good'], 
    subset='validation'
)

# 3. Final Check: Did we actually find images?
print(f"✅ Found {train_data.samples} training images.")
print(f"✅ Class Mapping: {train_data.class_indices}")

if train_data.samples == 0:
    print("❌ ERROR: No images found. Ensure photos are directly inside dataset/Bad and dataset/Good.")
    sys.exit()

# 4. Build Model (Transfer Learning with MobileNetV2)
print("--- Building Model ---")
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the pre-trained weights

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') # 0 to 1 output
])

# 5. Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 6. Train
print("--- Starting Training ---")
try:
    model.fit(
        train_data, 
        validation_data=val_data, 
        epochs=EPOCHS
    )
    
    # 7. Save Final Model
    model.save('jackfruit_model.keras')
    print("\n✅ SUCCESS: Model saved as 'jackfruit_model.keras'")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
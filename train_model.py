# This is train_model.py (Corrected for FAST MODE)
# Uses a small subset of data to generate deliverables quickly.

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import numpy as np
import os
import sys

# --- Constants ---
INPUT_SIZE = 224 # MobileNetV2 requires 224x224
NUM_CLASSES = 10
BATCH_SIZE = 32 
MODEL_SAVE_PATH = 'deliverables/model_transfer_learning.h5'
METRICS_SAVE_PATH = 'deliverables/final_metrics.txt'

# --- FAST MODE SETTINGS ---
# We will only use 5000 images to train quickly for the deadline.
# This will result in low accuracy, but will complete the project.
INPUT_SIZE = 224 # MobileNetV2 requires 224x224
NUM_CLASSES = 10
EPOCHS = 50 
PATIENCE = 10 # Stop if no improvement for 10 epochs
BATCH_SIZE = 32 # Process 32 images at a time
MODEL_SAVE_PATH = 'deliverables/model_transfer_learning.h5'
METRICS_SAVE_PATH = 'deliverables/final_metrics.txt'
# --- END FAST MODE SETTINGS ---


# --- 1. Data Pipeline ---

def load_and_split_data():
    """
    Loads and splits CIFAR-10 data. 
    """
    print("[INFO] Loading CIFAR-10 data...")
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

    # --- Create Validation Set (Using 15% of 50,000) ---
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full, 
        y_train_full, 
        test_size=0.15, # 15% for validation
        random_state=42
    )

    print(f"[INFO] Data split complete:")
    print(f"  Training samples:   {len(x_train)}")
    print(f"  Validation samples: {len(x_valid)}")
    print(f"  Test samples:       {len(x_test)}")

    # --- One-Hot Encode Labels ---
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_valid = to_categorical(y_valid, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def create_dataset(x, y, is_training=True):
    """
    Creates a tf.data.Dataset that handles resizing and
    preprocessing on-the-fly in batches.
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    # Preprocessing function
    def _preprocess(image, label):
        image = tf.image.resize(image, (INPUT_SIZE, INPUT_SIZE), method='area')
        image = preprocess_input(image)
        return image, label

    dataset = dataset.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# --- 2. Model Architecture (TASK 1: Transfer Learning) ---

def build_model():
    """
    Builds the Transfer Learning model (MobileNetV2).
    """
    print("[INFO] Building MobileNetV2 Transfer Learning model...")
    
    base_model = MobileNetV2(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("[INFO] Model built and compiled successfully.")
    model.summary()
    return model

# --- 3. Evaluation Saving ---

def save_evaluation_to_file(loss, acc, report, matrix):
    """
    Saves the final evaluation metrics to a text file.
    """
    print(f"[INFO] Saving metrics to {METRICS_SAVE_PATH}...")
    os.makedirs(os.path.dirname(METRICS_SAVE_PATH), exist_ok=True)
    
    with open(METRICS_SAVE_PATH, 'w') as f:
        f.write("--- AI-Powered Image Classification ---")
        f.write("\n--- Final Test Metrics ---\n")
        f.write(f"Final Test Loss:     {loss:.4f}\n")
        f.write(f"Final Test Accuracy: {acc:.4f}\n")
        f.write("\n--- Classification Report ---\n")
        f.write(report)
        f.write("\n--- Confusion Matrix ---\n")
        f.write(np.array2string(matrix))
    print("[INFO] Metrics saved.")

# --- 4. Main Execution ---

if __name__ == "__main__":
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # 1. Get Data (small 32x32 images)
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_and_split_data()

    # 2. Create Data Generators
    print("[INFO] Creating data generators...")
    train_dataset = create_dataset(x_train, y_train, is_training=True)
    valid_dataset = create_dataset(x_valid, y_valid, is_training=False)
    test_dataset = create_dataset(x_test, y_test, is_training=False)
    
    # Store the true labels for the test set for later reports
    y_true_classes = np.argmax(y_test, axis=1)

    # 3. Build Model
    model = build_model()

    # 4. Define Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE, # Use fast patience
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # 5. Train Model (using the generators)
    print("\n--- [START] Model Training ---")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS, # Use fast epochs
        validation_data=valid_dataset,
        callbacks=[early_stopping, model_checkpoint]
    )
    print("--- [END] Model Training ---\n")

    # 6. Evaluate Model
    print("--- [START] Final Evaluation ---")
    print(f"[INFO] Loading best model from {MODEL_SAVE_PATH}...")
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    
    # Evaluate using the test generator
    loss, acc = best_model.evaluate(test_dataset, verbose=0)
    
    # Predict using the test generator
    y_pred_probs = best_model.predict(test_dataset)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    y_pred_classes = y_pred_classes[:len(y_true_classes)]

    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0)
    matrix = confusion_matrix(y_true_classes, y_pred_classes)
    
    print("\n--- Final Test Metrics ---")
    print(f"Final Test Loss:     {loss:.4f}")
    print(f"Final Test Accuracy: {acc:.4f}\n")
    print("--- Classification Report ---")
    print(report)
    print("--- Confusion Matrix ---")
    print(matrix)
    print("--- [END] Final Evaluation ---\n")

    # 7. Save Evaluation
    save_evaluation_to_file(loss, acc, report, matrix)
    
    print(f"[SUCCESS] Project complete. Model saved to {MODEL_SAVE_PATH}")
import os
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- Config ---
data_path = "Dataset/Garbage classification"
img_size = (224, 224)
batch_size = 32
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
initial_epochs = 10
fine_tune_epochs = 10

# --- Data loading ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Augmentation ---
augment = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomFlip('vertical'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# --- Class weights ---
class_counts = [393, 491, 400, 584, 472, 127]
total = sum(class_counts)
classes = len(class_counts)
class_weight = {i: total / (classes * class_counts[i]) for i in range(classes)}

# --- Model ---
preprocess_input = keras.applications.mobilenet_v2.preprocess_input
img_shape = img_size + (3,)
base_model = keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
base_model.trainable = False

model = keras.Sequential([
    layers.Lambda(preprocess_input),
    augment,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names)),
])

callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
]

# --- Phase 1: Feature extraction ---
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

history = model.fit(
    train_ds,
    epochs=initial_epochs + fine_tune_epochs,
    validation_data=val_ds,
    callbacks=callbacks,
    class_weight=class_weight,
)

# --- Phase 2: Fine-tuning ---
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

history_finetuning = model.fit(
    train_ds,
    epochs=fine_tune_epochs,
    validation_data=val_ds,
    callbacks=callbacks,
    class_weight=class_weight,
)

model.save('garbage_classifier.keras')

# --- Plot ---
acc = history.history['accuracy'] + history_finetuning.history['accuracy']
val_acc = history.history['val_accuracy'] + history_finetuning.history['val_accuracy']
loss = history.history['loss'] + history_finetuning.history['loss']
val_loss = history.history['val_loss'] + history_finetuning.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.0, 1])
plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('accuracy_plot16.png')
plt.show()
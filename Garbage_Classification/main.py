import os, warnings     


import tensorflow as tf 
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers, models  # type: ignore

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

data_path = "Dataset/Garbage classification"

img_size = (224,224)
batch_size = 32
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore")

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split = 0.2 ,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split = 0.2 ,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

augment = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
])

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# for image, _ in train_ds.take(1):
#     plt.figure(figsize=(10,10))
#     first_image= image[0]
#     for i in range(9):
#         ax = plt.subplot(3,3, i+1)
#         augmented_img = augment(tf.expand_dims(first_image, 0))
#         plt.imshow(augmented_img[0]/255)
#         plt.axis('off')
# plt.savefig('augmentation1.png')
# plt.show()


preprocess_input = keras.applications.mobilenet_v2.preprocess_input

img_shape = img_size + (3,)
base_model = keras.applications.MobileNetV2(input_shape=img_shape, include_top = False, weights='imagenet')
base_model.trainable=False

# image_batch, label_batch = next(iter(train_ds))
# feature_batch = base_model(image_batch)
# print(feature_batch.shape)

# base_model.summary()

model = keras.Sequential([
    layers.Lambda(preprocess_input),
    augment,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],)

history = model.fit(train_ds, epochs=10, validation_data=val_ds)

# Plot accuracy over epochs
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_plot1.png')
plt.show()
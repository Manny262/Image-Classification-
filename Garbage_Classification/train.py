import os, warnings     


import tensorflow as tf 
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers, models  # type: ignore

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
# tf.random.set_seed(42)
# np.random.seed(42)

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
    layers.RandomFlip('vertical'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    # layers.RandomTranslation(0.1, 0.1),
    # layers.RandomBrightness(0.2),
    # layers.RandomContrast(0.2),
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
    # layers.Dropout(0.3),
    # layers.BatchNormalization(),
    layers.Dropout(0.5), #was 0.5
    layers.Dense(len(class_names))
])

class_counts = [393, 491, 400, 584, 472, 127]
total = sum(class_counts)
classes = len(class_counts)

callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)]

class_weight= {i: total / (classes * class_counts[i]) for i in range(classes)}


model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],)

history_base_epoch=10
history_epoch = history_base_epoch + 10

history = model.fit(train_ds, epochs=history_epoch, validation_data=val_ds, callbacks=callbacks, class_weight=class_weight,)

base_model.trainable = True

for layer in base_model.layers[:-20]: 
    layer.trainable = False
    
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],)

history_finetuning= model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=callbacks, class_weight=class_weight,)

model.save('garbage_classifier.keras')

print("History accuracy:", history.history['accuracy'])
print("History val_accuracy:", history.history['val_accuracy'])

print("History_finetuning accuracy:",history_finetuning.history['accuracy'])
print("History_finetuning val_accuracy:", history_finetuning.history['val_accuracy'])

acc = history.history['accuracy'] + history_finetuning.history['accuracy']
val_acc = history.history['val_accuracy'] + history_finetuning.history['val_accuracy']
loss = history.history['loss'] + history_finetuning.history['loss']
val_loss = history.history['val_loss'] + history_finetuning.history['val_loss']
initial_epochs = 10

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.0, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('accuracy_plot16.png')
plt.show()
# print("Number of layers in the base model: ", len(base_model.layers))
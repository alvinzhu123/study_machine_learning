import tensorflow as tf
from tensorflow.keras.layers import Dense ,Conv2D, Flatten, AveragePooling2D,\
    Dropout, BatchNormalization,Activation,Embedding
from tensorflow.keras import datasets
from tensorflow.keras import Model, Sequential
import numpy as np
import os

(x_img_train, y_label_train), (x_img_test, y_label_test) = datasets.mnist.load_data()
#add a new dim for mnist because of conv2D
x_img_train = np.expand_dims(x_img_train, axis=3)
x_img_test = np.expand_dims(x_img_test, axis=3)
x_img_train_normalize = x_img_train.astype('float32') / 255.
x_img_test_normalize = x_img_test.astype('float32') / 255.
y_label_train_Onehot = tf.keras.utils.to_categorical(1, len(y_label_train))
y_label_test_Onehot = tf.keras.utils.to_categorical(1, len(y_label_test))


model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    AveragePooling2D(pool_size=(2,2), strides=2),
    Dropout(.2),
    Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    AveragePooling2D(pool_size=(2,2), strides=2),
    Dropout(.2),
    Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    AveragePooling2D(pool_size=(2,2), strides=2),
    Dropout(.2),
    Flatten(),
    Dense(10, activation='softmax')
])
lr_callbacks = tf.keras.callbacks.ReduceLROnPlateau('val_loss', 0.1, 200)
es_callbacks = tf.keras.callbacks.EarlyStopping('val_loss',min_delta=0,patience=300)
tb_callbacks = tf.keras.callbacks.TensorBoard('./tensorboard')
checkpoint_save_path = "./ckpt/cifar10.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

ckpt_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
model.compile(
    optimizer=tf.keras.optimizers.Nadam(0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

history = model.fit(
    x=x_img_train_normalize,
    y=y_label_train_Onehot,
    batch_size=128,
    epochs=10,
    callbacks=[lr_callbacks, es_callbacks, ckpt_callbacks],
    validation_data=(x_img_test_normalize, y_label_test_Onehot),
    shuffle=True,
    validation_freq=1)
model.summary()
# from tensorboard import notebook
# notebook.list()
# #在tensorboard中查看模型
# notebook.start("--logdir ./tensorboard")
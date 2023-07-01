import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

target_size = (250, 150)
batch_size = 32
epochs = 50
directory = "dataset"
validation_split = 0.20

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (5, 5), padding='valid', activation='relu', input_shape=(250, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        #keras.layers.Dropout(rate=0.05),  # adding dropout regularization throughout the model to deal with overfitting
        tf.keras.layers.BatchNormalization(),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3, 3), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.00005),
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # keras.layers.Dropout(rate=0.10),
        tf.keras.layers.BatchNormalization(),
        # The third convolution
        tf.keras.layers.Conv2D(32, (3, 3), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.00005),
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        #keras.layers.Dropout(rate=0.15),
        tf.keras.layers.BatchNormalization(),

        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # neuron hidden layer
        tf.keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(rate=0.5),
        # 8 output neuron for the 8 classes of Tank Images
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    return model

def train_tanks():
    # Defines & compiles the model
    model = create_model()
    print(model.summary())

    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    reduce_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=math.sqrt(0.1), patience=5)
    #
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])

    # Creates an instance of an ImageDataGenerator called train_datagen, and a train_generator, train_datagen.flow_from_directory
    # splits data into training and testing(validation) sets
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,  validation_split=validation_split)

    # training data
    train_generator = train_datagen.flow_from_directory(
        directory,  # Source directory
        target_size=target_size,  # Resizes images
        batch_size=batch_size,
        class_mode='categorical', subset='training')

    # Testing data
    validation_generator = train_datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')  # set as validation data

    labels = display_classes(train_generator)
    #display_sample_images(labels, train_generator)

    # Model fitting for a number of epochs
    #steps_per_epoch = train_generator.samples // train_generator.batch_size
    #validation_steps = validation_generator.samples // validation_generator.batch_size

    history = model.fit(
        train_generator,
        #steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        #validation_steps=validation_steps,
        verbose=1,
        callbacks=[reduce_rate])
    display_graph(history)

    # displays accuracy of training
    print("Training Accuracy:"), print(history.history['acc'][-1])
    print("Testing Accuracy:"), print(history.history['val_acc'][-1])


def display_graph(history):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['acc'], label='Training Accuracy')
    plt.plot(epochs_range, history.history['val_acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], label='Training Loss')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def display_sample_images(labels, train_generator):
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(6, 6))
    idx = 0
    for i in range(2):
        for j in range(5):
            label = labels[np.argmax(train_generator[0][1][idx])]
            ax[i, j].set_title(label)
            ax[i, j].imshow(train_generator[0][0][idx][:, :, :])
            ax[i, j].axis("off")
            idx += 1
    plt.tight_layout()
    plt.suptitle("Sample training images")
    plt.show()


def display_classes(train_generator):
    labels = {value: key for key, value in train_generator.class_indices.items()}
    print("Labels")
    for key, value in labels.items():
        print(key, value)
    return labels


if __name__ == '__main__':
    train_tanks()


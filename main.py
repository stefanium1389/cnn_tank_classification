import math
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

target_size = (250, 150)
batch_size = 32
epochs = 10
directory = "dataset"
test_directory = "test"
validation_split = 0.20

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu', input_shape=(250, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(rate=0.05),  # adding dropout regularization throughout the model to deal with overfitting
        tf.keras.layers.BatchNormalization(),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.00005),
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(rate=0.10),
        tf.keras.layers.BatchNormalization(),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.00005),
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(rate=0.15),
        tf.keras.layers.BatchNormalization(),

        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # neuron hidden layer
        tf.keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(rate=0.25),
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

    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(test_directory,
                                                      shuffle=False,
                                                      batch_size=batch_size,
                                                      target_size=(250, 150),
                                                      class_mode='categorical')

    predictions = model.predict(test_generator)

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(6, 4))
    idx = 0

    for i in range(2):
        for j in range(5):
            predicted_label = labels[np.argmax(predictions[idx])]
            ax[i, j].set_title(f"{predicted_label}")
            ax[i, j].imshow(test_generator[0][0][idx])
            ax[i, j].axis("off")
            idx += 1

    plt.tight_layout()
    plt.suptitle("Test Dataset Predictions", fontsize=20)
    plt.show()

    test_loss, test_accuracy = model.evaluate(test_generator, batch_size=batch_size)

    print(f"Test Loss:     {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    y_pred, y_true = display_confusion_matrix(labels, predictions, test_generator)

    print(classification_report(y_true, y_pred, target_names=labels.values()))

    errors = (y_true - y_pred != 0)
    display_error_predictions(errors, labels, test_generator, y_pred, y_true)


def display_confusion_matrix(labels, predictions, test_generator):
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    cf_mtx = confusion_matrix(y_true, y_pred)
    group_counts = ["{0:0.0f}".format(value) for value in cf_mtx.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_mtx.flatten() / np.sum(cf_mtx)]
    box_labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(8, 8)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cf_mtx, xticklabels=labels.values(), yticklabels=labels.values(),
                cmap="YlGnBu", fmt="", annot=box_labels)
    plt.xlabel('Predicted Classes')
    plt.ylabel('True Classes')
    plt.show()
    return y_pred, y_true


def display_error_predictions(errors, labels, test_generator, y_pred, y_true):
    y_true_errors = y_true[errors]
    y_pred_errors = y_pred[errors]
    test_images = test_generator.filenames
    test_img = np.asarray(test_images)[errors]
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(6, 4))
    idx = 0
    for i in range(2):
        for j in range(5):
            idx = np.random.randint(0, len(test_img))
            true_index = y_true_errors[idx]
            true_label = labels[true_index]
            predicted_index = y_pred_errors[idx]
            predicted_label = labels[predicted_index]
            ax[i, j].set_title(f"True Label: {true_label} \n Predicted Label: {predicted_label}")
            img_path = os.path.join(test_directory, test_img[idx])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[i, j].imshow(img)
            ax[i, j].axis("off")
    plt.tight_layout()
    plt.suptitle('Wrong Predictions made on test set', fontsize=20)
    plt.show()


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


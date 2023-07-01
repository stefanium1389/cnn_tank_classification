import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

target_size = (250, 150)
batch_size = 40
epochs = 15
directory = 'dataset'
validation_split = 0.25

def train_tanks():
    # Defines & compiles the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (5, 5), activation='relu', input_shape=(250, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(rate=0.05),  # adding dropout regularization throughout the model to deal with overfitting
        # The second convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(rate=0.10),
        # The third convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(rate=0.15),

        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),

        # 8 output neuron for the 8 classes of Tank Images
        tf.keras.layers.Dense(8, activation='softmax')
    ])

    adam = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])

    # Creates an instance of an ImageDataGenerator called train_datagen, and a train_generator, train_datagen.flow_from_directory
    # splits data into training and testing(validation) sets
    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=validation_split)



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

    # Model fitting for a number of epochs
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = validation_generator.samples // validation_generator.batch_size

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        verbose=1)

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
    # returns accuracy of training
    print("Training Accuracy:"), print(history.history['acc'][-1])
    print("Testing Accuracy:"), print(history.history['val_acc'][-1])


if __name__ == '__main__':
    train_tanks()


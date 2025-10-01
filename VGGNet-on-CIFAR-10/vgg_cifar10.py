import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def VGG16(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential()

    # Conv Block 1
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Conv Block 2
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Conv Block 3
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Conv Block 4
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten + Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model

if __name__ == "__main__":
    model = VGG16()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Print model summary
    model.summary()

    # Train model (short training for demo, increase epochs if GPU available)
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
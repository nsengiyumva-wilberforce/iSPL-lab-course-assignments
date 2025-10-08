# ==============================================
# VGGNet Model Performance Comparison
# CIFAR-10 Color vs Grayscale with Augmentations
# ==============================================
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load and normalize dataset
# ----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

# ----------------------------
# 2. Define a VGG-like model
# ----------------------------
def build_vgg(input_shape=(32, 32, 3)):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),

        # Block 2
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),

        # Block 3
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),

        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------------------------------------------
# 3. Define 10 augmentation scenarios
# ---------------------------------------------------
param_list = [
    {"rotation_range": 0, "width_shift_range": 0, "height_shift_range": 0, "zoom_range": 0, "horizontal_flip": False},
    {"rotation_range": 0, "width_shift_range": 0, "height_shift_range": 0, "zoom_range": 0, "horizontal_flip": True},
    {"rotation_range": 0, "width_shift_range": 0.1, "height_shift_range": 0.1, "zoom_range": 0, "horizontal_flip": False},
    {"rotation_range": 0, "width_shift_range": 0, "height_shift_range": 0, "zoom_range": 0.1, "horizontal_flip": False},
    {"rotation_range": 0, "width_shift_range": 0.1, "height_shift_range": 0.1, "zoom_range": 0, "horizontal_flip": True},
    {"rotation_range": 0, "width_shift_range": 0, "height_shift_range": 0, "zoom_range": 0.1, "horizontal_flip": True},
    {"rotation_range": 0, "width_shift_range": 0.1, "height_shift_range": 0.1, "zoom_range": 0.1, "horizontal_flip": False},
    {"rotation_range": 0, "width_shift_range": 0.1, "height_shift_range": 0.1, "zoom_range": 0.1, "horizontal_flip": True},
    {"rotation_range": 15, "width_shift_range": 0.1, "height_shift_range": 0.1, "zoom_range": 0.1, "horizontal_flip": True},
    {"rotation_range": 15, "width_shift_range": 0.1, "height_shift_range": 0.1, "zoom_range": 0.1, "horizontal_flip": True}
]

# ---------------------------------------------------
# 4. Train and evaluate models
# ---------------------------------------------------
results = []

for i, params in enumerate(param_list):
    print(f"\n===== ROUND {i+1} / 10 =====")
    print(f"Augmentation Parameters: {params}")

    datagen = ImageDataGenerator(**params)
    datagen.fit(x_train)

    # ----- Color VGG model -----
    color_model = build_vgg()
    color_model.fit(datagen.flow(x_train, y_train, batch_size=128),
                    epochs=10, validation_data=(x_test, y_test), verbose=1)
    color_acc = color_model.evaluate(x_test, y_test, verbose=0)[1]

    # ----- Grayscale VGG model -----
    x_train_gray = tf.image.rgb_to_grayscale(x_train)
    x_test_gray = tf.image.rgb_to_grayscale(x_test)
    datagen_gray = ImageDataGenerator(**params)
    datagen_gray.fit(x_train_gray)

    gray_model = build_vgg(input_shape=(32, 32, 1))
    gray_model.fit(datagen_gray.flow(x_train_gray, y_train, batch_size=128),
                   epochs=10, validation_data=(x_test_gray, y_test), verbose=1)
    gray_acc = gray_model.evaluate(x_test_gray, y_test, verbose=0)[1]

    # Choose best
    best_model = "Color" if color_acc >= gray_acc else "B&W"

    results.append({
        "Round": f"{i+1}차",
        "Rotation": params["rotation_range"],
        "Flip": "✅" if params["horizontal_flip"] else "❌",
        "Movement": params["width_shift_range"],
        "Zoom": params["zoom_range"],
        "Color_Acc": round(color_acc, 4),
        "B&W_Acc": round(gray_acc, 4),
        "Best_Model": best_model
    })

# ---------------------------------------------------
# 5. Display results
# ---------------------------------------------------
df = pd.DataFrame(results)
print("\n===== VGGNet Summary Table =====")
print(df.to_string(index=False))

# df.to_csv("vggnet_augmentation_results.csv", index=False)
# print("\nResults saved to vggnet_augmentation_results.csv")

# ---------------------------------------------------
# 6. Plot results
# ---------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(df["Round"], df["Color_Acc"], marker='o', label='Color VGGNet')
plt.plot(df["Round"], df["B&W_Acc"], marker='o', label='B&W VGGNet')
plt.title("VGGNet Performance Comparison per Augmentation Scenario")
plt.xlabel("Round (Augmentation Scenario)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import tensorflow as tf
import os
import cv2
import numpy as np

# Define the model architecture (you may need to adjust this based on your requirements)
model = tf.keras.Sequential(
    [
        # Add your convolutional layers, pooling layers, etc. here
        # For example:
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="linear", input_shape=(240, 320, 3)
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="sigmoid"),
        tf.keras.layers.Dense(4, activation="sigmoid"),  # 4 outputs for (x, y, w, h)
    ]
)

# Compile the model
model.compile(
    # optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"],
    optimizer="adam", loss="mse", metrics=["accuracy"],
    # optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"],
)  # You can adjust the loss and metrics as needed

# Load images and labels
image_dir = "./FaceRecognition/images/"
label_dir = "./FaceRecognition/labels/"

images = []
labels = []

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(image_dir, filename))
        img = cv2.resize(img, (320, 240))  # Resize down by 2x

        img = img / 255.0  # Normalize pixel values to range [0, 1]

        images.append(img)

        label_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(label_dir, label_filename), "r") as label_file:
            label_info = label_file.readline().split()
            label = [float(x) for x in label_info[1:]]  # Exclude label_index
            labels.append(label)

images = np.array(images)
labels = np.array(labels)



# Normalize label coordinates to range [0, 1]
print(labels[0])
# labels[:, [0, 2]] /= 640  # Normalize x and w
# labels[:, [1, 3]] /= 480  # Normalize y and h
# print(labels[0])
# Train the model
model.fit(images, labels, epochs=50)

# Save the trained model
model.save("model/FaceRecognition.keras")

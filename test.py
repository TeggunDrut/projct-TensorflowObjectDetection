import os

train_path = "./rock_paper_scissors/train/"
val_path   = "./rock_paper_scissors/val/"

def getFileNames(path):
    # Get all the file names from the path that end with .jpg
    file_names = [f for f in os.listdir(path) if f.endswith(".jpg")]
    return file_names

def getLabels(path):
    labels = [f for f in os.listdir(path) if f.endswith(".txt")]
    return labels

training_file_names = getFileNames(train_path)
training_file_labels = getLabels(train_path)
validation_file_names = getFileNames(val_path)
validation_file_labels = getLabels(val_path)

types = ["rock", "paper", "scissors"]

def readLabelFile(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    split = content[0].split(" ")
    t = int(split[0])
    
    x = float(split[1])
    y = float(split[2])
    w = float(split[3])
    h = float(split[4])
    
    if t == 0:
        return np.array([1, 0, 0, x, y, w, h])
    elif t == 1:
        return np.array([0, 1, 0, x, y, w, h])
    elif t == 2:
        return np.array([0, 0, 1, x, y, w, h])
    
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

images = np.array([plt.imread(train_path + training_file_names[0])])
images = images / 255

labels = np.array([readLabelFile(train_path + training_file_labels[0])])

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(480, 640, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),  
    tf.keras.layers.Dense(7, activation='sigmoid'),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(images, labels, epochs=100)

# predict on validation set
validation_images = np.array([plt.imread(val_path + validation_file_names[0])])
validation_images = validation_images / 255
validation_labels = np.array([readLabelFile(val_path + validation_file_labels[0])])
model.predict(validation_images)

# get the predicted bounding box
prediction = model.predict(validation_images)
print(prediction)

# predict in real time
import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    frame = frame / 255
    
    prediction = model.predict(np.array([frame]))
    prediction = prediction[0]
    
    # get the bounding box
    t = np.argmax(prediction[0:3])
    x = prediction[3]
    y = prediction[4]
    w = prediction[5]
    h = prediction[6]
    
    if t == 0:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        print("rock")
    elif t == 1:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        print("paper")
    elif t == 2:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        print("scissors")
        
    cv2.rectangle(frame, (0, 0), (100, 100), (0, 0, 0), -1)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import tensorflow as tf

# img_width = 640/4
# img_height = 480/4
img_width = 160
img_height = 120
# Load the trained model
model = tf.keras.models.load_model('model/maybe5.h5')

# Function to draw a bounding box on the frame
def draw_bounding_box(frame, x, y, w, h):
    # x1 = int(x * frame.shape[1])
    # y1 = int(y * frame.shape[0])
    # x2 = int((x + w) * frame.shape[1])
    # y2 = int((y + h) * frame.shape[0])
    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match the model input size
    resized_frame = cv2.resize(frame, (img_width, img_height))
    resized_frame = resized_frame / 255.0  # Normalize pixel values to range [0, 1]
    
    # Make predictions using the model
    input_data = np.expand_dims(resized_frame, axis=0)
    predictions = model.predict(input_data, verbose=0)
    
    # print(predictions)
    
    # Denormalize the predicted coordinates
    x, y, w, h = predictions[0]

    
    x *= (resized_frame.shape[1] * 4)
    y *= (resized_frame.shape[0] * 4)
    w *= (resized_frame.shape[1] * 4)
    h *= (resized_frame.shape[0] * 4)
    
    print(resized_frame.shape[0], resized_frame.shape[1])
    # Draw the bounding box on the frame
    draw_bounding_box(frame, x, y, w, h)
    # draw_bounding_box(frame, 500, 500, 50, 50)

    # Display the frame with the bounding box
    cv2.imshow('Face Detection', frame)

    # Exit when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

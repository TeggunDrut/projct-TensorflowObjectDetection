import cv2
import time
import os 

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default camera (you might need to change this if you have multiple cameras)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Create a variable to track the last capture time
last_capture_time = time.time()

imgNum = 0
types = [
    "rock",
    "paper",
    "scissors"
]

typeNum = 0

if not os.path.exists("./images"):
    os.mkdir("./images")

try:
    while typeNum < len(types):
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print("Error: Could not read a frame.")
            break

        # Check if it's time to capture a new image
        current_time = time.time()
        # check if space is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            if not os.path.exists(f"./images/{types[typeNum]}"):
                os.mkdir(f"./images/{types[typeNum]}")
            # Save the captured frame as an image
            image_filename = f"./images/{types[typeNum]}/{types[typeNum]}{imgNum}.jpg"
            cv2.imwrite(image_filename, frame)
            print(f"Captured {types[typeNum]}/{types[typeNum]}{imgNum}.jpg")

            # Update the last capture time
            last_capture_time = current_time
            
            imgNum += 1
            if imgNum >= 5:
                typeNum += 1
                imgNum = 0
                print("Press space to capture the next image")

        # Display the frame (optional)
        cv2.imshow("Webcam Capture", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the webcam and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

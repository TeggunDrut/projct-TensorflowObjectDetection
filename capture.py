import cv2
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default camera (you might need to change this if you have multiple cameras)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Define the interval (in milliseconds)
interval_ms = 1000

# Create a variable to track the last capture time
last_capture_time = time.time()

imgNum = 0

try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print("Error: Could not read a frame.")
            break

        # Check if it's time to capture a new image
        current_time = time.time()
        if current_time - last_capture_time >= interval_ms / 1000:
            # Save the captured frame as an image
            timestamp = int(time.time())
            image_filename = f"REAL/NewImages/face_{imgNum}.jpg"
            cv2.imwrite(image_filename, frame)
            print(f"Captured {image_filename}")

            # Update the last capture time
            last_capture_time = current_time
            
            imgNum += 1

        # Display the frame (optional)
        cv2.imshow("Webcam Capture", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the webcam and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

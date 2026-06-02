# pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv8 model 
model = YOLO("/path/to/your/pt/file.pt")  # or "yolov8s.pt", "yolov8m.pt", etc.

# Open the default camera (0). You can change the index if using an external camera.
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame 
    results = model.predict(source=frame, imgsz=640, conf=0.3)

    # Visualize the result on the frame
    annotated_frame = results[0].plot()  # returns a numpy array with the bounding boxes drawn

    # Display the resulting frame
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()





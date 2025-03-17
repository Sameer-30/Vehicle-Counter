from ultralytics import YOLO
import cv2  
import cvzone  
import numpy as np
from sort import *

# Load Video
cap = cv2.VideoCapture(r"C:\Users\LEGION\Desktop\Computer Vision\YOLO with web cam and video\Vehicle Counter\video.mp4")

# Load YOLO Model
model = YOLO("../Yolo-weights/yolov10l.pt")

# Class Names for YOLO Detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "pen", "watch", "charger"]

# Load Mask for Region of Interest (ROI)
mask = cv2.imread("Mask.png")
if mask is None:
    print("Error: Mask.png not found! Please check the file path.")
    exit()

# Object Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define Line for Counting Vehicles
limits = [156, 397, 1210, 397]
totalCount = []

# Video Processing Loop
while True:
    success, img = cap.read()
    if not success:
        break  # Exit loop if video ends

    # Apply Mask (to process only ROI)
    imgRegion = cv2.bitwise_and(img, mask)

    # YOLO Object Detection
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Detect only vehicles (car, truck, bus, bicycle, motorbike)
            if currentClass in ["car", "truck", "bus", "bicycle", "motorbike"] and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{currentClass} {conf:.2f}', (x1, y1 - 10), scale=1.3, thickness=1, offset=5)

                # Add detection to the array
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update Object Tracker
    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)  # Red Line

    for result in resultsTracker:
        x1, y1, x2, y2, obj_id = map(int, result)
        w, h = x2 - x1, y2 - y1

        # Display Object ID
        cvzone.putTextRect(img, f'ID: {int(obj_id)}', (x1, y1 - 10), scale=1.3, thickness=1, offset=5)

        # Center of Bounding Box
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # Purple Center Dot

        # Count Vehicles Crossing Line
        if limits[0] < cx < limits[2] and limits[1] - 40 < cy < limits[1] + 40:
            if obj_id not in totalCount:
                totalCount.append(obj_id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)  # Green Line after Count

    # Display Total Count
    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50), scale=2, thickness=3, offset=5)

    # Show Image
    cv2.imshow("Vehicle Counter", img)

    # Press 'q' to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

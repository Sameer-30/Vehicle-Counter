import cv2

# Load Video
cap = cv2.VideoCapture(r"C:\Users\LEGION\Desktop\Computer Vision\YOLO with web cam and video\Vehicle Counter\video.mp4")

# Callback function to get mouse click coordinates
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left-click event
        print(f"Clicked at: ({x}, {y})")  # Print coordinates

# Create window and set mouse callback
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", get_coordinates)

while True:
    success, frame = cap.read()
    if not success:
        break  # Stop if video ends

    cv2.imshow("Video", frame)  # Show frame

    # Press 'q' to exit
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

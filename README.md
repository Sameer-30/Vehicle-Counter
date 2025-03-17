# Vehicle Counter Project

The Vehicle Counter project is an advanced Python-based tool designed to introduce users to the intricacies of computer vision and object detection. This project implements a real-time vehicle counting system using YOLO (You Only Look Once) version 10L and OpenCV. It is ideal for intermediate to advanced users, students, and enthusiasts who want to delve deeper into image processing and computer vision concepts.

## Features

- **Real-Time Vehicle Detection:** Captures video from the default webcam and processes each frame to detect and count vehicles in real-time, providing immediate visual feedback.
- **Object Detection with YOLO:** Utilizes YOLO version 10L for accurate and efficient vehicle detection, ensuring high performance even in complex scenarios.
- **Mask Creation:** Applies a custom mask prepared in Canvas to isolate vehicles within the frame, allowing for precise detection and highlighting of the target objects.
- **Bounding Box Creation:** Generates a bounding box around detected vehicles, illustrating how to identify and highlight specific areas of interest within an image.
- **Interactive Display:** Displays processed frames with detected vehicles highlighted, providing a clear visualization of the vehicle counting process.

## Requirements

The requirements for this project are listed in the `requirements.txt` file. To install the necessary dependencies, run:

```bash
pip install -r requirements.txt

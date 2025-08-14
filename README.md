# Real-time Object Detection with Mask R-CNN

This project implements real-time object detection using Mask R-CNN with your webcam. It uses PyTorch and OpenCV to perform instance segmentation and object detection in real-time.

## Requirements

- Python 3.8 or higher
- PyTorch
- OpenCV
- Other dependencies listed in requirements.txt

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:
```bash
python real_time_detection.py
```

2. The program will:
   - Open your webcam
   - Display the video feed with object detection results
   - Show bounding boxes, masks, and labels for detected objects
   - Display FPS (Frames Per Second) in the top-left corner

3. Press 'q' to quit the program.

## Features

- Real-time object detection and instance segmentation
- Displays bounding boxes around detected objects
- Shows object labels and confidence scores
- Visualizes instance masks
- FPS counter
- Supports all COCO dataset classes

## Notes

- The model runs on CPU by default. If you have a CUDA-capable GPU, it will automatically use it.
- The confidence threshold is set to 0.5 by default. You can modify this in the code to adjust detection sensitivity.
- Performance may vary depending on your hardware specifications. 
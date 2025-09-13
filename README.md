# Real-time Object Detection with Edge Derivatives

This project explores the mathematical foundations of edge derivatives and their application to real-time object detection. It includes both a comprehensive mathematical paper and a practical implementation using PyTorch and OpenCV for instance segmentation and object detection in real-time.

## Project Components

### Mathematical Paper
- **LaTeX Source**: `edge_derivatives_paper.tex` - A comprehensive mathematical paper titled "Applying Edge Derivatives to Real-Time Object Detection"
- **PDF Document**: `edge_derivatives_paper.pdf` - The compiled PDF version of the mathematical paper
- **Sobel Example**: `sobel_edge_detection_example.png` - Visual demonstration of Sobel edge detection

The paper covers:
- Mathematical foundations of image derivatives and edge detection
- Sobel operators and gradient computation
- Laplacian operators for second-derivative edge detection
- OpenCV implementations and practical code examples
- Integration of edge detection with real-time object detection pipelines

### Implementation
- **Python Script**: `realtime_object_detection.py` - Real-time object detection implementation
- **Model**: `yolov8n-seg.pt` - YOLOv8 segmentation model weights

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

## Viewing the Mathematical Paper

### PDF Version
Simply open `edge_derivatives_paper.pdf` in any PDF viewer to read the complete mathematical paper.

### LaTeX Source
To compile the LaTeX source yourself:
```bash
pdflatex edge_derivatives_paper.tex
```

The paper includes:
- Mathematical derivations of edge detection algorithms
- Visual examples with the included `sobel_edge_detection_example.png`
- Complete OpenCV code implementations
- References to relevant academic sources

## Usage

### Real-time Object Detection
1. Run the script:
```bash
python realtime_object_detection.py
```

2. The program will:
   - Open your webcam
   - Display the video feed with object detection results
   - Show bounding boxes, masks, and labels for detected objects
   - Display FPS (Frames Per Second) in the top-left corner

3. Press 'q' to quit the program.

## Mathematical Concepts Covered

The paper explores several key mathematical concepts:

- **Image Gradients**: First partial derivatives ∂I/∂x and ∂I/∂y
- **Gradient Magnitude**: ||∇I|| = √((∂I/∂x)² + (∂I/∂y)²)
- **Sobel Operators**: Discrete convolution kernels for gradient approximation
- **Laplacian Operator**: ∇²I = ∂²I/∂x² + ∂²I/∂y² for second-derivative edge detection
- **Edge Detection Pipelines**: From raw images to binary masks and contours
- **OpenCV Integration**: Practical implementation of mathematical concepts

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
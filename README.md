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
- **Original Script**: `realtime_object_detection.py` - Real-time object detection implementation
- **Hybrid Script**: `hybrid_edge_yolo_detection.py` - Edge-enhanced YOLOv8 detection with mathematical integration
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
Simply open [edge_derivatives_paper.pdf](./edge_derivatives_paper.pdf) in any PDF viewer to read the complete mathematical paper.

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

### Real-time Object Detection (Original)
1. Run the original script:
```bash
python realtime_object_detection.py
```

2. The program will:
   - Open your webcam
   - Display the video feed with object detection results
   - Show bounding boxes, masks, and labels for detected objects
   - Display FPS (Frames Per Second) in the top-left corner

3. Press 'q' to quit the program.

### Hybrid Edge-YOLOv8 Detection (New)
1. Run the hybrid script:
```bash
python hybrid_edge_yolo_detection.py
```

2. The program will open three windows:
   - **Original YOLOv8**: Standard YOLOv8 detections with bounding boxes and masks
   - **Edge Detection**: Canny edge map showing detected edges
   - **Edge-Refined Detection**: Enhanced detections using edge information

3. Features:
   - Real-time edge detection using Canny algorithm
   - Boundary refinement combining YOLOv8 masks with edge contours
   - FPS counter on all windows
   - Mathematical integration of edge derivatives from the paper

4. Press 'q' to quit the program.

## Mathematical Concepts Covered

The paper explores several key mathematical concepts:

- **Image Gradients**: First partial derivatives ∂I/∂x and ∂I/∂y
- **Gradient Magnitude**: ||∇I|| = √((∂I/∂x)² + (∂I/∂y)²)
- **Sobel Operators**: Discrete convolution kernels for gradient approximation
- **Laplacian Operator**: ∇²I = ∂²I/∂x² + ∂²I/∂y² for second-derivative edge detection
- **Edge Detection Pipelines**: From raw images to binary masks and contours
- **OpenCV Integration**: Practical implementation of mathematical concepts

## Features

### Original Implementation (`realtime_object_detection.py`)
- Real-time object detection and instance segmentation
- Displays bounding boxes around detected objects
- Shows object labels and confidence scores
- Visualizes instance masks
- FPS counter
- Supports all COCO dataset classes

### Hybrid Edge-YOLOv8 Implementation (`hybrid_edge_yolo_detection.py`)
- **Edge-Enhanced Detection**: Combines YOLOv8 with Canny edge detection for refined object boundaries
- **Multiple Visualization Windows**: 
  - Original YOLOv8 detections
  - Canny edge map visualization
  - Edge-refined detection results
- **Mathematical Integration**: Implements edge derivative techniques from the paper
- **Real-time Performance**: Optimized threading architecture maintains high FPS
- **Boundary Refinement**: Uses edge contours to enhance YOLOv8 mask accuracy

## Mathematical Integration

The hybrid implementation bridges the gap between traditional computer vision (your paper) and modern deep learning (YOLOv8):

### Edge Derivative Techniques Applied
- **Gaussian Blur Preprocessing**: Reduces noise before edge detection (as described in the paper)
- **Canny Edge Detection**: Implements `cv2.Canny(gray, 100, 200)` for binary edge maps
- **Contour Extraction**: Uses `cv2.findContours()` to extract object boundaries from edge maps
- **Boundary Refinement**: Combines YOLOv8 masks with edge contours for enhanced accuracy

### Technical Implementation
The hybrid approach works by:
1. Running YOLOv8 detection on each frame
2. Applying Canny edge detection with Gaussian preprocessing
3. Finding contours within each detected object's bounding box
4. Refining YOLOv8 masks by intersecting with strong edge regions
5. Displaying both original and refined results for comparison

This demonstrates how traditional edge derivative mathematics can enhance modern deep learning approaches.

## Notes

- The model runs on CPU by default. If you have a CUDA-capable GPU, it will automatically use it.
- The confidence threshold is set to 0.5 by default. You can modify this in the code to adjust detection sensitivity.
- Performance may vary depending on your hardware specifications.
- The hybrid approach adds minimal computational overhead while providing enhanced boundary detection. 
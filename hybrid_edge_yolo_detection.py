import cv2
import numpy as np
import time
from ultralytics import YOLO
from ultralytics import solutions
from threading import Thread, Lock
import queue

class HybridEdgeYOLO:
    def __init__(self):
        # Load YOLOv11 model
        self.model = YOLO('yolo11n-seg.pt')  # Using nano model for maximum speed
        
        # Initialize Speed Estimator
        self.speed_estimator = solutions.SpeedEstimator(
            model='yolo11n-seg.pt',
            fps=30.0,
            max_hist=5,
            meter_per_pixel=0.05,  # Adjust based on camera setup
            max_speed=120,  # km/h
            tracker='botsort.yaml',
            conf=0.3,
            iou=0.5,
            classes=[2, 3, 5, 7],  # Focus on vehicles: car, motorcycle, bus, truck
            verbose=True,
            show=False,  # We'll handle our own display
            show_conf=True,
            show_labels=True
        )
        
        # COCO classes
        self.COCO_CLASSES = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Threading setup
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.latest_result = None
        self.latest_speed_result = None
        self.lock = Lock()
        
        # Edge detection parameters
        self.canny_low = 100
        self.canny_high = 200
        self.gaussian_kernel = (3, 3)
        
        # Speed tracking parameters
        self.speed_enabled = True
        self.calibration_mode = False
        
    def preprocess_frame(self, frame):
        """Minimal preprocessing for maximum speed"""
        # Resize to 640x640 (YOLO's optimal size)
        return cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    
    def apply_edge_detection(self, frame):
        """Apply Canny edge detection with Gaussian blur preprocessing"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise (as per paper)
        blurred = cv2.GaussianBlur(gray, self.gaussian_kernel, 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        return edges, gray
    
    def refine_mask_with_edges(self, mask, edges, bbox):
        """Refine YOLOv8 mask using edge information within bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within bounds
        h, w = mask.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract region of interest
        roi_edges = edges[y1:y2, x1:x2]
        roi_mask = mask[y1:y2, x1:x2]
        
        if roi_edges.size == 0 or roi_mask.size == 0:
            return mask
        
        # Ensure both arrays have the same shape and type
        if roi_edges.shape != roi_mask.shape:
            roi_edges = cv2.resize(roi_edges, (roi_mask.shape[1], roi_mask.shape[0]))
        
        # Convert to same data type (uint8 for findContours)
        roi_edges = roi_edges.astype(np.uint8)
        
        # Find contours in the edge map
        contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        # Create refined mask by combining original mask with edge information
        refined_roi = roi_mask.copy()
        
        # For each contour, enhance the mask if it aligns with the original mask
        for contour in contours:
            # Create a temporary mask for this contour
            contour_mask = np.zeros_like(roi_edges, dtype=np.uint8)
            cv2.fillPoly(contour_mask, [contour], 255)
            
            # Convert contour mask to match roi_mask data type
            contour_mask = contour_mask.astype(roi_mask.dtype)
            
            # Find intersection with original mask
            intersection = cv2.bitwise_and(contour_mask, roi_mask)
            
            # If there's significant overlap, enhance the mask
            if np.sum(intersection) > 0.1 * np.sum(roi_mask):
                # Dilate the contour slightly and add to refined mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                dilated_contour = cv2.dilate(contour_mask, kernel, iterations=1)
                refined_roi = cv2.bitwise_or(refined_roi, dilated_contour)
        
        # Update the original mask
        mask[y1:y2, x1:x2] = refined_roi
        
        return mask
    
    def inference_thread(self):
        """Separate thread for inference"""
        while True:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if frame is None:
                    break
                
                # Run YOLOv11 inference
                results = self.model(frame, conf=0.5, verbose=False)[0]
                
                # Process results
                processed_results = []
                if results.masks is not None:
                    for box, mask, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                                  results.masks.data.cpu().numpy(),
                                                  results.boxes.cls.cpu().numpy(),
                                                  results.boxes.conf.cpu().numpy()):
                        processed_results.append((box, mask, int(cls), conf))
                
                # Run speed estimation if enabled
                speed_result = None
                if self.speed_enabled:
                    try:
                        speed_result = self.speed_estimator(frame)
                    except Exception as e:
                        print(f"Speed estimation error: {e}")
                
                with self.lock:
                    self.latest_result = processed_results
                    self.latest_speed_result = speed_result
                
                # Clear queue to prevent lag
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                        
            except queue.Empty:
                continue
    
    def draw_results(self, frame, results, edges=None, refined_results=None, speed_result=None):
        """Draw detection results on frame with different visualization modes"""
        if not results:
            return frame, frame, frame
        
        h, w = frame.shape[:2]
        scale_x = w / 640
        scale_y = h / 640
        
        # Create different visualization frames
        original_frame = frame.copy()
        edge_frame = frame.copy()
        refined_frame = frame.copy()
        
        for i, (box, mask, cls, conf) in enumerate(results):
            # Scale coordinates
            x1, y1, x2, y2 = map(int, box * [scale_x, scale_y, scale_x, scale_y])
            
            # Scale mask to frame size
            mask_scaled = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Draw on original frame
            mask_color = np.array([255, 0, 0], dtype=frame.dtype)
            original_frame[mask_scaled > 0.5] = original_frame[mask_scaled > 0.5] * 0.6 + mask_color * 0.4
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw label
            label = f"{self.COCO_CLASSES[cls]} {conf:.2f}"
            cv2.putText(original_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw on edge frame (show edges + bounding boxes)
            if edges is not None:
                edge_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(edge_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(edge_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw refined results if available
            if refined_results and i < len(refined_results):
                refined_mask = refined_results[i]
                refined_mask_scaled = cv2.resize(refined_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                refined_color = np.array([0, 255, 255], dtype=frame.dtype)  # Cyan for refined
                refined_frame[refined_mask_scaled > 0.5] = refined_frame[refined_mask_scaled > 0.5] * 0.6 + refined_color * 0.4
                cv2.rectangle(refined_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(refined_frame, f"{label} (refined)", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Overlay speed information if available
        if speed_result is not None and hasattr(speed_result, 'plot_im'):
            # Use the speed estimator's visualization
            speed_frame = speed_result.plot_im.copy()
            # Resize to match our frame size
            speed_frame = cv2.resize(speed_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Overlay speed information on original frame
            if hasattr(speed_result, 'boxes') and speed_result.boxes is not None:
                # Extract speed information from tracking data
                if hasattr(speed_result.boxes, 'speed') and speed_result.boxes.speed is not None:
                    speeds = speed_result.boxes.speed.cpu().numpy()
                    if len(speeds) > 0:
                        avg_speed = np.mean(speeds)
                        speed_text = f"Avg Speed: {avg_speed:.1f} km/h"
                        cv2.putText(original_frame, speed_text, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add individual speed labels for each tracked vehicle
                if hasattr(speed_result.boxes, 'xyxy') and speed_result.boxes.xyxy is not None:
                    boxes = speed_result.boxes.xyxy.cpu().numpy()
                    if hasattr(speed_result.boxes, 'speed') and speed_result.boxes.speed is not None:
                        speeds = speed_result.boxes.speed.cpu().numpy()
                        track_ids = speed_result.boxes.id.cpu().numpy() if hasattr(speed_result.boxes, 'id') else None
                        
                        for i, (box, speed) in enumerate(zip(boxes, speeds)):
                            x1, y1, x2, y2 = map(int, box)
                            # Scale coordinates to match our frame size
                            x1, y1, x2, y2 = map(int, [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
                            
                            # Draw speed label above bounding box
                            speed_label = f"ID:{int(track_ids[i]) if track_ids is not None else i} {speed:.1f}km/h"
                            cv2.putText(original_frame, speed_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            
                            # Draw colored bounding box based on speed
                            color = (0, 255, 0) if speed < 50 else (0, 255, 255) if speed < 80 else (0, 0, 255)  # Green/Yellow/Red
                            cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 2)
            
            # Also overlay on refined frame
            if hasattr(speed_result, 'boxes') and speed_result.boxes is not None:
                if hasattr(speed_result.boxes, 'speed') and speed_result.boxes.speed is not None:
                    speeds = speed_result.boxes.speed.cpu().numpy()
                    if len(speeds) > 0:
                        avg_speed = np.mean(speeds)
                        speed_text = f"Avg Speed: {avg_speed:.1f} km/h"
                        cv2.putText(refined_frame, speed_text, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return original_frame, edge_frame, refined_frame
    
    def run(self):
        """Main loop"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Start inference thread
        inference_thread = Thread(target=self.inference_thread, daemon=True)
        inference_thread.start()
        
        print("Hybrid Edge-YOLOv11 Detection with Speed Estimation (Press 'q' to quit)")
        print("Windows: Original+Speed | Edges | Refined | Speed Only")
        print("Controls: 's' to toggle speed estimation, 'c' for calibration mode")
        
        frame_count = 0
        start_time = time.time()
        process_every = 1  # Process every frame
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Submit frame for processing
            if frame_count % process_every == 0:
                if self.frame_queue.empty():
                    processed_frame = self.preprocess_frame(frame)
                    try:
                        self.frame_queue.put_nowait(processed_frame)
                    except queue.Full:
                        pass
            
            # Get latest results
            with self.lock:
                current_result = self.latest_result
                current_speed_result = self.latest_speed_result
            
            # Apply edge detection
            edges, gray = self.apply_edge_detection(frame)
            
            # Refine results with edge information
            refined_results = None
            if current_result:
                refined_results = []
                for box, mask, cls, conf in current_result:
                    refined_mask = self.refine_mask_with_edges(mask.copy(), edges, box)
                    refined_results.append(refined_mask)
            
            # Draw results on different frames
            if current_result:
                original_frame, edge_frame, refined_frame = self.draw_results(
                    frame, current_result, edges, refined_results, current_speed_result
                )
            else:
                original_frame = frame.copy()
                edge_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                refined_frame = frame.copy()
            
            # Display FPS and status
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - start_time)
                start_time = time.time()
                fps_text = f"FPS: {fps:.0f}"
                cv2.putText(original_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(edge_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(refined_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Add status indicators
            status_text = f"Speed: {'ON' if self.speed_enabled else 'OFF'}"
            cv2.putText(original_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.speed_enabled else (0, 0, 255), 2)
            
            # Add speed legend
            if self.speed_enabled:
                legend_y = 90
                cv2.putText(original_frame, "Speed Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(original_frame, "Green: <50 km/h", (10, legend_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(original_frame, "Yellow: 50-80 km/h", (10, legend_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(original_frame, "Red: >80 km/h", (10, legend_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Create speed-only visualization window
            speed_only_frame = frame.copy()
            if current_speed_result is not None and hasattr(current_speed_result, 'plot_im'):
                speed_only_frame = current_speed_result.plot_im.copy()
                h, w = frame.shape[:2]
                speed_only_frame = cv2.resize(speed_only_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Display multiple windows
            cv2.imshow('Original YOLOv11 + Speed', original_frame)
            cv2.imshow('Edge Detection', edge_frame)
            cv2.imshow('Edge-Refined Detection', refined_frame)
            cv2.imshow('Speed Estimation Only', speed_only_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.speed_enabled = not self.speed_enabled
                print(f"Speed estimation {'enabled' if self.speed_enabled else 'disabled'}")
            elif key == ord('c'):
                self.calibration_mode = not self.calibration_mode
                print(f"Calibration mode {'enabled' if self.calibration_mode else 'disabled'}")
                if self.calibration_mode:
                    print("Click on a known object to calibrate meter_per_pixel ratio")
        
        # Cleanup
        self.frame_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = HybridEdgeYOLO()
    detector.run()

if __name__ == "__main__":
    main()

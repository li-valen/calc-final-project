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
        
        # Initialize Instance Segmentation and Tracking
        self.instance_segmenter = solutions.InstanceSegmentation(
            model='yolo11n-seg.pt',
            tracker='botsort.yaml',
            conf=0.3,
            iou=0.5,
            classes=[0],  # Focus on people (class 0)
            verbose=True,
            show=False,  # We'll handle our own display
            show_conf=True,
            show_labels=True
        )
        
        # Initialize Speed Estimator for people
        self.speed_estimator = solutions.SpeedEstimator(
            model='yolo11n-seg.pt',
            tracker='botsort.yaml',
            conf=0.3,
            iou=0.5,
            classes=[0],  # Focus on people (class 0)
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
        self.latest_tracking_result = None
        self.latest_speed_result = None
        self.lock = Lock()
        
        # Edge detection parameters
        self.canny_low = 100
        self.canny_high = 200
        self.gaussian_kernel = (3, 3)
        
        # Tracking parameters
        self.tracking_enabled = True
        self.speed_enabled = True
        self.calibration_mode = False
        
        # Speed estimation parameters
        self.meter_per_pixel = 0.1  # Default calibration - adjust as needed
        self.fps = 30  # Camera FPS for speed calculation
        
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
                
                # Run instance segmentation and tracking if enabled
                tracking_result = None
                if self.tracking_enabled:
                    try:
                        tracking_result = self.instance_segmenter(frame)
                    except Exception as e:
                        print(f"Tracking error: {e}")
                
                # Run speed estimation if enabled
                speed_result = None
                if self.speed_enabled:
                    try:
                        speed_result = self.speed_estimator(frame)
                    except Exception as e:
                        print(f"Speed estimation error: {e}")
                
                with self.lock:
                    self.latest_result = processed_results
                    self.latest_tracking_result = tracking_result
                    self.latest_speed_result = speed_result
                
                # Clear queue to prevent lag
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                        
            except queue.Empty:
                continue
    
    def draw_results(self, frame, results, edges=None, refined_results=None, tracking_result=None, speed_result=None):
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
        
        # Overlay tracking information if available
        if tracking_result is not None and hasattr(tracking_result, 'plot_im'):
            # Use the tracking visualization
            tracking_frame = tracking_result.plot_im.copy()
            # Resize to match our frame size
            tracking_frame = cv2.resize(tracking_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Overlay tracking information on original frame
            if hasattr(tracking_result, 'boxes') and tracking_result.boxes is not None:
                # Extract tracking information
                if hasattr(tracking_result.boxes, 'id') and tracking_result.boxes.id is not None:
                    track_ids = tracking_result.boxes.id.cpu().numpy()
                    boxes = tracking_result.boxes.xyxy.cpu().numpy()
                    confidences = tracking_result.boxes.conf.cpu().numpy()
                    
                    for i, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confidences)):
                        x1, y1, x2, y2 = map(int, box)
                        # Scale coordinates to match our frame size
                        x1, y1, x2, y2 = map(int, [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
                        
                        # Draw tracking ID and confidence above bounding box
                        track_label = f"ID:{int(track_id)} {conf:.2f}"
                        cv2.putText(original_frame, track_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Draw colored bounding box based on track ID
                        color = self.get_track_color(int(track_id))
                        cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 2)
            
            # Also overlay on refined frame
            if hasattr(tracking_result, 'boxes') and tracking_result.boxes is not None:
                if hasattr(tracking_result.boxes, 'id') and tracking_result.boxes.id is not None:
                    track_ids = tracking_result.boxes.id.cpu().numpy()
                    if len(track_ids) > 0:
                        track_count = len(track_ids)
                        track_text = f"People Tracked: {track_count}"
                        cv2.putText(refined_frame, track_text, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Overlay speed information if available
        if speed_result is not None and hasattr(speed_result, 'boxes') and speed_result.boxes is not None:
            if hasattr(speed_result.boxes, 'speed') and speed_result.boxes.speed is not None:
                speeds = speed_result.boxes.speed.cpu().numpy()
                boxes = speed_result.boxes.xyxy.cpu().numpy()
                confidences = speed_result.boxes.conf.cpu().numpy()
                
                if len(speeds) > 0:
                    # Calculate average speed
                    avg_speed = np.mean(speeds)
                    speed_text = f"Avg Speed: {avg_speed:.2f} m/s"
                    cv2.putText(original_frame, speed_text, (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw individual speed labels
                    for i, (box, speed, conf) in enumerate(zip(boxes, speeds, confidences)):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Draw speed next to confidence score
                        speed_label = f"Speed: {speed:.2f} m/s"
                        cv2.putText(original_frame, speed_label, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        
                        # Draw colored bounding box based on speed (faster = redder)
                        speed_color = self.get_speed_color(speed)
                        cv2.rectangle(original_frame, (x1, y1), (x2, y2), speed_color, 2)
        
        return original_frame, edge_frame, refined_frame
    
    def get_track_color(self, track_id):
        """Generate consistent colors for track IDs"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]
        return colors[track_id % len(colors)]
    
    def get_speed_color(self, speed):
        """Generate colors based on speed (faster = redder)"""
        # Normalize speed to 0-1 range (assuming max speed of 5 m/s)
        normalized_speed = min(speed / 5.0, 1.0)
        
        # Create color gradient from green (slow) to red (fast)
        red = int(255 * normalized_speed)
        green = int(255 * (1 - normalized_speed))
        blue = 0
        
        return (blue, green, red)
    
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
        
        print("Hybrid Edge-YOLOv11 Detection with People Tracking & Speed Estimation (Press 'q' to quit)")
        print("Windows: Edge Detection | Refined Detection | Everything + Speed")
        print("Controls: 't' to toggle tracking, 's' to toggle speed, 'c' for calibration mode")
        
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
                current_tracking_result = self.latest_tracking_result
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
                    frame, current_result, edges, refined_results, current_tracking_result, current_speed_result
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
            status_text = f"Tracking: {'ON' if self.tracking_enabled else 'OFF'}"
            cv2.putText(original_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.tracking_enabled else (0, 0, 255), 2)
            
            speed_status_text = f"Speed: {'ON' if self.speed_enabled else 'OFF'}"
            cv2.putText(original_frame, speed_status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.speed_enabled else (0, 0, 255), 2)
            
            # Add tracking legend
            if self.tracking_enabled:
                legend_y = 120
                cv2.putText(original_frame, "Tracking Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(original_frame, "Each person gets unique color", (10, legend_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(original_frame, "ID: Track ID, Confidence", (10, legend_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Add speed legend
            if self.speed_enabled:
                speed_legend_y = legend_y + 80 if self.tracking_enabled else 120
                cv2.putText(original_frame, "Speed Legend:", (10, speed_legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(original_frame, "Green = Slow, Red = Fast", (10, speed_legend_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(original_frame, "Speed shown below confidence", (10, speed_legend_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Display three windows
            cv2.imshow('Edge Detection', edge_frame)
            cv2.imshow('Refined Detection', refined_frame)
            cv2.imshow('Everything + Speed', original_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                self.tracking_enabled = not self.tracking_enabled
                print(f"People tracking {'enabled' if self.tracking_enabled else 'disabled'}")
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

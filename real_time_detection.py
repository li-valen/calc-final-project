import cv2
import numpy as np
import time
from ultralytics import YOLO
from threading import Thread, Lock
import queue

class UltraFastYOLO:
    def __init__(self):
        # Load YOLOv8 model
        self.model = YOLO('yolov8n-seg.pt')  # Using nano model for maximum speed
        
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
        self.lock = Lock()
        
    def preprocess_frame(self, frame):
        """Minimal preprocessing for maximum speed"""
        # Resize to 640x640 (YOLO's optimal size)
        return cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    
    def inference_thread(self):
        """Separate thread for inference"""
        while True:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if frame is None:
                    break
                
                # Run inference
                results = self.model(frame, conf=0.5, verbose=False)[0]
                
                # Process results
                processed_results = []
                if results.masks is not None:
                    for box, mask, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                                  results.masks.data.cpu().numpy(),
                                                  results.boxes.cls.cpu().numpy(),
                                                  results.boxes.conf.cpu().numpy()):
                        processed_results.append((box, mask, int(cls), conf))
                
                with self.lock:
                    self.latest_result = processed_results
                
                # Clear queue to prevent lag
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                        
            except queue.Empty:
                continue
    
    def draw_results(self, frame, results):
        """Draw detection results on frame"""
        if not results:
            return frame
        
        h, w = frame.shape[:2]
        scale_x = w / 640
        scale_y = h / 640
        
        for box, mask, cls, conf in results:
            # Scale coordinates
            x1, y1, x2, y2 = map(int, box * [scale_x, scale_y, scale_x, scale_y])
            
            # Draw mask
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[mask > 0.5] = frame[mask > 0.5] * 0.6 + np.array([255, 0, 0], dtype=frame.dtype) * 0.4
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw label
            label = f"{self.COCO_CLASSES[cls]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame
    
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
        
        print("Ultra-fast YOLOv8 Segmentation (Press 'q' to quit)")
        
        frame_count = 0
        start_time = time.time()
        process_every = 1  # Process every 2nd frame
        
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
            
            # Draw results
            if current_result:
                frame = self.draw_results(frame, current_result)
            
            # Display FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - start_time)
                start_time = time.time()
                cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow('Ultra-fast YOLOv8 Segmentation', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.frame_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = UltraFastYOLO()
    detector.run()

if __name__ == "__main__":
    main()

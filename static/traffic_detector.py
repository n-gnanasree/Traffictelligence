import cv2
import numpy as np
from ultralytics import YOLO
import torch

class TrafficDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
    def detect_vehicles(self, frame):
        results = self.model(frame, verbose=False)
        vehicles = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls in self.vehicle_classes:
                        conf = float(box.conf[0])
                        if conf > 0.5:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            vehicles.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': conf,
                                'class': cls
                            })
        return vehicles
    
    def count_traffic(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        total_vehicles = 0
        
        while cap.read()[0]:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % 30 == 0:  # Process every 30th frame
                vehicles = self.detect_vehicles(frame)
                total_vehicles += len(vehicles)
            
            frame_count += 1
        
        cap.release()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration = frame_count / fps
        
        return {
            'total_vehicles': total_vehicles,
            'duration_minutes': duration / 60,
            'vehicles_per_minute': total_vehicles / (duration / 60) if duration > 0 else 0
        }

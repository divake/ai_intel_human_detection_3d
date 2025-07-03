#!/usr/bin/env python3
"""
YOLOv11 Detection Engine with Intel Hardware Optimizations

Features:
- YOLOv11 model loading and inference
- Intel Extension for PyTorch optimizations
- Multi-threading support
- Real-time performance tuning
"""

import cv2
import numpy as np
import torch
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False
    print("Info: Intel Extension for PyTorch not available. Using standard PyTorch.")

class YOLOv11Detector:
    """YOLOv11 detection engine optimized for Intel hardware"""
    
    def __init__(self, config: Dict):
        """Initialize YOLOv11 detector with Intel optimizations"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.inference_times = []
        self.detection_count = 0
        
        # Initialize model
        self.model = None
        self.device = self._setup_device()
        self.load_model()
        
        # Class names for COCO dataset (YOLOv11 default)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
    def _setup_device(self) -> str:
        """Setup optimal device for inference"""
        device = self.config.get('device', 'cpu')
        
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self.logger.info("✅ Using CUDA GPU for inference")
            elif IPEX_AVAILABLE:
                device = 'cpu'
                self.logger.info("✅ Using Intel Extension for PyTorch on CPU")
            else:
                device = 'cpu'
                self.logger.info("✅ Using standard PyTorch CPU")
        
        return device
        
    def load_model(self):
        """Load YOLOv11 model with optimizations"""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")
            
        model_path = self.config.get('model_path', 'yolov11n.pt')
        
        try:
            # Load model
            self.logger.info(f"🔄 Loading YOLOv11 model: {model_path}")
            self.model = YOLO(model_path)
            
            # Apply Intel optimizations if available
            if IPEX_AVAILABLE and self.device == 'cpu' and self.config.get('intel_optimization', True):
                self.logger.info("🚀 Applying Intel Extension for PyTorch optimizations...")
                self.model.model = ipex.optimize(self.model.model)
                
            self.logger.info("✅ YOLOv11 model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            # Fallback to downloading default model
            self.logger.info("🔄 Downloading default YOLOv11n model...")
            self.model = YOLO('yolov11n.pt')
            
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run detection on image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2]
            - confidence: detection confidence
            - class: class name
            - class_id: class index
        """
        if self.model is None:
            self.logger.error("❌ Model not loaded")
            return []
            
        start_time = time.time()
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.config.get('confidence_threshold', 0.5),
                iou=self.config.get('iou_threshold', 0.45),
                max_det=self.config.get('max_detections', 50),
                verbose=False
            )
            
            detections = []
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        if class_id < len(self.class_names):
                            detection = {
                                'bbox': box.tolist(),
                                'confidence': float(conf),
                                'class': self.class_names[class_id],
                                'class_id': int(class_id)
                            }
                            detections.append(detection)
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.detection_count += len(detections)
            
            # Keep only last 100 measurements
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
                
            return detections
            
        except Exception as e:
            self.logger.error(f"❌ Detection failed: {e}")
            return []
            
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
            
        avg_time = np.mean(self.inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'fps': fps,
            'total_detections': self.detection_count,
            'samples': len(self.inference_times)
        }
        
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection results on image"""
        vis_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0) if class_name == 'person' else (255, 0, 0)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
        return vis_image
        
    def warmup(self, image_shape: Tuple[int, int, int] = (640, 480, 3)):
        """Warm up the model with dummy inference"""
        dummy_image = np.random.randint(0, 255, image_shape, dtype=np.uint8)
        self.logger.info("🔥 Warming up YOLOv11 model...")
        
        # Run several warmup inferences
        for i in range(5):
            _ = self.detect(dummy_image)
            
        self.logger.info("✅ Model warmup completed")
        
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            
if __name__ == "__main__":
    # Test the detector
    config = {
        'model_path': 'yolov11n.pt',
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'device': 'cpu',
        'intel_optimization': True,
        'max_detections': 50
    }
    
    detector = YOLOv11Detector(config)
    
    # Test with dummy image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = detector.detect(test_image)
    
    print(f"✅ Detector test completed. Found {len(detections)} detections")
    print(f"📊 Performance: {detector.get_performance_stats()}")
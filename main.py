#!/usr/bin/env python3
"""
Real-Time 3D Human Detection System
Main application entry point

Features:
- YOLOv11 person detection
- Real vs photo classification using depth
- Multi-person tracking with unique IDs
- 3D position estimation and speed calculation
- Posture classification
- Real-time 3D visualization
"""

import cv2
import numpy as np
import time
import argparse
import logging
from pathlib import Path
import yaml

# Import our custom modules
from model.yolo_detector import YOLOv11Detector
from utils.motion_tracker import MultiPersonTracker
from utils.photo_judge import RealPersonJudge
from utils.posture_classification import PostureClassifier
from utils.robust_3d_estimation import Robust3DEstimator
from utils.realsense_manager import RealSenseManager
from utils.visualization import RealTime3DVisualizer

class HumanDetection3D:
    def __init__(self, config_path="config/"):
        """Initialize the 3D Human Detection System"""
        self.config_path = Path(config_path)
        self.setup_logging()
        self.load_configs()
        self.initialize_components()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # Detection statistics
        self.stats = {
            'total_detections': 0,
            'real_people': 0,
            'fake_detections': 0,
            'tracked_ids': set(),
            'avg_distance': 0,
            'avg_speed': 0
        }
        
    def setup_logging(self):
        """Configure logging for the application"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('outputs/logs/detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_configs(self):
        """Load configuration files"""
        try:
            # Load camera configuration
            with open(self.config_path / "camera_config.yaml", 'r') as f:
                self.camera_config = yaml.safe_load(f)
                
            # Load model configuration  
            with open(self.config_path / "model_config.yaml", 'r') as f:
                self.model_config = yaml.safe_load(f)
                
            # Load tracking configuration
            with open(self.config_path / "tracking_config.yaml", 'r') as f:
                self.tracking_config = yaml.safe_load(f)
                
            self.logger.info("✅ Configuration files loaded successfully")
            
        except FileNotFoundError as e:
            self.logger.error(f"❌ Configuration file not found: {e}")
            self.create_default_configs()
            
    def create_default_configs(self):
        """Create default configuration files if they don't exist"""
        self.logger.info("Creating default configuration files...")
        
        # Default camera config
        self.camera_config = {
            'width': 640,
            'height': 480,
            'fps': 30,
            'enable_rgb': True,
            'enable_depth': True,
            'enable_imu': False,
            'depth_units': 0.001,
            'laser_power': 150,
            'visual_preset': 'medium_density'
        }
        
        # Default model config
        self.model_config = {
            'model_path': 'data/models/yolov11n.pt',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'device': 'cpu',
            'intel_optimization': True,
            'max_detections': 50
        }
        
        # Default tracking config
        self.tracking_config = {
            'max_disappeared': 30,
            'max_distance': 100,
            'min_hits': 3,
            'iou_threshold': 0.3,
            'speed_smoothing': 0.7,
            'position_smoothing': 0.8
        }
        
    def initialize_components(self):
        """Initialize all system components"""
        self.logger.info("🚀 Initializing system components...")
        
        try:
            # Initialize RealSense camera
            self.camera = RealSenseManager(self.camera_config)
            self.logger.info("✅ RealSense camera initialized")
            
            # Initialize YOLOv11 detector
            self.detector = YOLOv11Detector(self.model_config)
            self.logger.info("✅ YOLOv11 detector initialized")
            
            # Initialize real vs photo judge
            self.photo_judge = RealPersonJudge()
            self.logger.info("✅ Real person classifier initialized")
            
            # Initialize multi-person tracker
            self.tracker = MultiPersonTracker(self.tracking_config)
            self.logger.info("✅ Multi-person tracker initialized")
            
            # Initialize posture classifier
            self.posture_classifier = PostureClassifier()
            self.logger.info("✅ Posture classifier initialized")
            
            # Initialize 3D estimator
            self.estimator_3d = Robust3DEstimator(self.camera.get_intrinsics())
            self.logger.info("✅ 3D estimator initialized")
            
            # Initialize visualizer
            self.visualizer = RealTime3DVisualizer()
            self.logger.info("✅ 3D visualizer initialized")
            
            self.logger.info("🎯 All components initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            raise
            
    def process_frame(self, color_image, depth_image, timestamp):
        """Process a single frame through the entire pipeline"""
        results = {
            'timestamp': timestamp,
            'detections': [],
            'tracks': [],
            'real_people': 0,
            'fake_people': 0
        }
        
        try:
            # 1. Run YOLOv11 detection
            detections = self.detector.detect(color_image)
            
            if len(detections) > 0:
                # 2. Filter for real people using depth analysis
                real_detections = []
                for detection in detections:
                    if detection['class'] == 'person':  # Only process person detections
                        try:
                            is_real = self.photo_judge.is_real_person(
                                color_image, depth_image, detection['bbox']
                            )
                            
                            if is_real:
                                # 3. Calculate 3D position
                                position_3d = self.estimator_3d.estimate_position(
                                    detection['bbox'], depth_image
                                )
                                # Ensure position is a list, not numpy array
                                detection['3d_position'] = list(position_3d) if hasattr(position_3d, '__iter__') else [0.0, 0.0, 0.0]
                                
                                # 4. Classify posture
                                detection['posture'] = self.posture_classifier.classify(
                                    color_image, detection['bbox']
                                )
                                
                                real_detections.append(detection)
                                results['real_people'] += 1
                            else:
                                results['fake_people'] += 1
                                
                        except Exception as e:
                            self.logger.debug(f"Error processing detection: {e}")
                            # Skip this detection but continue with others
                            continue
                
                # 5. Update tracker with real detections
                if real_detections:
                    try:
                        tracks = self.tracker.update(real_detections, timestamp)
                        results['tracks'] = tracks
                        
                        # Update statistics
                        for track in tracks:
                            self.stats['tracked_ids'].add(track['id'])
                    except Exception as e:
                        self.logger.debug(f"Error in tracking: {e}")
                        results['tracks'] = []
                
                results['detections'] = real_detections
                self.stats['total_detections'] += len(real_detections)
                
        except Exception as e:
            self.logger.error(f"Error in frame processing: {e}")
            # Return empty results on error
        
        return results
        
    def draw_results(self, color_image, results):
        """Draw detection and tracking results on the image"""
        display_image = color_image.copy()
        
        # Draw tracks
        for track in results['tracks']:
            bbox = track['bbox']
            track_id = track['id']
            distance = track.get('distance', 0)
            speed = track.get('speed', 0)
            posture = track.get('posture', 'unknown')
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw track ID and info
            label = f"ID:{track_id} | {distance:.1f}m | {speed:.1f}m/s"
            cv2.putText(display_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw posture
            cv2.putText(display_image, f"Posture: {posture}", (x1, y2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw trajectory if available
            if 'trajectory' in track and len(track['trajectory']) > 1:
                points = track['trajectory']
                for i in range(1, len(points)):
                    pt1 = tuple(map(int, points[i-1][:2]))
                    pt2 = tuple(map(int, points[i][:2]))
                    cv2.line(display_image, pt1, pt2, (255, 0, 0), 2)
        
        # Draw fake detections (for debugging)
        if results['fake_people'] > 0:
            cv2.putText(display_image, f"Fake detections: {results['fake_people']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw statistics
        cv2.putText(display_image, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_image, f"Real People: {results['real_people']}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_image, f"Total IDs: {len(self.stats['tracked_ids'])}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return display_image
        
    def start_realtime_detection(self, show_3d=True):
        """Start real-time detection and tracking"""
        self.logger.info("🎬 Starting real-time detection...")
        
        try:
            self.camera.start()
            
            # Create display windows
            cv2.namedWindow('3D Human Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('3D Human Detection', 1280, 720)
            
            if show_3d:
                self.visualizer.initialize()
            
            while True:
                # Get frames from camera
                frames = self.camera.get_frames()
                if frames is None:
                    continue
                    
                color_image, depth_image = frames['color'], frames['depth']
                timestamp = time.time()
                
                # Process the frame
                results = self.process_frame(color_image, depth_image, timestamp)
                
                # Update 3D visualization
                if show_3d and results['tracks']:
                    self.visualizer.update(results['tracks'], depth_image)
                
                # Draw results on image
                display_image = self.draw_results(color_image, results)
                
                # Calculate FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed
                
                # Display
                cv2.imshow('3D Human Detection', display_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'outputs/screenshots/detection_{timestamp_str}.jpg', 
                               display_image)
                    self.logger.info(f"📸 Screenshot saved: detection_{timestamp_str}.jpg")
                elif key == ord('r'):
                    # Reset statistics
                    self.stats = {k: 0 if isinstance(v, (int, float)) else set() if isinstance(v, set) else v 
                                 for k, v in self.stats.items()}
                    self.logger.info("📊 Statistics reset")
                    
        except KeyboardInterrupt:
            self.logger.info("⏹️  Detection stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Error during detection: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources"""
        self.logger.info("🧹 Cleaning up resources...")
        
        if hasattr(self, 'camera'):
            self.camera.stop()
            
        cv2.destroyAllWindows()
        
        # Log final statistics
        self.logger.info("📊 Final Statistics:")
        self.logger.info(f"   Total detections: {self.stats['total_detections']}")
        self.logger.info(f"   Real people detected: {self.stats['real_people']}")
        self.logger.info(f"   Fake detections filtered: {self.stats['fake_detections']}")
        self.logger.info(f"   Unique IDs tracked: {len(self.stats['tracked_ids'])}")
        
def main():
    parser = argparse.ArgumentParser(description='Real-Time 3D Human Detection System')
    parser.add_argument('--config', type=str, default='config/',
                       help='Path to configuration directory')
    parser.add_argument('--no-3d', action='store_true',
                       help='Disable 3D visualization')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directories
    Path('outputs/logs').mkdir(parents=True, exist_ok=True)
    Path('outputs/screenshots').mkdir(parents=True, exist_ok=True)
    
    # Initialize and start the system
    system = HumanDetection3D(config_path=args.config)
    system.start_realtime_detection(show_3d=not args.no_3d)

if __name__ == "__main__":
    main()
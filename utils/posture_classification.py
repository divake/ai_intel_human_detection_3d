#!/usr/bin/env python3
"""
Posture Classification Module

Features:
- Standing, sitting, walking, crouching detection
- Body orientation analysis
- Movement pattern recognition
- Pose estimation integration
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Info: mediapipe not available. Install with: pip install mediapipe")

class PostureType(Enum):
    """Enumeration of posture types"""
    STANDING = "standing"
    SITTING = "sitting"
    WALKING = "walking"
    CROUCHING = "crouching"
    LYING = "lying"
    UNKNOWN = "unknown"

class PostureClassifier:
    """Classifies human postures using pose estimation and heuristics"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize posture classifier"""
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.config = config or {
            'pose_confidence': 0.5,
            'use_mediapipe': True,
            'temporal_smoothing': True,
            'smoothing_window': 5,
            'movement_threshold': 0.1,  # m/s for walking detection
        }
        
        # MediaPipe pose estimation
        self.mp_pose = None
        self.pose_detector = None
        
        if MEDIAPIPE_AVAILABLE and self.config['use_mediapipe']:
            self._init_mediapipe()
        
        # Temporal smoothing
        self.pose_history = {}  # track_id -> deque of poses
        self.posture_history = {}  # track_id -> deque of postures
        
        # Statistics
        self.stats = {
            'total_classifications': 0,
            'standing': 0,
            'sitting': 0,
            'walking': 0,
            'crouching': 0,
            'lying': 0,
            'unknown': 0
        }
        
        self.logger.info("🤸 Posture classifier initialized")
        
    def _init_mediapipe(self):
        """Initialize MediaPipe pose estimation"""
        try:
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=self.config['pose_confidence'],
                min_tracking_confidence=self.config['pose_confidence']
            )
            self.logger.info("✅ MediaPipe pose estimation initialized")
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to initialize MediaPipe: {e}")
            self.mp_pose = None
            self.pose_detector = None
            
    def classify(self, image: np.ndarray, bbox: List[float], 
                track_info: Optional[Dict] = None) -> str:
        """
        Classify posture for person in bounding box
        
        Args:
            image: RGB image
            bbox: Bounding box [x1, y1, x2, y2]
            track_info: Optional tracking information for temporal smoothing
            
        Returns:
            Posture classification string
        """
        self.stats['total_classifications'] += 1
        
        try:
            # Extract person region
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return PostureType.UNKNOWN.value
                
            person_roi = image[y1:y2, x1:x2]
            
            # Get pose landmarks
            pose_landmarks = self._extract_pose_landmarks(person_roi)
            
            # Classify posture based on landmarks and heuristics
            posture = self._classify_from_landmarks(pose_landmarks, bbox, track_info)
            
            # Apply temporal smoothing if tracking info available
            if track_info and self.config['temporal_smoothing']:
                posture = self._apply_temporal_smoothing(track_info['id'], posture)
                
            # Update statistics
            self.stats[posture] = self.stats.get(posture, 0) + 1
            
            return posture
            
        except Exception as e:
            self.logger.error(f"❌ Error in posture classification: {e}")
            return PostureType.UNKNOWN.value
            
    def _extract_pose_landmarks(self, person_roi: np.ndarray) -> Optional[Dict]:
        """Extract pose landmarks using MediaPipe"""
        
        if not self.pose_detector:
            return None
            
        try:
            # Convert BGR to RGB
            rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose_detector.process(rgb_roi)
            
            if not results.pose_landmarks:
                return None
                
            # Extract key landmarks
            landmarks = {}
            
            # Get important landmarks for posture classification
            important_landmarks = {
                'nose': self.mp_pose.PoseLandmark.NOSE,
                'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
                'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
                'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
                'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
                'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
                'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
                'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
                'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
                'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE
            }
            
            for name, landmark_id in important_landmarks.items():
                landmark = results.pose_landmarks.landmark[landmark_id]
                if landmark.visibility > 0.5:  # Only use visible landmarks
                    landmarks[name] = {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    }
                    
            return landmarks if landmarks else None
            
        except Exception as e:
            self.logger.debug(f"Failed to extract pose landmarks: {e}")
            return None
            
    def _classify_from_landmarks(self, landmarks: Optional[Dict], bbox: List[float], 
                                track_info: Optional[Dict]) -> str:
        """Classify posture from pose landmarks and heuristics"""
        
        # If no landmarks, use fallback heuristics
        if not landmarks:
            return self._classify_from_heuristics(bbox, track_info)
            
        try:
            # Calculate key angles and ratios
            analysis = self._analyze_pose_geometry(landmarks)
            
            # Classification rules based on pose analysis
            
            # Check for lying down (horizontal orientation)
            if analysis['is_horizontal']:
                return PostureType.LYING.value
                
            # Check for crouching (knees highly bent, low center of mass)
            if (analysis['knee_bend_avg'] > 0.7 and 
                analysis['torso_vertical_ratio'] < 0.6):
                return PostureType.CROUCHING.value
                
            # Check for sitting (hip-knee angle, torso upright)
            if (analysis['hip_knee_angle_avg'] < 140 and
                analysis['torso_vertical_ratio'] > 0.7):
                return PostureType.SITTING.value
                
            # Check for walking (movement + asymmetric leg positions)
            if track_info and self._is_walking(track_info, analysis):
                return PostureType.WALKING.value
                
            # Default to standing for upright postures
            if analysis['torso_vertical_ratio'] > 0.6:
                return PostureType.STANDING.value
                
            return PostureType.UNKNOWN.value
            
        except Exception as e:
            self.logger.debug(f"Error in landmark-based classification: {e}")
            return self._classify_from_heuristics(bbox, track_info)
            
    def _analyze_pose_geometry(self, landmarks: Dict) -> Dict:
        """Analyze geometric properties of the pose"""
        
        analysis = {
            'torso_vertical_ratio': 0,
            'knee_bend_avg': 0,
            'hip_knee_angle_avg': 180,
            'is_horizontal': False,
            'center_of_mass_y': 0.5,
            'limb_symmetry': 0
        }
        
        try:
            # Calculate torso vertical ratio (shoulder-hip vs total height)
            if all(k in landmarks for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                shoulder_y = (landmarks['left_shoulder']['y'] + landmarks['right_shoulder']['y']) / 2
                hip_y = (landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 2
                
                # Calculate vertical span
                head_y = landmarks.get('nose', {}).get('y', shoulder_y - 0.1)
                foot_y = max(
                    landmarks.get('left_ankle', {}).get('y', hip_y + 0.4),
                    landmarks.get('right_ankle', {}).get('y', hip_y + 0.4)
                )
                
                total_height = foot_y - head_y
                torso_height = hip_y - shoulder_y
                
                if total_height > 0:
                    analysis['torso_vertical_ratio'] = abs(torso_height) / total_height
                    
                # Check if horizontal (shoulders and hips at similar height)
                analysis['is_horizontal'] = abs(shoulder_y - hip_y) < 0.2
                
                # Center of mass estimation
                analysis['center_of_mass_y'] = (shoulder_y + hip_y) / 2
                
            # Calculate knee bend (using hip-knee-ankle angles)
            knee_bends = []
            hip_knee_angles = []
            
            for side in ['left', 'right']:
                hip_key = f'{side}_hip'
                knee_key = f'{side}_knee'
                ankle_key = f'{side}_ankle'
                
                if all(k in landmarks for k in [hip_key, knee_key, ankle_key]):
                    hip = landmarks[hip_key]
                    knee = landmarks[knee_key]
                    ankle = landmarks[ankle_key]
                    
                    # Calculate angle at knee
                    angle = self._calculate_angle(
                        [hip['x'], hip['y']],
                        [knee['x'], knee['y']],
                        [ankle['x'], ankle['y']]
                    )
                    
                    # Convert to bend ratio (0 = straight, 1 = fully bent)
                    bend_ratio = 1 - (angle / 180.0)
                    knee_bends.append(bend_ratio)
                    hip_knee_angles.append(angle)
                    
            if knee_bends:
                analysis['knee_bend_avg'] = np.mean(knee_bends)
                analysis['hip_knee_angle_avg'] = np.mean(hip_knee_angles)
                
        except Exception as e:
            self.logger.debug(f"Error in pose geometry analysis: {e}")
            
        return analysis
        
    def _calculate_angle(self, point1: List[float], point2: List[float], point3: List[float]) -> float:
        """Calculate angle between three points (point2 is the vertex)"""
        
        # Vectors from point2 to point1 and point3
        v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
        
    def _is_walking(self, track_info: Dict, pose_analysis: Dict) -> bool:
        """Determine if person is walking based on movement and pose"""
        
        # Check movement speed
        speed = track_info.get('speed', 0)
        if speed < self.config['movement_threshold']:
            return False
            
        # Check for alternating leg patterns (simplified)
        # In a real implementation, this would analyze leg phase differences
        knee_bend = pose_analysis.get('knee_bend_avg', 0)
        
        # Walking typically has moderate knee bend and movement
        return (0.2 < knee_bend < 0.6) and speed > 0.3
        
    def _classify_from_heuristics(self, bbox: List[float], track_info: Optional[Dict]) -> str:
        """Fallback classification using simple heuristics"""
        
        # Aspect ratio heuristics
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = height / width if width > 0 else 1.0
        
        # Movement-based classification
        if track_info:
            speed = track_info.get('speed', 0)
            
            # If moving fast, likely walking
            if speed > self.config['movement_threshold']:
                return PostureType.WALKING.value
                
        # Aspect ratio based classification
        if aspect_ratio < 1.2:  # Wide, short bbox
            return PostureType.SITTING.value
        elif aspect_ratio > 2.5:  # Very tall, narrow bbox
            return PostureType.STANDING.value
        else:
            return PostureType.STANDING.value  # Default assumption
            
    def _apply_temporal_smoothing(self, track_id: int, current_posture: str) -> str:
        """Apply temporal smoothing to reduce classification noise"""
        
        from collections import deque
        
        # Initialize history for new tracks
        if track_id not in self.posture_history:
            self.posture_history[track_id] = deque(maxlen=self.config['smoothing_window'])
            
        # Add current classification
        self.posture_history[track_id].append(current_posture)
        
        # Get most common posture in recent history
        if len(self.posture_history[track_id]) >= 3:
            from collections import Counter
            counter = Counter(self.posture_history[track_id])
            smoothed_posture = counter.most_common(1)[0][0]
            return smoothed_posture
        else:
            return current_posture
            
    def visualize_posture(self, image: np.ndarray, bbox: List[float], 
                         posture: str, landmarks: Optional[Dict] = None) -> np.ndarray:
        """Visualize posture classification results"""
        
        vis_image = image.copy()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Color based on posture
        color_map = {
            PostureType.STANDING.value: (0, 255, 0),    # Green
            PostureType.SITTING.value: (255, 255, 0),   # Yellow
            PostureType.WALKING.value: (0, 255, 255),   # Cyan
            PostureType.CROUCHING.value: (255, 0, 255), # Magenta
            PostureType.LYING.value: (255, 0, 0),       # Red
            PostureType.UNKNOWN.value: (128, 128, 128)  # Gray
        }
        
        color = color_map.get(posture, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw posture label
        cv2.putText(vis_image, posture.upper(), (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw pose landmarks if available
        if landmarks and self.mp_pose:
            self._draw_pose_landmarks(vis_image, landmarks, (x1, y1, x2, y2))
            
        return vis_image
        
    def _draw_pose_landmarks(self, image: np.ndarray, landmarks: Dict, bbox: Tuple[int, int, int, int]):
        """Draw pose landmarks on image"""
        
        x1, y1, x2, y2 = bbox
        roi_width = x2 - x1
        roi_height = y2 - y1
        
        # Draw key points
        for name, landmark in landmarks.items():
            x = int(x1 + landmark['x'] * roi_width)
            y = int(y1 + landmark['y'] * roi_height)
            
            # Draw point
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            
        # Draw skeleton connections
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee'),
            ('left_knee', 'left_ankle'),
            ('right_knee', 'right_ankle'),
            ('left_shoulder', 'left_elbow'),
            ('right_shoulder', 'right_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_elbow', 'right_wrist'),
        ]
        
        for point1, point2 in connections:
            if point1 in landmarks and point2 in landmarks:
                x1_conn = int(x1 + landmarks[point1]['x'] * roi_width)
                y1_conn = int(y1 + landmarks[point1]['y'] * roi_height)
                x2_conn = int(x1 + landmarks[point2]['x'] * roi_width)
                y2_conn = int(y1 + landmarks[point2]['y'] * roi_height)
                
                cv2.line(image, (x1_conn, y1_conn), (x2_conn, y2_conn), (255, 255, 255), 2)
                
    def get_statistics(self) -> Dict:
        """Get classification statistics"""
        return self.stats.copy()
        
    def reset_statistics(self):
        """Reset classification statistics"""
        self.stats = {k: 0 for k in self.stats.keys()}
        
if __name__ == "__main__":
    # Test the posture classifier
    classifier = PostureClassifier()
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_bbox = [200, 100, 400, 450]
    
    print("🧪 Testing Posture Classifier...")
    
    # Test classification
    posture = classifier.classify(test_image, test_bbox)
    print(f"✅ Posture classification: {posture}")
    
    # Test visualization
    vis_image = classifier.visualize_posture(test_image, test_bbox, posture)
    print(f"✅ Visualization completed")
    
    print(f"📊 Statistics: {classifier.get_statistics()}")
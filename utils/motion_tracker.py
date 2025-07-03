#!/usr/bin/env python3
"""
Multi-Person Tracking System

Features:
- Unique ID assignment for each person
- Trajectory tracking and prediction
- Speed and direction calculation
- Track persistence and re-identification
- Kalman filter for motion smoothing
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial.distance import cdist
from collections import deque

try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    print("Info: filterpy not available. Install with: pip install filterpy")

class PersonTrack:
    """Represents a single person track"""
    
    def __init__(self, track_id: int, detection: Dict, timestamp: float):
        """Initialize person track"""
        self.id = track_id
        self.bbox = detection['bbox']
        self.confidence = detection.get('confidence', 0.0)
        self.position_3d = detection.get('3d_position', [0, 0, 0])
        
        # Timestamps
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.last_update = timestamp
        
        # Track state
        self.hits = 1  # Number of detections
        self.time_since_update = 0
        self.state = 'tentative'  # tentative, confirmed, deleted
        
        # Trajectory and motion
        self.trajectory = deque(maxlen=50)  # Last 50 positions
        self.trajectory.append([*self.position_3d, timestamp])
        
        self.velocity_3d = [0, 0, 0]
        self.speed = 0.0
        self.direction = 0.0
        
        # Kalman filter for position smoothing
        self.kalman = None
        if KALMAN_AVAILABLE:
            self._init_kalman_filter()
            
        # Additional attributes
        self.posture = detection.get('posture', 'unknown')
        self.distance = np.linalg.norm(self.position_3d) if self.position_3d else 0
        
    def _init_kalman_filter(self):
        """Initialize Kalman filter for position tracking"""
        # State: [x, y, z, vx, vy, vz] (position + velocity)
        self.kalman = KalmanFilter(dim_x=6, dim_z=3)
        
        # State transition matrix (constant velocity model)
        dt = 1/30.0  # Assume 30 FPS
        self.kalman.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we observe position only)
        self.kalman.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance
        self.kalman.Q *= 0.1
        
        # Measurement noise covariance
        self.kalman.R *= 0.5
        
        # Initial state covariance
        self.kalman.P *= 100
        
        # Initialize state
        if self.position_3d and len(self.position_3d) >= 3:
            self.kalman.x[:3] = np.array(self.position_3d).reshape(-1)
        
    def update(self, detection: Dict, timestamp: float):
        """Update track with new detection"""
        self.bbox = detection['bbox']
        self.confidence = detection.get('confidence', 0.0)
        self.position_3d = detection.get('3d_position', [0, 0, 0])
        self.posture = detection.get('posture', 'unknown')
        
        self.last_seen = timestamp
        self.last_update = timestamp
        self.hits += 1
        self.time_since_update = 0
        
        # Update Kalman filter
        if self.kalman and self.position_3d:
            self.kalman.predict()
            # Ensure position is correct shape for Kalman filter
            pos_array = np.array(self.position_3d).reshape(-1)
            if len(pos_array) >= 3:
                self.kalman.update(pos_array[:3])
            
            # Get smoothed position and velocity
            smoothed_pos = self.kalman.x[:3].copy()
            self.velocity_3d = self.kalman.x[3:6].copy()
            
            # Update trajectory
            self.trajectory.append([*smoothed_pos, timestamp])
        else:
            # Fallback: simple trajectory tracking
            self.trajectory.append([*self.position_3d, timestamp])
            
            # Calculate velocity manually
            if len(self.trajectory) >= 2:
                dt = timestamp - self.trajectory[-2][3]
                if dt > 0:
                    dx = self.position_3d[0] - self.trajectory[-2][0]
                    dy = self.position_3d[1] - self.trajectory[-2][1]
                    dz = self.position_3d[2] - self.trajectory[-2][2]
                    self.velocity_3d = [dx/dt, dy/dt, dz/dt]
        
        # Calculate speed and direction
        self._update_motion_stats()
        
        # Update distance
        self.distance = np.linalg.norm(self.position_3d) if self.position_3d else 0
        
        # Update state
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'
            
    def predict(self, timestamp: float):
        """Predict track position for current timestamp"""
        self.time_since_update += 1
        
        if self.kalman:
            self.kalman.predict()
            predicted_pos = self.kalman.x[:3].copy()
        else:
            # Simple linear prediction
            if len(self.trajectory) >= 2:
                dt = timestamp - self.last_update
                predicted_pos = [
                    self.position_3d[0] + self.velocity_3d[0] * dt,
                    self.position_3d[1] + self.velocity_3d[1] * dt,
                    self.position_3d[2] + self.velocity_3d[2] * dt
                ]
            else:
                predicted_pos = self.position_3d
                
        return predicted_pos
        
    def _update_motion_stats(self):
        """Update speed and direction statistics"""
        if len(self.trajectory) < 2:
            return
            
        # Calculate speed from velocity
        self.speed = np.linalg.norm(self.velocity_3d)
        
        # Calculate direction (angle in XY plane)
        if abs(self.velocity_3d[0]) > 0.01 or abs(self.velocity_3d[1]) > 0.01:
            self.direction = np.arctan2(self.velocity_3d[1], self.velocity_3d[0]) * 180 / np.pi
            
    def get_track_info(self) -> Dict:
        """Get comprehensive track information"""
        return {
            'id': self.id,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'position_3d': self.position_3d,
            'velocity_3d': self.velocity_3d,
            'speed': self.speed,
            'direction': self.direction,
            'distance': self.distance,
            'posture': self.posture,
            'trajectory': list(self.trajectory),
            'hits': self.hits,
            'age': self.last_seen - self.first_seen,
            'time_since_update': self.time_since_update,
            'state': self.state
        }
        
    def is_valid(self, max_age: float, max_time_since_update: int) -> bool:
        """Check if track is still valid"""
        age = time.time() - self.last_seen
        return (age < max_age and 
                self.time_since_update < max_time_since_update and
                self.state != 'deleted')

class MultiPersonTracker:
    """Multi-person tracking system with unique ID assignment"""
    
    def __init__(self, config: Dict):
        """Initialize multi-person tracker"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Tracks management
        self.tracks: Dict[int, PersonTrack] = {}
        self.next_id = 1
        self.deleted_tracks = []
        
        # Configuration
        self.max_disappeared = config.get('max_disappeared', 30)
        self.max_distance = config.get('max_distance', 100)  # cm
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        self.max_age = config.get('max_age', 5.0)  # seconds
        
        # Performance tracking
        self.total_tracks = 0
        self.active_tracks = 0
        
        self.logger.info("🎯 Multi-person tracker initialized")
        
    def update(self, detections: List[Dict], timestamp: float) -> List[Dict]:
        """Update tracker with new detections"""
        
        # Predict existing tracks
        for track in self.tracks.values():
            track.predict(timestamp)
            
        # Associate detections with existing tracks
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
            detections, list(self.tracks.values())
        )
        
        # Update matched tracks
        for det_idx, trk_idx in matched:
            track_id = list(self.tracks.keys())[trk_idx]
            self.tracks[track_id].update(detections[det_idx], timestamp)
            
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_new_track(detections[det_idx], timestamp)
            
        # Mark unmatched tracks as missed
        for trk_idx in unmatched_trks:
            track_id = list(self.tracks.keys())[trk_idx]
            self.tracks[track_id].time_since_update += 1
            
        # Remove invalid tracks
        self._remove_invalid_tracks()
        
        # Update statistics
        self.active_tracks = len(self.tracks)
        
        # Return confirmed tracks
        return self._get_confirmed_tracks()
        
    def _associate_detections_to_tracks(self, detections: List[Dict], tracks: List[PersonTrack]) -> Tuple[List, List, List]:
        """Associate detections to existing tracks using distance and IoU"""
        
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
            
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
            
        # Calculate distance matrix (3D positions)
        det_positions = np.array([det.get('3d_position', [0, 0, 0]) for det in detections])
        trk_positions = np.array([trk.position_3d for trk in tracks])
        
        # Calculate Euclidean distances in 3D space
        distance_matrix = cdist(det_positions, trk_positions, metric='euclidean')
        
        # Calculate IoU matrix for bounding boxes
        iou_matrix = self._calculate_iou_matrix(detections, tracks)
        
        # Combine distance and IoU (weighted)
        # Convert distance to similarity (closer = higher similarity)
        max_dist = self.max_distance / 100.0  # Convert cm to m
        distance_similarity = 1.0 - np.clip(distance_matrix / max_dist, 0, 1)
        
        # Combined cost matrix (lower is better)
        cost_matrix = 1.0 - (0.6 * distance_similarity + 0.4 * iou_matrix)
        
        # Solve assignment problem using simple greedy approach
        matched_indices = []
        
        # Convert to list for processing
        cost_list = []
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                cost_list.append((cost_matrix[i, j], i, j))
                
        # Sort by cost (ascending)
        cost_list.sort()
        
        used_detections = set()
        used_tracks = set()
        
        for cost, det_idx, trk_idx in cost_list:
            # Check thresholds
            if (det_idx not in used_detections and 
                trk_idx not in used_tracks and
                distance_matrix[det_idx, trk_idx] < max_dist and
                iou_matrix[det_idx, trk_idx] > self.iou_threshold):
                
                matched_indices.append((det_idx, trk_idx))
                used_detections.add(det_idx)
                used_tracks.add(trk_idx)
                
        # Find unmatched detections and tracks
        unmatched_detections = [i for i in range(len(detections)) if i not in used_detections]
        unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
        
        return matched_indices, unmatched_detections, unmatched_tracks
        
    def _calculate_iou_matrix(self, detections: List[Dict], tracks: List[PersonTrack]) -> np.ndarray:
        """Calculate IoU matrix between detections and tracks"""
        
        iou_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, detection in enumerate(detections):
            det_bbox = detection['bbox']
            
            for j, track in enumerate(tracks):
                trk_bbox = track.bbox
                iou = self._calculate_bbox_iou(det_bbox, trk_bbox)
                iou_matrix[i, j] = iou
                
        return iou_matrix
        
    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0
            
        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
            
        return intersection / union
        
    def _create_new_track(self, detection: Dict, timestamp: float):
        """Create new track for unmatched detection"""
        
        track = PersonTrack(self.next_id, detection, timestamp)
        self.tracks[self.next_id] = track
        
        self.next_id += 1
        self.total_tracks += 1
        
        self.logger.debug(f"🆕 Created new track ID: {track.id}")
        
    def _remove_invalid_tracks(self):
        """Remove tracks that are no longer valid"""
        
        to_remove = []
        
        for track_id, track in self.tracks.items():
            if not track.is_valid(self.max_age, self.max_disappeared):
                to_remove.append(track_id)
                track.state = 'deleted'
                self.deleted_tracks.append(track)
                
        for track_id in to_remove:
            del self.tracks[track_id]
            self.logger.debug(f"🗑️  Removed track ID: {track_id}")
            
        # Keep only recent deleted tracks
        if len(self.deleted_tracks) > 100:
            self.deleted_tracks = self.deleted_tracks[-50:]
            
    def _get_confirmed_tracks(self) -> List[Dict]:
        """Get information for confirmed tracks only"""
        
        confirmed = []
        
        for track in self.tracks.values():
            if track.state == 'confirmed' or track.hits >= self.min_hits:
                confirmed.append(track.get_track_info())
                
        return confirmed
        
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        
        return {
            'total_tracks_created': self.total_tracks,
            'active_tracks': self.active_tracks,
            'deleted_tracks': len(self.deleted_tracks),
            'confirmed_tracks': len([t for t in self.tracks.values() if t.state == 'confirmed']),
            'tentative_tracks': len([t for t in self.tracks.values() if t.state == 'tentative']),
            'next_id': self.next_id
        }
        
    def reset(self):
        """Reset tracker state"""
        
        self.tracks.clear()
        self.deleted_tracks.clear()
        self.next_id = 1
        self.total_tracks = 0
        self.active_tracks = 0
        
        self.logger.info("🔄 Tracker reset")
        
if __name__ == "__main__":
    # Test the tracker
    config = {
        'max_disappeared': 30,
        'max_distance': 100,
        'min_hits': 3,
        'iou_threshold': 0.3,
        'max_age': 5.0
    }
    
    tracker = MultiPersonTracker(config)
    
    # Test with sample detections
    test_detections = [
        {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.8,
            '3d_position': [1.0, 0.5, 2.0],
            'posture': 'standing'
        },
        {
            'bbox': [300, 150, 400, 350],
            'confidence': 0.9,
            '3d_position': [2.0, 1.0, 2.5],
            'posture': 'walking'
        }
    ]
    
    timestamp = time.time()
    
    print("🧪 Testing Multi-Person Tracker...")
    
    # Update tracker multiple times
    for i in range(5):
        tracks = tracker.update(test_detections, timestamp + i * 0.033)  # 30 FPS
        print(f"Frame {i}: {len(tracks)} confirmed tracks")
        
        # Slightly modify detections to simulate movement
        for det in test_detections:
            det['3d_position'][0] += 0.1  # Move along X axis
            
    print(f"📊 Final statistics: {tracker.get_statistics()}")
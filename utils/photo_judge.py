#!/usr/bin/env python3
"""
Real vs Photo Detection using Depth Analysis

Features:
- Depth-based authenticity verification
- Prevents false positives from screens/photos
- Configurable depth thresholds and analysis
- Statistical depth analysis within bounding boxes
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional

class RealPersonJudge:
    """Distinguishes real people from photos/screens using depth analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize real person classifier"""
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.config = config or {
            'min_depth_mm': 400,        # Minimum depth (40cm)
            'max_depth_mm': 8000,       # Maximum depth (8m)
            'min_valid_pixels': 0.3,    # Minimum 30% valid depth pixels
            'depth_variance_threshold': 50,  # Minimum depth variance
            'edge_depth_ratio': 0.7,    # Edge pixels depth consistency
            'screen_detection_enabled': True,
            'screen_depth_threshold': 100,  # <10cm likely a screen
        }
        
        # Statistics tracking
        self.stats = {
            'total_classifications': 0,
            'real_detections': 0,
            'fake_detections': 0,
            'screen_detections': 0,
            'invalid_depth_rejections': 0
        }
        
        self.logger.info("🔍 Real person classifier initialized")
        
    def is_real_person(self, color_image: np.ndarray, depth_image: np.ndarray, 
                      bbox: List[float]) -> bool:
        """
        Determine if detected person is real or fake using depth analysis
        
        Args:
            color_image: RGB image
            depth_image: Depth image (uint16, in mm)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            True if real person, False if photo/screen
        """
        self.stats['total_classifications'] += 1
        
        try:
            # Extract bounding box region
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(depth_image.shape[1], x2)
            y2 = min(depth_image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                self.logger.warning("⚠️  Invalid bounding box")
                return False
                
            # Extract depth region
            depth_roi = depth_image[y1:y2, x1:x2]
            
            # Perform depth analysis
            analysis_result = self._analyze_depth_region(depth_roi)
            
            # Make classification decision
            is_real = self._classify_from_analysis(analysis_result)
            
            # Update statistics
            if is_real:
                self.stats['real_detections'] += 1
            else:
                self.stats['fake_detections'] += 1
                
                # Categorize fake detection type
                if analysis_result['avg_depth'] < self.config['screen_depth_threshold']:
                    self.stats['screen_detections'] += 1
                elif analysis_result['valid_pixel_ratio'] < self.config['min_valid_pixels']:
                    self.stats['invalid_depth_rejections'] += 1
                    
            return is_real
            
        except Exception as e:
            self.logger.error(f"❌ Error in real person classification: {e}")
            return False
            
    def _analyze_depth_region(self, depth_roi: np.ndarray) -> Dict:
        """Analyze depth characteristics within region"""
        
        # Filter valid depth values
        valid_mask = (depth_roi > 0) & (depth_roi < 65535)
        valid_depths = depth_roi[valid_mask]
        
        total_pixels = depth_roi.size
        valid_pixels = len(valid_depths)
        valid_pixel_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0
        
        analysis = {
            'total_pixels': total_pixels,
            'valid_pixels': valid_pixels,
            'valid_pixel_ratio': valid_pixel_ratio,
            'avg_depth': 0,
            'median_depth': 0,
            'depth_std': 0,
            'depth_range': 0,
            'edge_consistency': 0,
            'depth_histogram': None
        }
        
        if valid_pixels > 0:
            analysis['avg_depth'] = float(np.mean(valid_depths))
            analysis['median_depth'] = float(np.median(valid_depths))
            analysis['depth_std'] = float(np.std(valid_depths))
            analysis['depth_range'] = float(np.max(valid_depths) - np.min(valid_depths))
            
            # Analyze edge consistency
            analysis['edge_consistency'] = self._calculate_edge_consistency(depth_roi, valid_mask)
            
            # Create depth histogram
            analysis['depth_histogram'] = self._create_depth_histogram(valid_depths)
            
        return analysis
        
    def _calculate_edge_consistency(self, depth_roi: np.ndarray, valid_mask: np.ndarray) -> float:
        """Calculate depth consistency near edges (real people have depth gradients)"""
        
        if depth_roi.size < 100:  # Too small for edge analysis
            return 0.5
            
        try:
            # Create edge mask (border pixels)
            h, w = depth_roi.shape
            edge_mask = np.zeros_like(depth_roi, dtype=bool)
            
            # Top and bottom edges
            edge_mask[0:2, :] = True
            edge_mask[-2:, :] = True
            
            # Left and right edges  
            edge_mask[:, 0:2] = True
            edge_mask[:, -2:] = True
            
            # Get edge and center regions
            edge_valid = valid_mask & edge_mask
            center_valid = valid_mask & (~edge_mask)
            
            if not np.any(edge_valid) or not np.any(center_valid):
                return 0.5
                
            edge_depths = depth_roi[edge_valid]
            center_depths = depth_roi[center_valid]
            
            if len(edge_depths) == 0 or len(center_depths) == 0:
                return 0.5
                
            # Calculate depth difference between edge and center
            edge_mean = np.mean(edge_depths)
            center_mean = np.mean(center_depths)
            
            # Real people typically have depth variation
            # Photos/screens have uniform depth
            depth_diff = abs(edge_mean - center_mean)
            
            # Normalize to 0-1 range
            consistency = min(depth_diff / 100.0, 1.0)  # 100mm = full consistency
            
            return consistency
            
        except Exception:
            return 0.5
            
    def _create_depth_histogram(self, valid_depths: np.ndarray) -> np.ndarray:
        """Create depth histogram for analysis"""
        
        if len(valid_depths) == 0:
            return np.zeros(50)
            
        # Create histogram with 50 bins
        hist, _ = np.histogram(valid_depths, bins=50, range=(0, 10000))
        
        # Normalize
        hist = hist.astype(float)
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
            
        return hist
        
    def _classify_from_analysis(self, analysis: Dict) -> bool:
        """Make classification decision based on depth analysis"""
        
        # Check 1: Sufficient valid depth pixels
        if analysis['valid_pixel_ratio'] < self.config['min_valid_pixels']:
            return False
            
        # Check 2: Depth within reasonable range
        avg_depth = analysis['avg_depth']
        if (avg_depth < self.config['min_depth_mm'] or 
            avg_depth > self.config['max_depth_mm']):
            return False
            
        # Check 3: Screen detection (very close, uniform depth)
        if (self.config['screen_detection_enabled'] and 
            avg_depth < self.config['screen_depth_threshold']):
            return False
            
        # Check 4: Depth variance (real people have depth variation)
        if analysis['depth_std'] < self.config['depth_variance_threshold']:
            return False
            
        # Check 5: Edge consistency (real people have depth gradients)
        if analysis['edge_consistency'] < self.config['edge_depth_ratio']:
            return False
            
        # All checks passed - likely a real person
        return True
        
    def visualize_depth_analysis(self, color_image: np.ndarray, depth_image: np.ndarray, 
                                bbox: List[float], is_real: bool) -> np.ndarray:
        """Visualize depth analysis results"""
        
        vis_image = color_image.copy()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box with color indicating real/fake
        color = (0, 255, 0) if is_real else (0, 0, 255)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = "REAL" if is_real else "FAKE"
        cv2.putText(vis_image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Extract and analyze depth region
        depth_roi = depth_image[y1:y2, x1:x2]
        analysis = self._analyze_depth_region(depth_roi)
        
        # Add analysis info
        info_lines = [
            f"Avg Depth: {analysis['avg_depth']:.0f}mm",
            f"Valid: {analysis['valid_pixel_ratio']:.1%}",
            f"Std: {analysis['depth_std']:.0f}mm",
            f"Edge: {analysis['edge_consistency']:.2f}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(vis_image, line, (x1, y2 + 15 + i * 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                       
        return vis_image
        
    def get_statistics(self) -> Dict:
        """Get classification statistics"""
        total = self.stats['total_classifications']
        
        if total == 0:
            return self.stats.copy()
            
        return {
            **self.stats,
            'real_percentage': (self.stats['real_detections'] / total) * 100,
            'fake_percentage': (self.stats['fake_detections'] / total) * 100,
            'screen_percentage': (self.stats['screen_detections'] / total) * 100,
        }
        
    def reset_statistics(self):
        """Reset classification statistics"""
        self.stats = {k: 0 for k in self.stats.keys()}
        self.logger.info("📊 Statistics reset")
        
    def update_config(self, new_config: Dict):
        """Update configuration parameters"""
        self.config.update(new_config)
        self.logger.info(f"⚙️  Configuration updated: {new_config}")
        
if __name__ == "__main__":
    # Test the classifier
    import time
    
    judge = RealPersonJudge()
    
    # Create test data
    height, width = 480, 640
    color_test = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Test with real person (varied depth)
    depth_real = np.random.normal(2000, 300, (height, width)).astype(np.uint16)
    depth_real = np.clip(depth_real, 500, 8000)
    
    # Test with photo (uniform depth)
    depth_fake = np.full((height, width), 50, dtype=np.uint16)  # Very close, uniform
    
    # Test cases
    test_bbox = [100, 100, 300, 400]
    
    print("🧪 Testing Real Person Classifier...")
    
    # Test real person
    start_time = time.time()
    result_real = judge.is_real_person(color_test, depth_real, test_bbox)
    print(f"✅ Real person test: {result_real} ({time.time() - start_time:.3f}s)")
    
    # Test fake person  
    start_time = time.time()
    result_fake = judge.is_real_person(color_test, depth_fake, test_bbox)
    print(f"❌ Fake person test: {result_fake} ({time.time() - start_time:.3f}s)")
    
    # Print statistics
    print(f"📊 Statistics: {judge.get_statistics()}")
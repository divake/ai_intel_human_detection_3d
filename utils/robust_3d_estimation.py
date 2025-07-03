#!/usr/bin/env python3
"""
Robust 3D Position Estimation

Features:
- 3D point cloud processing from depth images
- Multi-point depth sampling for robustness
- Outlier filtering and statistical validation
- Camera coordinate to world coordinate transformation
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Any
import time

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Info: Open3D not available. Install with: pip install open3d")

class Robust3DEstimator:
    """Robust 3D position estimation from depth data"""
    
    def __init__(self, camera_intrinsics: Dict):
        """Initialize 3D estimator with camera parameters"""
        self.logger = logging.getLogger(__name__)
        
        # Camera intrinsics
        self.color_intrinsics = camera_intrinsics.get('color', {})
        self.depth_intrinsics = camera_intrinsics.get('depth', {})
        
        # Fallback intrinsics if not provided
        if not self.color_intrinsics:
            self.color_intrinsics = {
                'fx': 614.4, 'fy': 614.4,
                'ppx': 320.0, 'ppy': 240.0,
                'width': 640, 'height': 480
            }
            
        if not self.depth_intrinsics:
            self.depth_intrinsics = self.color_intrinsics.copy()
            
        # Configuration
        self.config = {
            'sampling_strategy': 'multi_point',  # single_point, center_mass, multi_point
            'num_sample_points': 9,              # For multi-point sampling
            'outlier_threshold': 0.2,            # Meter threshold for outlier removal
            'min_valid_points': 3,               # Minimum points for robust estimation
            'depth_units': 0.001,                # Depth units (mm to m)
            'max_depth': 8.0,                    # Maximum valid depth (meters)
            'min_depth': 0.3,                    # Minimum valid depth (meters)
        }
        
        # Performance tracking
        self.estimation_times = []
        self.total_estimations = 0
        
        self.logger.info("📐 Robust 3D estimator initialized")
        
    def estimate_position(self, bbox: List[float], depth_image: np.ndarray) -> List[float]:
        """
        Estimate robust 3D position from bounding box and depth image
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            depth_image: Depth image (uint16, in mm typically)
            
        Returns:
            3D position [x, y, z] in meters from camera
        """
        start_time = time.time()
        self.total_estimations += 1
        
        try:
            # Extract region of interest
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(depth_image.shape[1], x2)
            y2 = min(depth_image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return [0.0, 0.0, 0.0]
                
            # Extract depth ROI
            depth_roi = depth_image[y1:y2, x1:x2]
            
            # Sample depth points using selected strategy
            depth_points = self._sample_depth_points(depth_roi, (x1, y1))
            
            if len(depth_points) < self.config['min_valid_points']:
                return [0.0, 0.0, 0.0]
                
            # Filter outliers
            filtered_points = self._filter_outliers(depth_points)
            
            if len(filtered_points) < self.config['min_valid_points']:
                return [0.0, 0.0, 0.0]
                
            # Calculate robust 3D position
            position_3d = self._calculate_robust_position(filtered_points)
            
            # Update performance metrics
            estimation_time = time.time() - start_time
            self.estimation_times.append(estimation_time)
            
            # Keep only recent measurements
            if len(self.estimation_times) > 100:
                self.estimation_times = self.estimation_times[-100:]
                
            return position_3d
            
        except Exception as e:
            self.logger.error(f"❌ Error in 3D position estimation: {e}")
            return [0.0, 0.0, 0.0]
            
    def _sample_depth_points(self, depth_roi: np.ndarray, offset: Tuple[int, int]) -> List[Dict]:
        """Sample depth points from ROI using configured strategy"""
        
        strategy = self.config['sampling_strategy']
        x_offset, y_offset = offset
        
        if strategy == 'single_point':
            return self._sample_single_point(depth_roi, x_offset, y_offset)
        elif strategy == 'center_mass':
            return self._sample_center_of_mass(depth_roi, x_offset, y_offset)
        elif strategy == 'multi_point':
            return self._sample_multi_point(depth_roi, x_offset, y_offset)
        else:
            return self._sample_center_of_mass(depth_roi, x_offset, y_offset)
            
    def _sample_single_point(self, depth_roi: np.ndarray, x_offset: int, y_offset: int) -> List[Dict]:
        """Sample single point at center of ROI"""
        
        h, w = depth_roi.shape
        center_y, center_x = h // 2, w // 2
        
        depth_value = depth_roi[center_y, center_x]
        
        if self._is_valid_depth(depth_value):
            return [{
                'pixel_x': x_offset + center_x,
                'pixel_y': y_offset + center_y,
                'depth': depth_value * self.config['depth_units']
            }]
        else:
            return []
            
    def _sample_center_of_mass(self, depth_roi: np.ndarray, x_offset: int, y_offset: int) -> List[Dict]:
        """Sample at center of mass of valid depth pixels"""
        
        # Create mask for valid depths
        valid_mask = (depth_roi > 0) & (depth_roi < 65535)
        valid_depths = depth_roi[valid_mask]
        
        if len(valid_depths) == 0:
            return []
            
        # Calculate center of mass
        y_coords, x_coords = np.where(valid_mask)
        
        if len(x_coords) == 0:
            return []
            
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        depth_value = depth_roi[center_y, center_x]
        
        if self._is_valid_depth(depth_value):
            return [{
                'pixel_x': x_offset + center_x,
                'pixel_y': y_offset + center_y,
                'depth': depth_value * self.config['depth_units']
            }]
        else:
            return []
            
    def _sample_multi_point(self, depth_roi: np.ndarray, x_offset: int, y_offset: int) -> List[Dict]:
        """Sample multiple points in a grid pattern"""
        
        h, w = depth_roi.shape
        points = []
        
        # Create grid of sample points
        grid_size = int(np.sqrt(self.config['num_sample_points']))
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate grid position
                y = int((i + 0.5) * h / grid_size)
                x = int((j + 0.5) * w / grid_size)
                
                if 0 <= y < h and 0 <= x < w:
                    depth_value = depth_roi[y, x]
                    
                    if self._is_valid_depth(depth_value):
                        points.append({
                            'pixel_x': x_offset + x,
                            'pixel_y': y_offset + y,
                            'depth': depth_value * self.config['depth_units']
                        })
                        
        return points
        
    def _is_valid_depth(self, depth_value: float) -> bool:
        """Check if depth value is valid"""
        
        if depth_value <= 0 or depth_value >= 65535:
            return False
            
        depth_meters = depth_value * self.config['depth_units']
        
        return (self.config['min_depth'] <= depth_meters <= self.config['max_depth'])
        
    def _filter_outliers(self, depth_points: List[Dict]) -> List[Dict]:
        """Filter outlier depth measurements using statistical methods"""
        
        if len(depth_points) <= 3:
            return depth_points
            
        # Extract depth values
        depths = np.array([p['depth'] for p in depth_points])
        
        # Calculate median and MAD (Median Absolute Deviation)
        median_depth = np.median(depths)
        mad = np.median(np.abs(depths - median_depth))
        
        # Modified Z-score using MAD
        if mad > 0:
            modified_z_scores = 0.6745 * (depths - median_depth) / mad
            outlier_mask = np.abs(modified_z_scores) < 3.5  # Threshold for outliers
        else:
            # All points are very similar, keep all
            outlier_mask = np.ones(len(depths), dtype=bool)
            
        # Also filter by absolute threshold
        threshold_mask = np.abs(depths - median_depth) < self.config['outlier_threshold']
        
        # Combine masks
        final_mask = outlier_mask & threshold_mask
        
        # Return filtered points
        return [depth_points[i] for i in range(len(depth_points)) if final_mask[i]]
        
    def _calculate_robust_position(self, depth_points: List[Dict]) -> List[float]:
        """Calculate robust 3D position from filtered depth points"""
        
        # Convert depth points to 3D coordinates
        points_3d = []
        
        for point in depth_points:
            x_3d, y_3d, z_3d = self._pixel_to_3d(
                point['pixel_x'],
                point['pixel_y'],
                point['depth']
            )
            points_3d.append([x_3d, y_3d, z_3d])
            
        if not points_3d:
            return [0.0, 0.0, 0.0]
            
        # Calculate robust estimate (median for each coordinate)
        points_array = np.array(points_3d)
        
        robust_position = [
            float(np.median(points_array[:, 0])),  # X
            float(np.median(points_array[:, 1])),  # Y  
            float(np.median(points_array[:, 2]))   # Z
        ]
        
        return robust_position
        
    def _pixel_to_3d(self, pixel_x: int, pixel_y: int, depth: float) -> Tuple[float, float, float]:
        """Convert 2D pixel coordinates + depth to 3D point"""
        
        # Use depth camera intrinsics
        fx = self.depth_intrinsics.get('fx', 614.4)
        fy = self.depth_intrinsics.get('fy', 614.4)
        ppx = self.depth_intrinsics.get('ppx', 320.0)
        ppy = self.depth_intrinsics.get('ppy', 240.0)
        
        # Convert to 3D coordinates
        x = (pixel_x - ppx) * depth / fx
        y = (pixel_y - ppy) * depth / fy
        z = depth
        
        return (x, y, z)
        
    def create_point_cloud(self, color_image: np.ndarray, depth_image: np.ndarray, 
                          roi: Optional[Tuple[int, int, int, int]] = None) -> Optional[Any]:
        """Create 3D point cloud from RGB-D data"""
        
        if not OPEN3D_AVAILABLE:
            self.logger.warning("⚠️  Open3D not available for point cloud creation")
            return None
            
        try:
            # Extract ROI if specified
            if roi:
                x1, y1, x2, y2 = roi
                color_roi = color_image[y1:y2, x1:x2]
                depth_roi = depth_image[y1:y2, x1:x2]
            else:
                color_roi = color_image
                depth_roi = depth_image
                
            # Convert to Open3D format
            color_o3d = o3d.geometry.Image(cv2.cvtColor(color_roi, cv2.COLOR_BGR2RGB))
            depth_o3d = o3d.geometry.Image(depth_roi.astype(np.uint16))
            
            # Create RGB-D image
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1000.0,  # Convert mm to m
                depth_trunc=self.config['max_depth'],
                convert_rgb_to_intensity=False
            )
            
            # Create camera intrinsic parameters
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=self.depth_intrinsics.get('width', 640),
                height=self.depth_intrinsics.get('height', 480),
                fx=self.depth_intrinsics.get('fx', 614.4),
                fy=self.depth_intrinsics.get('fy', 614.4),
                cx=self.depth_intrinsics.get('ppx', 320.0),
                cy=self.depth_intrinsics.get('ppy', 240.0)
            )
            
            # Create point cloud
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic
            )
            
            # Filter point cloud
            if len(point_cloud.points) > 0:
                # Remove outliers
                point_cloud, _ = point_cloud.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=2.0
                )
                
                # Downsample if too many points
                if len(point_cloud.points) > 10000:
                    point_cloud = point_cloud.voxel_down_sample(voxel_size=0.01)
                    
            return point_cloud
            
        except Exception as e:
            self.logger.error(f"❌ Error creating point cloud: {e}")
            return None
            
    def estimate_object_dimensions(self, depth_points: List[Dict]) -> Dict[str, float]:
        """Estimate object dimensions from depth points"""
        
        if len(depth_points) < 3:
            return {'width': 0, 'height': 0, 'depth_span': 0}
            
        # Convert to 3D points
        points_3d = []
        for point in depth_points:
            x, y, z = self._pixel_to_3d(
                point['pixel_x'], point['pixel_y'], point['depth']
            )
            points_3d.append([x, y, z])
            
        points_array = np.array(points_3d)
        
        # Calculate bounding box dimensions
        min_coords = np.min(points_array, axis=0)
        max_coords = np.max(points_array, axis=0)
        
        dimensions = {
            'width': float(max_coords[0] - min_coords[0]),
            'height': float(max_coords[1] - min_coords[1]),
            'depth_span': float(max_coords[2] - min_coords[2])
        }
        
        return dimensions
        
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        
        if not self.estimation_times:
            return {}
            
        return {
            'avg_estimation_time_ms': np.mean(self.estimation_times) * 1000,
            'total_estimations': self.total_estimations,
            'samples': len(self.estimation_times)
        }
        
    def visualize_depth_sampling(self, image: np.ndarray, depth_image: np.ndarray, 
                                bbox: List[float], position_3d: List[float]) -> np.ndarray:
        """Visualize depth sampling and 3D estimation results"""
        
        vis_image = image.copy()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Extract and sample depth points
        depth_roi = depth_image[y1:y2, x1:x2]
        depth_points = self._sample_depth_points(depth_roi, (x1, y1))
        
        # Draw sample points
        for point in depth_points:
            cv2.circle(vis_image, (point['pixel_x'], point['pixel_y']), 3, (255, 0, 0), -1)
            
        # Add 3D position info
        pos_text = f"3D: ({position_3d[0]:.2f}, {position_3d[1]:.2f}, {position_3d[2]:.2f})"
        cv2.putText(vis_image, pos_text, (x1, y1 - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                   
        # Add distance info
        distance = np.linalg.norm(position_3d)
        dist_text = f"Distance: {distance:.2f}m"
        cv2.putText(vis_image, dist_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                   
        return vis_image
        
if __name__ == "__main__":
    # Test the 3D estimator
    camera_intrinsics = {
        'color': {
            'fx': 614.4, 'fy': 614.4,
            'ppx': 320.0, 'ppy': 240.0,
            'width': 640, 'height': 480
        }
    }
    
    estimator = Robust3DEstimator(camera_intrinsics)
    
    # Create test data
    depth_test = np.random.randint(1000, 3000, (480, 640), dtype=np.uint16)
    test_bbox = [200, 150, 400, 400]
    
    print("🧪 Testing Robust 3D Estimator...")
    
    start_time = time.time()
    position_3d = estimator.estimate_position(test_bbox, depth_test)
    print(f"✅ 3D position: {position_3d} ({time.time() - start_time:.3f}s)")
    
    print(f"📊 Performance stats: {estimator.get_performance_stats()}")
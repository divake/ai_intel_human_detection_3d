#!/usr/bin/env python3
"""
RealSense D455 Camera Manager

Features:
- RGB + Depth + IMU stream management
- Camera configuration and calibration
- Frame synchronization and alignment
- Error handling and recovery
- Performance optimization for USB 2.0/3.0
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Tuple, Optional, Any
import threading

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available. Install with: pip install pyrealsense2")

class RealSenseManager:
    """Manages Intel RealSense D455 camera operations"""
    
    def __init__(self, config: Dict):
        """Initialize RealSense camera manager"""
        if not REALSENSE_AVAILABLE:
            raise ImportError("pyrealsense2 required. Install with: pip install pyrealsense2")
            
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Pipeline and streams
        self.pipeline = None
        self.align = None
        self.colorizer = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.dropped_frames = 0
        
        # Threading
        self.frame_lock = threading.Lock()
        self.latest_frames = None
        self.streaming = False
        
        # Camera intrinsics (will be populated on start)
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.depth_to_color_extrinsics = None
        
        self.logger.info("🎥 RealSense manager initialized")
        
    def start(self) -> bool:
        """Start the camera pipeline"""
        try:
            self.logger.info("🚀 Starting RealSense pipeline...")
            
            # Create pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Configure streams based on config
            width = self.config.get('width', 640)
            height = self.config.get('height', 480)
            fps = self.config.get('fps', 30)
            
            # Enable RGB stream
            if self.config.get('enable_rgb', True):
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                self.logger.info(f"✅ RGB stream: {width}x{height}@{fps}fps")
                
            # Enable Depth stream  
            if self.config.get('enable_depth', True):
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                self.logger.info(f"✅ Depth stream: {width}x{height}@{fps}fps")
                
            # Enable IMU if requested
            if self.config.get('enable_imu', False):
                config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
                config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
                self.logger.info("✅ IMU streams enabled")
            
            # Start pipeline
            profile = self.pipeline.start(config)
            
            # Get stream intrinsics
            self._extract_intrinsics(profile)
            
            # Configure advanced settings
            self._configure_advanced_settings(profile)
            
            # Create align object (depth to color)
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            # Create colorizer for depth visualization
            self.colorizer = rs.colorizer()
            
            # Warm up camera
            self._warmup()
            
            self.streaming = True
            self.start_time = time.time()
            self.logger.info("🎯 RealSense camera started successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start RealSense: {e}")
            return False
            
    def _extract_intrinsics(self, profile):
        """Extract camera intrinsics from pipeline profile"""
        try:
            # Get color stream intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            if color_stream:
                self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                self.logger.info(f"📷 Color intrinsics: fx={self.color_intrinsics.fx:.1f}, fy={self.color_intrinsics.fy:.1f}")
                
            # Get depth stream intrinsics
            depth_stream = profile.get_stream(rs.stream.depth)
            if depth_stream:
                self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
                self.logger.info(f"📷 Depth intrinsics: fx={self.depth_intrinsics.fx:.1f}, fy={self.depth_intrinsics.fy:.1f}")
                
                # Get extrinsics (depth to color)
                if color_stream:
                    self.depth_to_color_extrinsics = depth_stream.as_video_stream_profile().get_extrinsics_to(color_stream)
                    
        except Exception as e:
            self.logger.warning(f"⚠️  Could not extract intrinsics: {e}")
            
    def _configure_advanced_settings(self, profile):
        """Configure advanced camera settings"""
        try:
            # Get depth sensor
            depth_sensor = profile.get_device().first_depth_sensor()
            
            # Set depth units (default: 1mm = 0.001m)
            depth_units = self.config.get('depth_units', 0.001)
            depth_sensor.set_option(rs.option.depth_units, depth_units)
            
            # Configure laser power
            laser_power = self.config.get('laser_power', 150)  # 0-360
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, laser_power)
                self.logger.info(f"🔆 Laser power: {laser_power}")
                
            # Set visual preset
            preset = self.config.get('visual_preset', 'medium_density')
            preset_map = {
                'custom': 0,
                'default': 1, 
                'hand': 2,
                'high_accuracy': 3,
                'high_density': 4,
                'medium_density': 5
            }
            
            if preset in preset_map and depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, preset_map[preset])
                self.logger.info(f"👁️  Visual preset: {preset}")
                
        except Exception as e:
            self.logger.warning(f"⚠️  Could not configure advanced settings: {e}")
            
    def _warmup(self, frames_to_skip: int = 30):
        """Warm up camera by skipping initial frames"""
        self.logger.info("🔥 Warming up camera...")
        
        for i in range(frames_to_skip):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                if i % 10 == 0:
                    self.logger.info(f"   Warmup frame {i}/{frames_to_skip}")
            except Exception:
                continue
                
        self.logger.info("✅ Camera warmup completed")
        
    def get_frames(self, timeout_ms: int = 1000) -> Optional[Dict[str, np.ndarray]]:
        """Get latest synchronized frames"""
        if not self.streaming or self.pipeline is None:
            return None
            
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            
            # Align depth to color
            if self.align:
                aligned_frames = self.align.process(frames)
            else:
                aligned_frames = frames
            
            result = {}
            
            # Get color frame
            if self.config.get('enable_rgb', True):
                color_frame = aligned_frames.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    result['color'] = color_image
                    
            # Get depth frame
            if self.config.get('enable_depth', True):
                depth_frame = aligned_frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    result['depth'] = depth_image
                    
                    # Also provide colorized depth for visualization
                    colorized_depth = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
                    result['depth_colorized'] = colorized_depth
                    
            # Get IMU data if enabled
            if self.config.get('enable_imu', False):
                # Get accelerometer data
                if aligned_frames.first_or_default(rs.stream.accel):
                    accel_frame = aligned_frames.first_or_default(rs.stream.accel)
                    accel_data = accel_frame.as_motion_frame().get_motion_data()
                    result['accel'] = np.array([accel_data.x, accel_data.y, accel_data.z])
                    
                # Get gyroscope data  
                if aligned_frames.first_or_default(rs.stream.gyro):
                    gyro_frame = aligned_frames.first_or_default(rs.stream.gyro)
                    gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                    result['gyro'] = np.array([gyro_data.x, gyro_data.y, gyro_data.z])
            
            # Update performance metrics
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed
                
            # Store latest frames with thread safety
            with self.frame_lock:
                self.latest_frames = result
                
            return result
            
        except Exception as e:
            self.dropped_frames += 1
            if self.dropped_frames % 10 == 0:
                self.logger.warning(f"⚠️  Dropped {self.dropped_frames} frames. Latest error: {e}")
            return None
            
    def get_intrinsics(self) -> Dict[str, Any]:
        """Get camera intrinsic parameters"""
        intrinsics = {}
        
        if self.color_intrinsics:
            intrinsics['color'] = {
                'width': self.color_intrinsics.width,
                'height': self.color_intrinsics.height,
                'fx': self.color_intrinsics.fx,
                'fy': self.color_intrinsics.fy,
                'ppx': self.color_intrinsics.ppx,
                'ppy': self.color_intrinsics.ppy,
                'coeffs': self.color_intrinsics.coeffs
            }
            
        if self.depth_intrinsics:
            intrinsics['depth'] = {
                'width': self.depth_intrinsics.width,
                'height': self.depth_intrinsics.height,
                'fx': self.depth_intrinsics.fx,
                'fy': self.depth_intrinsics.fy,
                'ppx': self.depth_intrinsics.ppx,
                'ppy': self.depth_intrinsics.ppy,
                'coeffs': self.depth_intrinsics.coeffs
            }
            
        return intrinsics
        
    def get_3d_point(self, pixel_x: int, pixel_y: int, depth_value: float) -> Tuple[float, float, float]:
        """Convert 2D pixel + depth to 3D point"""
        if not self.depth_intrinsics:
            return (0, 0, 0)
            
        # Deproject pixel to 3D point
        point_3d = rs.rs2_deproject_pixel_to_point(
            self.depth_intrinsics, [pixel_x, pixel_y], depth_value
        )
        
        return tuple(point_3d)
        
    def get_performance_stats(self) -> Dict:
        """Get camera performance statistics"""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'uptime_seconds': time.time() - self.start_time,
            'drop_rate_percent': (self.dropped_frames / max(self.frame_count, 1)) * 100
        }
        
    def stop(self):
        """Stop the camera pipeline"""
        self.streaming = False
        
        if self.pipeline:
            try:
                self.pipeline.stop()
                self.logger.info("⏹️  RealSense pipeline stopped")
            except Exception as e:
                self.logger.error(f"❌ Error stopping pipeline: {e}")
                
        # Log final statistics
        stats = self.get_performance_stats()
        self.logger.info(f"📊 Final stats: {stats['fps']:.1f} FPS, {stats['dropped_frames']} dropped frames")
        
    def __del__(self):
        """Cleanup resources"""
        self.stop()
        
if __name__ == "__main__":
    # Test the camera manager
    config = {
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
    
    camera = RealSenseManager(config)
    
    if camera.start():
        print("✅ Camera test started. Press 'q' to quit...")
        
        try:
            while True:
                frames = camera.get_frames()
                
                if frames and 'color' in frames:
                    cv2.imshow('RealSense Test', frames['color'])
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            camera.stop()
            cv2.destroyAllWindows()
    else:
        print("❌ Camera test failed")
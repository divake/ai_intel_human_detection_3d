#!/usr/bin/env python3
"""
Real-time 3D Visualization System

Features:
- 3D point cloud rendering
- Real-time tracking visualization
- Interactive 3D scene with trajectories
- Performance-optimized rendering
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
import threading
from collections import deque

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Info: Open3D not available. Install with: pip install open3d")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Info: matplotlib not available. Install with: pip install matplotlib")

class RealTime3DVisualizer:
    """Real-time 3D visualization for human tracking"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize 3D visualizer"""
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = config or {
            'window_width': 1024,
            'window_height': 768,
            'point_size': 2.0,
            'coordinate_frame_size': 0.5,
            'trajectory_length': 50,
            'update_interval': 0.033,  # ~30 FPS
            'background_color': [0.1, 0.1, 0.1],
            'person_colors': [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
                [1.0, 0.5, 0.0],  # Orange
                [0.5, 0.0, 1.0],  # Purple
            ]
        }
        
        # Visualization state
        self.visualizer = None
        self.is_initialized = False
        self.is_running = False
        
        # Track data
        self.track_trajectories = {}  # track_id -> deque of positions
        self.track_colors = {}        # track_id -> color
        self.current_tracks = []      # Current frame tracks
        
        # Point cloud data
        self.current_point_cloud = None
        self.point_cloud_updated = False
        
        # Rendering objects
        self.coordinate_frame = None
        self.trajectory_lines = {}
        self.person_spheres = {}
        
        # Threading
        self.render_thread = None
        self.data_lock = threading.Lock()
        
        self.logger.info("🎨 3D visualizer initialized")
        
    def initialize(self) -> bool:
        """Initialize the 3D visualization window"""
        
        if not OPEN3D_AVAILABLE:
            self.logger.warning("⚠️  Open3D not available, using fallback 2D visualization")
            return self._initialize_matplotlib_fallback()
            
        try:
            # Create visualizer
            self.visualizer = o3d.visualization.Visualizer()
            self.visualizer.create_window(
                window_name="3D Human Detection",
                width=self.config['window_width'],
                height=self.config['window_height']
            )
            
            # Set rendering options
            render_option = self.visualizer.get_render_option()
            render_option.background_color = np.array(self.config['background_color'])
            render_option.point_size = self.config['point_size']
            
            # Add coordinate frame
            self._add_coordinate_frame()
            
            # Set camera view
            self._set_camera_view()
            
            self.is_initialized = True
            self.is_running = True
            
            # Start rendering thread
            self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
            self.render_thread.start()
            
            self.logger.info("✅ 3D visualizer initialized with Open3D")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Open3D visualizer: {e}")
            return self._initialize_matplotlib_fallback()
            
    def _initialize_matplotlib_fallback(self) -> bool:
        """Initialize matplotlib-based 3D visualization as fallback"""
        
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("❌ Neither Open3D nor matplotlib available for 3D visualization")
            return False
            
        try:
            plt.ion()  # Interactive mode
            self.fig = plt.figure(figsize=(12, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # Set labels and limits
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            
            # Set reasonable limits
            self.ax.set_xlim([-3, 3])
            self.ax.set_ylim([-3, 3])
            self.ax.set_zlim([0, 6])
            
            self.is_initialized = True
            self.logger.info("✅ 3D visualizer initialized with matplotlib fallback")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize matplotlib visualizer: {e}")
            return False
            
    def update(self, tracks: List[Dict], depth_image: Optional[np.ndarray] = None):
        """Update visualization with new tracking data"""
        
        if not self.is_initialized:
            return
            
        with self.data_lock:
            self.current_tracks = tracks.copy()
            
            # Update trajectories
            for track in tracks:
                track_id = track['id']
                position = track.get('position_3d', [0, 0, 0])
                
                # Initialize trajectory if new track
                if track_id not in self.track_trajectories:
                    self.track_trajectories[track_id] = deque(
                        maxlen=self.config['trajectory_length']
                    )
                    # Assign color
                    color_idx = len(self.track_colors) % len(self.config['person_colors'])
                    self.track_colors[track_id] = self.config['person_colors'][color_idx]
                    
                # Add current position to trajectory
                self.track_trajectories[track_id].append(position)
                
            # Clean up old trajectories
            active_ids = {track['id'] for track in tracks}
            for track_id in list(self.track_trajectories.keys()):
                if track_id not in active_ids:
                    # Keep trajectory for a while after track disappears
                    if len(self.track_trajectories[track_id]) > 0:
                        # Gradually fade out by reducing trajectory length
                        if len(self.track_trajectories[track_id]) > 10:
                            for _ in range(5):
                                if self.track_trajectories[track_id]:
                                    self.track_trajectories[track_id].popleft()
                        else:
                            del self.track_trajectories[track_id]
                            if track_id in self.track_colors:
                                del self.track_colors[track_id]
                                
        # Update visualization based on backend
        if OPEN3D_AVAILABLE and self.visualizer:
            self._update_open3d()
        elif MATPLOTLIB_AVAILABLE:
            self._update_matplotlib()
            
    def _update_open3d(self):
        """Update Open3D visualization"""
        
        # Update will be handled in render loop
        pass
        
    def _update_matplotlib(self):
        """Update matplotlib visualization"""
        
        try:
            self.ax.clear()
            
            # Set labels and limits
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_xlim([-3, 3])
            self.ax.set_ylim([-3, 3])
            self.ax.set_zlim([0, 6])
            
            # Draw trajectories
            for track_id, trajectory in self.track_trajectories.items():
                if len(trajectory) > 1:
                    positions = np.array(list(trajectory))
                    color = self.track_colors.get(track_id, [0.5, 0.5, 0.5])
                    
                    # Draw trajectory line
                    self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                               color=color, alpha=0.7, linewidth=2)
                    
                    # Draw current position
                    if len(positions) > 0:
                        current_pos = positions[-1]
                        self.ax.scatter(current_pos[0], current_pos[1], current_pos[2],
                                      color=color, s=100, alpha=1.0)
                        
                        # Add track ID label
                        self.ax.text(current_pos[0], current_pos[1], current_pos[2] + 0.1,
                                   f'ID:{track_id}', fontsize=8, color=color)
                                   
            # Add coordinate frame
            origin = [0, 0, 0]
            self.ax.quiver(origin[0], origin[1], origin[2], 0.5, 0, 0, color='red', alpha=0.8)
            self.ax.quiver(origin[0], origin[1], origin[2], 0, 0.5, 0, color='green', alpha=0.8)
            self.ax.quiver(origin[0], origin[1], origin[2], 0, 0, 0.5, color='blue', alpha=0.8)
            
            # Add statistics
            num_tracks = len([t for t in self.current_tracks if t.get('state') == 'confirmed'])
            self.ax.text2D(0.02, 0.98, f"Active Tracks: {num_tracks}", 
                          transform=self.ax.transAxes, fontsize=12, 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            plt.draw()
            plt.pause(0.001)
            
        except Exception as e:
            self.logger.debug(f"Error updating matplotlib visualization: {e}")
            
    def _render_loop(self):
        """Main rendering loop for Open3D"""
        
        while self.is_running:
            try:
                with self.data_lock:
                    self._update_3d_scene()
                    
                # Update visualizer
                self.visualizer.poll_events()
                self.visualizer.update_renderer()
                
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                self.logger.debug(f"Error in render loop: {e}")
                break
                
    def _update_3d_scene(self):
        """Update 3D scene objects"""
        
        if not self.visualizer:
            return
            
        # Clear previous person objects
        for sphere in self.person_spheres.values():
            self.visualizer.remove_geometry(sphere, reset_bounding_box=False)
        self.person_spheres.clear()
        
        # Clear previous trajectory lines
        for line in self.trajectory_lines.values():
            self.visualizer.remove_geometry(line, reset_bounding_box=False)
        self.trajectory_lines.clear()
        
        # Add current person positions
        for track in self.current_tracks:
            track_id = track['id']
            position = track.get('position_3d', [0, 0, 0])
            
            if any(abs(p) > 0.01 for p in position):  # Valid position
                # Create sphere for person
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                sphere.translate(position)
                
                # Set color
                color = self.track_colors.get(track_id, [0.5, 0.5, 0.5])
                sphere.paint_uniform_color(color)
                
                self.visualizer.add_geometry(sphere, reset_bounding_box=False)
                self.person_spheres[track_id] = sphere
                
        # Add trajectory lines
        for track_id, trajectory in self.track_trajectories.items():
            if len(trajectory) > 1:
                positions = np.array(list(trajectory))
                
                # Create line set
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(positions)
                
                # Create line indices
                lines = []
                for i in range(len(positions) - 1):
                    lines.append([i, i + 1])
                line_set.lines = o3d.utility.Vector2iVector(lines)
                
                # Set color
                color = self.track_colors.get(track_id, [0.5, 0.5, 0.5])
                colors = [color for _ in lines]
                line_set.colors = o3d.utility.Vector3dVector(colors)
                
                self.visualizer.add_geometry(line_set, reset_bounding_box=False)
                self.trajectory_lines[track_id] = line_set
                
    def _add_coordinate_frame(self):
        """Add coordinate frame to the scene"""
        
        if not self.visualizer:
            return
            
        # Create coordinate frame
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.config['coordinate_frame_size']
        )
        self.visualizer.add_geometry(self.coordinate_frame)
        
    def _set_camera_view(self):
        """Set optimal camera view"""
        
        if not self.visualizer:
            return
            
        # Set camera parameters for good viewing angle
        view_control = self.visualizer.get_view_control()
        
        # Set camera position and orientation
        view_control.set_front([0.4, -0.2, -0.9])
        view_control.set_up([0.0, -1.0, 0.0])
        view_control.set_lookat([0.0, 0.0, 2.0])
        view_control.set_zoom(0.8)
        
    def add_point_cloud(self, point_cloud):
        """Add point cloud to visualization"""
        
        if not self.is_initialized or not OPEN3D_AVAILABLE:
            return
            
        with self.data_lock:
            if self.current_point_cloud:
                self.visualizer.remove_geometry(self.current_point_cloud, reset_bounding_box=False)
                
            if point_cloud and len(point_cloud.points) > 0:
                # Downsample for performance
                if len(point_cloud.points) > 5000:
                    point_cloud = point_cloud.voxel_down_sample(voxel_size=0.02)
                    
                self.visualizer.add_geometry(point_cloud, reset_bounding_box=False)
                self.current_point_cloud = point_cloud
                
    def create_2d_overlay(self, tracks: List[Dict]) -> np.ndarray:
        """Create 2D overlay with track information"""
        
        # Create overlay image
        overlay = np.zeros((200, 400, 3), dtype=np.uint8)
        
        # Add title
        cv2.putText(overlay, "3D Tracking Stats", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add track information
        y_offset = 60
        for i, track in enumerate(tracks[:5]):  # Show first 5 tracks
            track_id = track['id']
            position = track.get('position_3d', [0, 0, 0])
            speed = track.get('speed', 0)
            posture = track.get('posture', 'unknown')
            
            # Get color
            color_idx = track_id % len(self.config['person_colors'])
            color = tuple(int(c * 255) for c in self.config['person_colors'][color_idx])
            
            # Format text
            text = f"ID{track_id}: ({position[0]:.1f},{position[1]:.1f},{position[2]:.1f})"
            speed_text = f"Speed: {speed:.1f}m/s | {posture}"
            
            cv2.putText(overlay, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(overlay, speed_text, (10, y_offset + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            y_offset += 35
            
        return overlay
        
    def save_trajectory_plot(self, filename: str):
        """Save trajectory plot to file"""
        
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("⚠️  matplotlib not available for saving plots")
            return
            
        try:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot all trajectories
            for track_id, trajectory in self.track_trajectories.items():
                if len(trajectory) > 1:
                    positions = np.array(list(trajectory))
                    color = self.track_colors.get(track_id, [0.5, 0.5, 0.5])
                    
                    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                           color=color, linewidth=2, label=f'Track {track_id}')
                           
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.legend()
            ax.set_title('3D Human Tracking Trajectories')
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📸 Trajectory plot saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving trajectory plot: {e}")
            
    def get_statistics(self) -> Dict:
        """Get visualization statistics"""
        
        return {
            'active_trajectories': len(self.track_trajectories),
            'total_track_colors': len(self.track_colors),
            'is_running': self.is_running,
            'backend': 'open3d' if OPEN3D_AVAILABLE and self.visualizer else 'matplotlib'
        }
        
    def close(self):
        """Close visualization window"""
        
        self.is_running = False
        
        if self.render_thread:
            self.render_thread.join(timeout=1.0)
            
        if OPEN3D_AVAILABLE and self.visualizer:
            self.visualizer.close()
            
        if MATPLOTLIB_AVAILABLE:
            plt.close('all')
            
        self.logger.info("🎨 3D visualizer closed")
        
    def __del__(self):
        """Cleanup resources"""
        self.close()
        
if __name__ == "__main__":
    # Test the visualizer
    visualizer = RealTime3DVisualizer()
    
    if visualizer.initialize():
        print("✅ Visualizer initialized")
        
        # Test with sample tracks
        test_tracks = [
            {
                'id': 1,
                'position_3d': [1.0, 0.5, 2.0],
                'speed': 0.8,
                'posture': 'walking',
                'state': 'confirmed'
            },
            {
                'id': 2,
                'position_3d': [-0.5, 1.0, 1.5],
                'speed': 0.0,
                'posture': 'standing',
                'state': 'confirmed'
            }
        ]
        
        # Update visualization multiple times
        for i in range(10):
            # Simulate movement
            test_tracks[0]['position_3d'][0] += 0.1
            test_tracks[1]['position_3d'][1] += 0.05
            
            visualizer.update(test_tracks)
            time.sleep(0.1)
            
        print("✅ Visualization test completed")
        
        # Keep window open for a moment
        if MATPLOTLIB_AVAILABLE:
            time.sleep(2)
            
        visualizer.close()
    else:
        print("❌ Visualizer initialization failed")
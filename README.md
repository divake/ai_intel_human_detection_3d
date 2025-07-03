# Real-Time 3D Human Detection System
*Powered by Intel RealSense D455 + YOLOv11 + Intel Hardware Acceleration*

## 🎯 Project Overview

Advanced real-time human detection system that combines:
- **YOLOv11**: Latest object detection from NeurIPS 2024
- **Intel RealSense D455**: RGB + Depth + IMU sensor fusion
- **3D Point Cloud**: Real-time 3D visualization
- **Multi-person Tracking**: Unique ID assignment with trajectory tracking
- **Real vs Photo Detection**: Depth analysis to distinguish real people from images
- **Motion Analysis**: Speed calculation and posture classification

## 🏗️ System Architecture

```
RealSense D455 Camera
├── RGB Stream (640x480@30fps)
├── Depth Stream (640x480@30fps)
└── IMU Data (Accelerometer + Gyroscope)
         ↓
YOLOv11 Detection Engine
├── Person Detection & Bounding Boxes
├── Real-time Inference (Intel CPU Optimized)
└── Multi-object Detection
         ↓
Depth Analysis & Filtering
├── Real vs Photo Classification
├── 3D Position Estimation
└── Distance Calculation
         ↓
Motion Tracking System
├── Multi-person ID Assignment
├── Trajectory Smoothing
├── Speed Calculation
└── Posture Classification
         ↓
3D Visualization & Output
├── Point Cloud Rendering
├── Real-time Dashboard
└── Data Logging
```

## 💻 Hardware Utilization

### Intel Core Ultra 7 165H Acceleration
- **CPU**: YOLOv11 inference with Intel Extension for PyTorch (2.3x speedup)
- **NPU**: Matrix operations and specific ML workloads
- **GPU**: OpenCL compute for point cloud processing
- **Memory**: 64GB for multi-stream processing

### RealSense D455 Configuration
- **Color**: 640x480 @ 30fps (USB 2.0 optimized)
- **Depth**: 640x480 @ 30fps with laser emitter
- **IMU**: 400Hz accelerometer + gyroscope
- **Range**: 0.4m - 20m detection capability

## 📁 Project Structure

```
human_detection_3d/
├── model/
│   ├── yolo_detector.py         # YOLOv11 detection engine
│   ├── model_loader.py          # Model management
│   └── intel_optimizations.py   # CPU/NPU acceleration
├── utils/
│   ├── motion_tracker.py        # Multi-object tracking
│   ├── photo_judge.py           # Real vs fake detection
│   ├── posture_classification.py # Pose analysis
│   ├── robust_3d_estimation.py  # 3D point cloud processing
│   ├── realsense_manager.py     # Camera interface
│   └── visualization.py         # 3D rendering
├── config/
│   ├── camera_config.yaml       # RealSense settings
│   ├── model_config.yaml        # YOLOv11 parameters
│   └── tracking_config.yaml     # Motion tracking settings
├── data/
│   ├── models/                  # Pre-trained weights
│   ├── calibration/             # Camera calibration
│   └── test_videos/             # Sample data
├── outputs/
│   ├── logs/                    # Detection logs
│   ├── recordings/              # Video recordings
│   └── point_clouds/            # 3D data exports
├── notebooks/
│   ├── camera_calibration.ipynb # Setup and testing
│   ├── model_evaluation.ipynb   # Performance analysis
│   └── visualization_demo.ipynb # 3D visualization demos
├── main.py                      # Main application
├── requirements.txt             # Dependencies
└── setup.py                     # Installation script
```

## 🚀 Key Features

### 1. Advanced Person Detection
- YOLOv11 state-of-the-art accuracy
- Real-time inference optimized for Intel hardware
- Multi-person simultaneous detection

### 2. Real vs Photo Classification
- Depth-based authenticity verification
- Prevents false positives from screens/photos
- Configurable depth thresholds

### 3. 3D Spatial Tracking
- Real-time 3D position estimation
- Distance measurement from camera
- Speed calculation and trajectory analysis

### 4. Posture Classification
- Standing, sitting, walking detection
- Body orientation analysis
- Movement pattern recognition

### 5. Real-time Visualization
- 3D point cloud rendering
- Live tracking dashboard
- Configurable overlay graphics

## 📊 Performance Targets

- **Detection Latency**: <50ms per frame
- **Tracking Accuracy**: >95% ID consistency
- **Real vs Photo**: >99% classification accuracy
- **3D Position Error**: <10cm at 5m distance
- **System FPS**: 25-30 fps end-to-end

## 🛠️ Installation & Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download YOLOv11 model
python setup.py --download-models

# 3. Calibrate camera
python notebooks/camera_calibration.ipynb

# 4. Run the system
python main.py
```

## 📈 Development Roadmap

- [x] Project setup and architecture
- [ ] YOLOv11 integration with Intel optimizations
- [ ] RealSense D455 interface implementation
- [ ] Real vs photo detection algorithm
- [ ] Multi-person tracking system
- [ ] 3D point cloud processing
- [ ] Posture classification module
- [ ] Real-time visualization dashboard
- [ ] Performance optimization and testing
- [ ] Documentation and deployment

## 🎮 Usage Examples

### Basic Detection
```python
from main import HumanDetection3D

detector = HumanDetection3D()
detector.start_realtime_detection()
```

### Batch Processing
```python
detector.process_video("input.mp4", output_dir="outputs/")
```

### 3D Visualization
```python
detector.enable_3d_visualization()
detector.export_point_cloud("person_tracking.ply")
```

---
*Built with ❤️ for Intel AI hardware acceleration and RealSense depth sensing*
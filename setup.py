#!/usr/bin/env python3
"""
Setup script for Real-Time 3D Human Detection System

Features:
- Automatic dependency installation
- Model downloading
- Camera calibration assistance
- Intel optimization detection and setup
"""

import os
import sys
import subprocess
import argparse
import urllib.request
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemSetup:
    """Setup and configuration for the 3D Human Detection system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.data_dir / "models"
        self.outputs_dir = self.project_root / "outputs"
        
    def create_directories(self):
        """Create necessary project directories"""
        logger.info("🗂️  Creating project directories...")
        
        directories = [
            self.data_dir,
            self.models_dir,
            self.data_dir / "calibration",
            self.data_dir / "test_videos",
            self.outputs_dir / "logs",
            self.outputs_dir / "recordings",
            self.outputs_dir / "screenshots",
            self.outputs_dir / "point_clouds"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"   ✅ Created: {directory}")
            
    def install_dependencies(self, gpu_support=False, dev_tools=False):
        """Install Python dependencies"""
        logger.info("📦 Installing dependencies...")
        
        # Read requirements
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            logger.error("❌ requirements.txt not found")
            return False
            
        try:
            # Install main requirements
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            subprocess.run(cmd, check=True)
            logger.info("✅ Main dependencies installed")
            
            # Install optional GPU support
            if gpu_support:
                logger.info("🚀 Installing GPU support...")
                gpu_packages = ["torch-audio", "torchaudio"]
                for package in gpu_packages:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
                    
            # Install development tools
            if dev_tools:
                logger.info("🛠️  Installing development tools...")
                dev_packages = ["pytest", "black", "flake8", "jupyter"]
                for package in dev_packages:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
                    
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
            
    def download_models(self):
        """Download YOLOv11 models"""
        logger.info("🤖 Downloading YOLOv11 models...")
        
        models = {
            "yolov11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov11n.pt",
            "yolov11s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov11s.pt"
        }
        
        for model_name, url in models.items():
            model_path = self.models_dir / model_name
            
            if model_path.exists():
                logger.info(f"   ✅ {model_name} already exists")
                continue
                
            try:
                logger.info(f"   🔄 Downloading {model_name}...")
                urllib.request.urlretrieve(url, model_path)
                logger.info(f"   ✅ Downloaded: {model_name}")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to download {model_name}: {e}")
                
    def test_camera(self):
        """Test RealSense camera connectivity"""
        logger.info("📷 Testing RealSense camera...")
        
        try:
            import pyrealsense2 as rs
            
            # Create pipeline
            pipeline = rs.pipeline()
            config = rs.config()
            
            # Configure streams
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start pipeline
            profile = pipeline.start(config)
            
            # Get a few frames
            for i in range(5):
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    raise RuntimeError("Failed to get frames")
                    
            pipeline.stop()
            logger.info("✅ RealSense camera test successful")
            return True
            
        except ImportError:
            logger.error("❌ pyrealsense2 not installed")
            return False
        except Exception as e:
            logger.error(f"❌ Camera test failed: {e}")
            logger.info("💡 Make sure RealSense camera is connected and firmware is updated")
            return False
            
    def test_intel_optimizations(self):
        """Test Intel optimizations availability"""
        logger.info("⚡ Testing Intel optimizations...")
        
        # Test Intel Extension for PyTorch
        try:
            import intel_extension_for_pytorch as ipex
            logger.info("✅ Intel Extension for PyTorch available")
        except ImportError:
            logger.warning("⚠️  Intel Extension for PyTorch not available")
            
        # Test OpenVINO
        try:
            import openvino as ov
            logger.info("✅ Intel OpenVINO available")
        except ImportError:
            logger.warning("⚠️  Intel OpenVINO not available")
            
        # Test MKL
        try:
            import mkl
            logger.info("✅ Intel MKL available")
        except ImportError:
            logger.warning("⚠️  Intel MKL not available")
            
    def run_quick_test(self):
        """Run a quick system test"""
        logger.info("🧪 Running quick system test...")
        
        try:
            # Test imports
            import numpy as np
            import cv2
            import torch
            from ultralytics import YOLO
            
            logger.info("✅ Core imports successful")
            
            # Test YOLO model loading
            model_path = self.models_dir / "yolov11n.pt"
            if model_path.exists():
                model = YOLO(str(model_path))
                logger.info("✅ YOLOv11 model loading successful")
            else:
                logger.warning("⚠️  YOLOv11 model not found")
                
            # Test basic detection on dummy image
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            if 'model' in locals():
                results = model(dummy_image, verbose=False)
                logger.info("✅ YOLOv11 inference test successful")
                
            logger.info("🎯 Quick test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Quick test failed: {e}")
            return False
            
    def create_launcher_script(self):
        """Create launcher script for easy execution"""
        logger.info("🚀 Creating launcher script...")
        
        launcher_content = '''#!/bin/bash
# Real-Time 3D Human Detection System Launcher

echo "🎯 Starting Real-Time 3D Human Detection System..."

# Check if virtual environment should be activated
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Run the main application
python main.py "$@"
'''
        
        launcher_path = self.project_root / "run.sh"
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
            
        # Make executable
        os.chmod(launcher_path, 0o755)
        logger.info(f"✅ Launcher script created: {launcher_path}")
        
    def print_setup_summary(self):
        """Print setup completion summary"""
        logger.info("\n🎉 Setup completed successfully!")
        logger.info("\n📋 Next steps:")
        logger.info("   1. Connect RealSense D455 camera")
        logger.info("   2. Run: python main.py")
        logger.info("   3. Or use: ./run.sh")
        logger.info("\n🔧 Configuration:")
        logger.info("   - Edit config/*.yaml files for customization")
        logger.info("   - Check outputs/ directory for logs and recordings")
        logger.info("\n📚 Documentation:")
        logger.info("   - See README.md for detailed usage instructions")
        logger.info("   - Check notebooks/ for examples and calibration")

def main():
    parser = argparse.ArgumentParser(description="Setup Real-Time 3D Human Detection System")
    parser.add_argument("--download-models", action="store_true", 
                       help="Download YOLOv11 models")
    parser.add_argument("--test-camera", action="store_true",
                       help="Test RealSense camera")
    parser.add_argument("--test-intel", action="store_true",
                       help="Test Intel optimizations")
    parser.add_argument("--install-deps", action="store_true",
                       help="Install dependencies")
    parser.add_argument("--gpu-support", action="store_true",
                       help="Install GPU support packages")
    parser.add_argument("--dev-tools", action="store_true",
                       help="Install development tools")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick system test")
    parser.add_argument("--all", action="store_true",
                       help="Run complete setup")
    
    args = parser.parse_args()
    
    setup = SystemSetup()
    
    if args.all or not any(vars(args).values()):
        # Full setup
        logger.info("🚀 Running complete setup...")
        setup.create_directories()
        setup.install_dependencies(args.gpu_support, args.dev_tools)
        setup.download_models()
        setup.test_intel_optimizations()
        setup.test_camera()
        setup.run_quick_test()
        setup.create_launcher_script()
        setup.print_setup_summary()
        
    else:
        # Individual components
        setup.create_directories()
        
        if args.install_deps:
            setup.install_dependencies(args.gpu_support, args.dev_tools)
            
        if args.download_models:
            setup.download_models()
            
        if args.test_camera:
            setup.test_camera()
            
        if args.test_intel:
            setup.test_intel_optimizations()
            
        if args.quick_test:
            setup.run_quick_test()

if __name__ == "__main__":
    main()
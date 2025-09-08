#!/usr/bin/env python3
"""
Launcher script for the Image to UML Converter Playground
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    required_packages = ['streamlit', 'PIL', 'cv2', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def main():
    print("🚀 Starting Image to UML Converter Playground...")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Not in a virtual environment. Consider activating venv first.")
    
    # Check requirements
    missing = check_requirements()
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Please install requirements: pip install -r requirements.txt")
        return
    
    print("✅ All requirements satisfied")
    
    # Launch Streamlit
    try:
        print("🌐 Launching Streamlit app...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "playground_ui.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Playground stopped by user")
    except Exception as e:
        print(f"❌ Error launching playground: {e}")

if __name__ == "__main__":
    main()
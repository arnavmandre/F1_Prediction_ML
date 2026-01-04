"""Comprehensive fix script for F1 Prediction Model issues."""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and show output."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("="*60)
    print("F1 Prediction Model - Complete Fix Script")
    print("="*60)
    
    # Show current Python info
    print(f"\nCurrent Python: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Step 1: Upgrade pip
    print("\n[STEP 1] Upgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Step 2: Install/upgrade all dependencies
    print("\n[STEP 2] Installing all dependencies...")
    if os.path.exists("requirements.txt"):
        run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing from requirements.txt")
    else:
        print("requirements.txt not found, installing manually...")
        packages = [
            "numpy<2.0",  # Force numpy 1.x for compatibility
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "scikit-learn>=1.2.0",
            "xgboost>=1.7.0",
            "fastf1>=3.0.0",
            "opencv-python>=4.8.0",
            "Pillow>=10.0.0",
            "pytesseract>=0.3.10",
            "easyocr>=1.7.0",
            "tqdm>=4.65.0"
        ]
        for package in packages:
            run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}")
    
    # Step 3: Verify imports
    print("\n[STEP 3] Verifying all imports...")
    test_imports = [
        "import numpy",
        "import pandas",
        "import sklearn",
        "import xgboost",
        "import matplotlib",
        "import seaborn",
        "import fastf1",
        "import cv2",
        "from PIL import Image",
        "import easyocr",
        "import tqdm"
    ]
    
    failed_imports = []
    for imp in test_imports:
        cmd = f'{sys.executable} -c "{imp}; print(\'OK\')"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [OK] {imp}")
        else:
            print(f"  [FAIL] {imp}")
            failed_imports.append(imp)
    
    # Step 4: Test main module
    print("\n[STEP 4] Testing main module import...")
    cmd = f'{sys.executable} -c "import f1_prediction_model; print(\'Module import OK\')"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("  [OK] f1_prediction_model imports successfully")
    else:
        print("  [FAIL] f1_prediction_model import failed")
        print("  Error output:")
        print(result.stderr)
    
    # Step 5: Run test
    print("\n[STEP 5] Running test_model.py...")
    if os.path.exists("test_model.py"):
        result = subprocess.run(f'{sys.executable} test_model.py', shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode == 0:
            print("\n  [SUCCESS] All tests passed!")
        else:
            print("\n  [FAIL] Tests failed")
    else:
        print("  [SKIP] test_model.py not found")
    
    # Summary
    print("\n" + "="*60)
    print("FIX SUMMARY")
    print("="*60)
    
    if failed_imports:
        print(f"\n[ISSUES FOUND] {len(failed_imports)} imports failed:")
        for imp in failed_imports:
            print(f"  - {imp}")
        print("\nTry running this script again, or manually install:")
        print(f"  {sys.executable} -m pip install <package_name>")
    else:
        print("\n[SUCCESS] All imports working!")
        print("\nYou can now run:")
        print("  python test_model.py")
        print("  python f1_prediction_model.py")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()


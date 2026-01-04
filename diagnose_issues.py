"""Diagnostic script to identify all issues with the F1 prediction model."""

import sys
import os

print("=" * 60)
print("F1 Prediction Model - Diagnostic Check")
print("=" * 60)

errors = []
warnings = []

# Check Python version
print("\n1. Python Environment:")
print(f"   Python version: {sys.version}")
print(f"   Python executable: {sys.executable}")

# Check critical imports one by one
print("\n2. Checking Critical Imports:")

try:
    import numpy as np
    print(f"   [OK] numpy {np.__version__}")
except Exception as e:
    print(f"   [ERROR] numpy: {e}")
    errors.append(f"numpy import failed: {e}")

try:
    import pandas as pd
    print(f"   [OK] pandas {pd.__version__}")
except Exception as e:
    print(f"   [ERROR] pandas: {e}")
    errors.append(f"pandas import failed: {e}")

try:
    import sklearn
    print(f"   [OK] scikit-learn {sklearn.__version__}")
except Exception as e:
    print(f"   [ERROR] scikit-learn: {e}")
    errors.append(f"scikit-learn import failed: {e}")

try:
    import xgboost as xgb
    print(f"   [OK] xgboost {xgb.__version__}")
except Exception as e:
    print(f"   [ERROR] xgboost: {e}")
    errors.append(f"xgboost import failed: {e}")

try:
    import matplotlib
    print(f"   [OK] matplotlib {matplotlib.__version__}")
except Exception as e:
    print(f"   [ERROR] matplotlib: {e}")
    errors.append(f"matplotlib import failed: {e}")

try:
    import seaborn as sns
    print(f"   [OK] seaborn {sns.__version__}")
except Exception as e:
    print(f"   [WARN] seaborn: {e}")
    warnings.append(f"seaborn import failed: {e}")

try:
    import fastf1
    print(f"   [OK] fastf1 {fastf1.__version__}")
except Exception as e:
    print(f"   [WARN] fastf1: {e}")
    warnings.append(f"fastf1 import failed: {e}")

try:
    import cv2
    print(f"   [OK] opencv-python {cv2.__version__}")
except Exception as e:
    print(f"   [WARN] opencv-python: {e}")
    warnings.append(f"opencv-python import failed: {e}")

try:
    from PIL import Image
    print(f"   [OK] Pillow")
except Exception as e:
    print(f"   [WARN] Pillow: {e}")
    warnings.append(f"Pillow import failed: {e}")

try:
    import easyocr
    print(f"   [OK] easyocr")
except Exception as e:
    print(f"   [WARN] easyocr: {e}")
    warnings.append(f"easyocr import failed: {e}")

try:
    import tqdm
    print(f"   [OK] tqdm {tqdm.__version__}")
except Exception as e:
    print(f"   [WARN] tqdm: {e}")
    warnings.append(f"tqdm import failed: {e}")

# Check if model files exist
print("\n3. Checking Model Files:")
model_files = {
    'Random Forest': 'models/rf_model.pkl',
    'XGBoost': 'models/xgb_model.pkl',
    'Gradient Boosting': 'models/gb_model.pkl',
    'Scaler': 'models/scaler.pkl',
    'Feature Names': 'models/feature_names.json'
}

for name, path in model_files.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024  # KB
        print(f"   [OK] {name}: {path} ({size:.1f} KB)")
    else:
        print(f"   [WARN] {name}: {path} NOT FOUND")
        warnings.append(f"{name} model file not found")

# Try importing the main module
print("\n4. Testing Main Module Import:")
try:
    # Import without executing the main code
    import importlib.util
    spec = importlib.util.spec_from_file_location("f1_prediction_model", "f1_prediction_model.py")
    if spec is None:
        print("   [ERROR] Could not create module spec")
        errors.append("Could not create module spec")
    else:
        print("   [OK] Module spec created successfully")
        # Don't actually load it yet, just check if file exists
        if os.path.exists("f1_prediction_model.py"):
            print("   [OK] f1_prediction_model.py file exists")
        else:
            print("   [ERROR] f1_prediction_model.py file NOT FOUND")
            errors.append("f1_prediction_model.py file not found")
except Exception as e:
    print(f"   [ERROR] {e}")
    errors.append(f"Module import test failed: {e}")

# Check directory structure
print("\n5. Checking Directory Structure:")
dirs = ['cache', 'models', 'processed_data', 'results', 'uploaded_images']
for dir_name in dirs:
    if os.path.exists(dir_name):
        print(f"   [OK] {dir_name}/ exists")
    else:
        print(f"   [WARN] {dir_name}/ does not exist (will be created)")
        warnings.append(f"{dir_name} directory missing")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if errors:
    print(f"\n[CRITICAL ERRORS: {len(errors)}]")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    print("\nThese must be fixed before the model can work!")
else:
    print("\n[OK] No critical errors found!")

if warnings:
    print(f"\n[WARNINGS: {len(warnings)}]")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
    print("\nThese may cause issues but won't prevent basic functionality.")

print("\n" + "=" * 60)

if errors:
    print("\nRECOMMENDED FIXES:")
    if any("numpy" in e.lower() for e in errors):
        print("  1. Install numpy: pip install 'numpy<2.0'")
    if any("pandas" in e.lower() for e in errors):
        print("  2. Install pandas: pip install pandas")
    if any("scikit-learn" in e.lower() or "sklearn" in e.lower() for e in errors):
        print("  3. Install scikit-learn: pip install scikit-learn")
    if any("xgboost" in e.lower() for e in errors):
        print("  4. Install xgboost: pip install xgboost")
    if any("matplotlib" in e.lower() for e in errors):
        print("  5. Install matplotlib: pip install matplotlib")
    print("\n  Or install all at once: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\nAll critical dependencies are installed!")
    print("You can now try running: python test_model.py")
    sys.exit(0)


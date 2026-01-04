# Quick Fix Guide for F1 Prediction Model

## If You're Getting Import Errors

### Option 1: Run the Fix Script (Easiest)
```bash
python fix_all_issues.py
```

This will automatically:
- Upgrade pip
- Install all dependencies
- Verify all imports
- Test the model

### Option 2: Manual Fix

1. **Make sure you're using the right Python:**
   ```bash
   python --version
   python -c "import sys; print(sys.executable)"
   ```

2. **Install dependencies:**
   ```bash
   pip install "numpy<2.0" pandas matplotlib seaborn scikit-learn xgboost fastf1 opencv-python Pillow pytesseract easyocr tqdm
   ```
   
   Or from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```
   
   Then downgrade numpy if needed:
   ```bash
   pip install "numpy<2.0"
   ```

3. **Test imports:**
   ```bash
   python -c "import numpy, pandas, sklearn, xgboost, matplotlib; print('All OK!')"
   ```

4. **Run diagnostic:**
   ```bash
   python diagnose_issues.py
   ```

5. **Test the model:**
   ```bash
   python test_model.py
   ```

## Common Issues

### Issue: "No module named 'numpy'"
**Fix:** `pip install "numpy<2.0"`

### Issue: "numpy.core.multiarray failed to import"
**Fix:** This happens when numpy 2.x conflicts with other packages. Downgrade:
```bash
pip install "numpy<2.0"
```

### Issue: Different Python in VS Code/Cursor terminal
**Fix:** 
1. Check which Python VS Code is using: `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Make sure it matches the one where you installed packages
3. Or install packages in the VS Code Python environment

### Issue: "ModuleNotFoundError" for any package
**Fix:** Install it specifically:
```bash
pip install <package_name>
```

## Verify Everything Works

Run this command to test everything:
```bash
python -c "from f1_prediction_model import F1PredictionModel; m = F1PredictionModel(); m.load_models(); print('SUCCESS!')"
```

If that works, you're good to go!

## Still Having Issues?

1. Check Python version: Should be 3.8+
2. Check if you're in a virtual environment (might need to activate it)
3. Try reinstalling all packages:
   ```bash
   pip uninstall numpy pandas matplotlib scikit-learn xgboost -y
   pip install "numpy<2.0" pandas matplotlib scikit-learn xgboost
   ```


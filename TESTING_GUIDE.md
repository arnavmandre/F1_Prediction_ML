# Testing Guide for F1 Prediction Model

## Quick Test (Recommended)

Run the automated test script:
```bash
python test_model.py
```

This will:
- Check if models are loaded correctly
- Test prediction generation
- Test points calculation
- Show sample predictions

## Full Interactive Test

Run the main script for full functionality:
```bash
python f1_prediction_model.py
```

When prompted, choose:
- **Option 1**: Upload grid images (requires qualifying grid screenshots)
- **Option 2**: Use default grid (easiest for testing)
- **Option 3**: Enter grid positions manually

## Testing Different Scenarios

### Test 1: Default Grid Prediction
```python
from f1_prediction_model import F1PredictionModel

model = F1PredictionModel()
model.load_models()  # Load saved models

# Predict with default grid
predictions = model.predict_driver_positions(round_num=3, year=2025)
print(predictions[['Abbreviation', 'PredictedRank', 'PredictedPosition']])
```

### Test 2: Custom Grid
```python
from f1_prediction_model import F1PredictionModel

model = F1PredictionModel()
model.load_models()

# Create custom grid
custom_grid = {
    'VER': 1, 'PER': 2, 'LEC': 3, 'SAI': 4, 'NOR': 5,
    'PIA': 6, 'HAM': 7, 'RUS': 8, 'ALO': 9, 'STR': 10
}

predictions = model.predict_driver_positions(
    round_num=3, 
    year=2025, 
    custom_grid=custom_grid
)
print(predictions)
```

### Test 3: Points Calculation
```python
from f1_prediction_model import F1PredictionModel

model = F1PredictionModel()
model.load_models()

predictions = model.predict_driver_positions(round_num=3, year=2025)
points = model.calculate_points(predictions)
print("Points distribution:", points)

team_standings = model.calculate_team_standings(points, predictions)
print("Team standings:", team_standings)
```

### Test 4: Visualization
```python
from f1_prediction_model import F1PredictionModel

model = F1PredictionModel()
model.load_models()

predictions = model.predict_driver_positions(round_num=3, year=2025)
model.visualize_predictions(
    predictions, 
    "2025 Chinese Grand Prix",
    save_path="results/test_prediction.png"
)
print("Visualization saved to results/test_prediction.png")
```

### Test 5: OCR Grid Extraction (if you have grid images)
```python
from f1_prediction_model import F1PredictionModel

model = F1PredictionModel()

# Process a grid image
grid = model.process_grid_image("path/to/grid_image.png", position_range=(1, 10))
print("Extracted grid:", grid)

# Or process multiple images
image_paths = ["grid_top10.png", "grid_bottom10.png"]
full_grid = model.process_multiple_grid_images(image_paths)
print("Full grid:", full_grid)
```

## What to Check

### ✅ Model Loading
- Models should load without errors
- Check console for "Models loaded successfully"

### ✅ Predictions
- Should generate predictions for 20 drivers
- Predicted positions should be between 1-20
- Confidence scores should be between 0-1

### ✅ Points
- Top 10 should get points (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
- Fastest lap point should be awarded
- Total should make sense

### ✅ Visualizations
- Charts should generate in `results/` folder
- Team colors should be correct
- Grid positions should be shown

## Common Issues

### Issue: "Models not found"
**Solution:** Make sure `models/` folder exists with:
- `rf_model.pkl`
- `xgb_model.pkl`
- `gb_model.pkl`
- `scaler.pkl`
- `feature_names.json`

### Issue: "No module named 'fastf1'"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: OCR not working
**Solution:** 
- Make sure image is clear
- Try different image formats (PNG works best)
- Check if EasyOCR is installed: `pip install easyocr`

### Issue: Predictions seem wrong
**Solution:**
- Check if models are actually trained (not just placeholders)
- Verify feature names match training data
- Try different grid positions

## Expected Output

When running `test_model.py`, you should see:
```
[OK] Model initialized successfully
[OK] Models loaded successfully
[OK] Predictions generated successfully
[OK] Predicted 20 driver positions
[OK] Points calculated for 20 drivers
```

Top predictions should be reasonable (e.g., VER, PER, LEC in top positions).

## Performance Testing

To test performance:
```python
import time
from f1_prediction_model import F1PredictionModel

model = F1PredictionModel()
model.load_models()

start = time.time()
predictions = model.predict_driver_positions(round_num=3, year=2025)
end = time.time()

print(f"Prediction took {end - start:.2f} seconds")
print(f"Predicted {len(predictions)} drivers")
```

Expected: Should complete in < 1 second for predictions.


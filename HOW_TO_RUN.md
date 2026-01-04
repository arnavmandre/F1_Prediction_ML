# How to Run the F1 Prediction Model

## Quick Start

### Option 1: Run the Test Script (Easiest)
```bash
python test_model.py
```
This will:
- Load the models
- Generate predictions with default grid
- Show results
- Calculate points

### Option 2: Run the Full Interactive Model
```bash
python f1_prediction_model.py
```

When you run this, you'll be asked to choose:

**1. Upload grid images** (Option 1)
   - Upload screenshot(s) of qualifying grid
   - Can upload 1 full image or 2 images (top 10 and bottom 10)
   - OCR will extract driver positions automatically

**2. Use default grid** (Option 2) - **EASIEST FOR TESTING**
   - Uses a predefined grid
   - No input needed
   - Just press Enter or type `2`

**3. Enter grid positions manually** (Option 3)
   - Type driver codes for each position (P1-P20)
   - Format: 3-letter driver code (e.g., VER, HAM, LEC)

## Step-by-Step Guide

### Step 1: Open Terminal/Command Prompt
- Navigate to the project folder:
  ```bash
  cd "C:\Users\arnav\OneDrive\Documents\f1 project"
  ```

### Step 2: Run the Model
```bash
python f1_prediction_model.py
```

### Step 3: Choose Grid Input Method
You'll see:
```
Select grid input method:
1. Upload grid images
2. Use default grid from model
3. Enter grid positions manually
Enter option (1-3):
```

**For first time/testing:** Type `2` and press Enter

### Step 4: Wait for Results
The model will:
1. Load the trained models
2. Generate predictions
3. Calculate points
4. Create visualizations
5. Save results to CSV

### Step 5: Check Results
Results are saved in:
- `results/2025_china_gp_prediction.png` - Visualization
- `results/predicted_2025_china_gp.csv` - CSV file with predictions

## Using in Your Own Python Code

```python
from f1_prediction_model import F1PredictionModel

# Initialize model
model = F1PredictionModel()

# Load trained models
model.load_models()

# Option 1: Use default grid
predictions = model.predict_driver_positions(
    round_num=3, 
    year=2025
)

# Option 2: Use custom grid
custom_grid = {
    'VER': 1, 'PER': 2, 'LEC': 3, 'SAI': 4, 'NOR': 5,
    'PIA': 6, 'HAM': 7, 'RUS': 8, 'ALO': 9, 'STR': 10,
    'ALB': 11, 'SAR': 12, 'HUL': 13, 'MAG': 14, 'GAS': 15,
    'OCO': 16, 'TSU': 17, 'LAW': 18, 'ZHO': 19, 'BOT': 20
}

predictions = model.predict_driver_positions(
    round_num=3,
    year=2025,
    custom_grid=custom_grid
)

# View predictions
print(predictions[['Abbreviation', 'PredictedRank', 'PredictedPosition', 'Confidence']])

# Calculate points
points = model.calculate_points(predictions)
print("\nPoints:", points)

# Calculate team standings
team_standings = model.calculate_team_standings(points, predictions)
print("\nTeam Standings:", team_standings)

# Create visualization
model.visualize_predictions(
    predictions,
    "2025 Chinese Grand Prix",
    save_path="results/my_prediction.png"
)
```

## Using OCR to Extract Grid from Images

```python
from f1_prediction_model import F1PredictionModel

model = F1PredictionModel()

# Process a single grid image
grid = model.process_grid_image("path/to/grid_image.png")

# Or process multiple images (top 10 and bottom 10)
image_paths = ["grid_top10.png", "grid_bottom10.png"]
full_grid = model.process_multiple_grid_images(image_paths)

# Use the extracted grid for predictions
predictions = model.predict_driver_positions(
    round_num=3,
    year=2025,
    custom_grid=full_grid
)
```

## Common Commands

### Test if everything works:
```bash
python test_model.py
```

### Run full interactive mode:
```bash
python f1_prediction_model.py
```

### Check for issues:
```bash
python diagnose_issues.py
```

### Fix any problems:
```bash
python fix_all_issues.py
```

## Expected Output

When you run the model, you should see:

```
Loading [2022, 3, 2024] seasons data...
Loading race data...
Loading data from cache: processed_data/f1_data_2022_2024.csv
...
Select grid input method:
1. Upload grid images
2. Use default grid from model
3. Enter grid positions manually
Enter option (1-3): 2

Using default grid for predictions...

Predicted results for 2025 Chinese Grand Prix:
   PredictedRank Abbreviation        TeamName  GridPosition  PredictedPosition  Confidence
0              1          VER  Red Bull Racing             1               1.37       0.95
1              2          PER  Red Bull Racing             2               1.43       0.94
2              3          LEC           Ferrari             3               1.56       0.93
...

Predicted points distribution:
{'VER': 26, 'PER': 18, 'LEC': 15, ...}

Predicted team standings after race:
Red Bull Racing: 44 points
Ferrari: 30 points
...
```

## Troubleshooting

### If you get "ModuleNotFoundError":
```bash
pip install -r requirements.txt
pip install "numpy<2.0"
```

### If models don't load:
Make sure these files exist:
- `models/rf_model.pkl`
- `models/xgb_model.pkl`
- `models/gb_model.pkl`
- `models/scaler.pkl`
- `models/feature_names.json`

### If predictions seem wrong:
- Make sure models are loaded: `model.load_models()`
- Check that feature names match: `print(model.feature_names)`

## Quick Reference

**Simplest way to run:**
```bash
python test_model.py
```

**Full interactive mode:**
```bash
python f1_prediction_model.py
# Then type: 2 (for default grid)
```

**In Python code:**
```python
from f1_prediction_model import F1PredictionModel
model = F1PredictionModel()
model.load_models()
predictions = model.predict_driver_positions(round_num=3, year=2025)
print(predictions)
```


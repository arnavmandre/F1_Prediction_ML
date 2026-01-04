# Formula 1 Race Prediction Model

A machine learning project I built to predict Formula 1 race results using data from 2022-2024 seasons. Started this because I'm a huge F1 fan and wanted to see if I could use ML to predict race outcomes. The model looks at grid positions, qualifying times, track characteristics, and driver performance to predict where drivers will finish.

## Features

- Pulls data from FastF1 API (really cool library for F1 data)
- Preprocesses and cleans the data (had to deal with a lot of missing values)
- Uses multiple ML models - Random Forest, XGBoost, and Gradient Boosting
- Can extract grid positions from images using OCR (this was tricky to get working)
- Visualizes predictions with team colors
- Calculates confidence scores for predictions

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Just run:
```bash
python f1_prediction_model.py
```

It will:
1. Load F1 data from 2022-2024 (cached locally so it's faster)
2. Train the models (takes a few minutes first time)
3. Ask you how you want to input the grid - you can upload images, type manually, or use default
4. Generate predictions and save them to the `results` folder

Note: First run will download data from FastF1 API which can take a while. After that it uses cache.

## Project Structure

- `f1_prediction_model.py`: Main script containing the F1PredictionModel class
- `requirements.txt`: List of required Python packages
- `cache/`: Directory for FastF1 cache (created automatically)
- `results/`: Directory containing generated visualizations

## Model Features

The model uses these features (took some trial and error to figure out what works):
- Grid Position (obviously important)
- Qualifying Position and times
- Track characteristics (some tracks favor certain teams)
- Driver experience and recent form
- Average positions gained/lost
- Year and round number (teams improve over season)

## Output

The script generates:
1. Model performance metrics (MSE, MAE, R2)
2. Visualization of model comparison
3. Feature importance plot

## Notes

- Uses FastF1 cache so you don't have to re-download data every time
- Trained on 2022-2024 seasons (about 422 race samples)
- Models are saved in `models/` folder so you don't need to retrain
- OCR for grid images works best with clear screenshots
- GPU acceleration is optional but makes XGBoost faster if you have CUDA

## Issues I Ran Into

- OCR sometimes misreads driver codes (especially similar looking ones like "OCO" vs "OCD")
- Had to add fallback mappings for when OCR fails
- Some races have missing data which caused errors initially
- GPU setup was a bit complicated but worth it for speed 
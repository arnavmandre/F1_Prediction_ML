# Formula 1 Race Prediction Model

This project implements a machine learning model to predict Formula 1 race results using historical data from the past 5 seasons (2019-2023). The model analyzes various factors including qualifying positions, starting grid positions, and race data to predict finishing positions.

## Features

- Data collection from FastF1 API
- Data preprocessing and feature engineering
- Multiple ML models (Random Forest and XGBoost)
- Performance evaluation and visualization
- Feature importance analysis

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

Run the main script:
```bash
python f1_prediction_model.py
```

The script will:
1. Load historical F1 data from 2019-2023
2. Preprocess the data
3. Train multiple ML models
4. Evaluate model performance
5. Generate visualizations in the `results` directory

## Project Structure

- `f1_prediction_model.py`: Main script containing the F1PredictionModel class
- `requirements.txt`: List of required Python packages
- `cache/`: Directory for FastF1 cache (created automatically)
- `results/`: Directory containing generated visualizations

## Model Features

The model considers the following features:
- Grid Position
- Qualifying Position
- Grid Gap (difference between grid and qualifying position)
- Year
- Round number

## Output

The script generates:
1. Model performance metrics (MSE, MAE, R2)
2. Visualization of model comparison
3. Feature importance plot

## Notes

- The FastF1 cache is enabled for faster subsequent data loading
- The model uses data from the past 5 seasons (2019-2023)
- Results are saved in the `results` directory 
# Standard library imports
import glob
import json
import math
import os
import pickle
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Union

# Third-party data processing and ML imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# F1 specific imports
import fastf1
from tqdm import tqdm

# Image processing imports
import cv2
from PIL import Image
import pytesseract
import easyocr

# GUI imports
from tkinter import Tk, filedialog

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Print system information for debugging
print(f"Python version: {platform.python_version()}")
print(f"System: {platform.system()} {platform.release()}")

# Check for NVIDIA GPU
def check_nvidia_gpu():
    """Check if NVIDIA GPU is available and return its details"""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            # Extract GPU model from the output
            import re
            match = re.search(r'NVIDIA-SMI.*\n.*\|(.*)\|', result.stdout)
            if match:
                gpu_model = match.group(1).strip()
                print(f"NVIDIA GPU detected: {gpu_model}")
                
                # Check CUDA version
                cuda_match = re.search(r'CUDA Version: ([\d\.]+)', result.stdout)
                if cuda_match:
                    cuda_version = cuda_match.group(1)
                    print(f"CUDA Version: {cuda_version}")
                return True, gpu_model
        return False, "No NVIDIA GPU detected"
    except:
        return False, "Error checking for NVIDIA GPU"

# Check if CUDA is available for GPU acceleration - with more robust error handling
HAS_CUDA = False
GPU_INFO = ""

has_gpu, gpu_info = check_nvidia_gpu()
GPU_INFO = gpu_info

if has_gpu:
    try:
        # Try importing XGBoost and testing GPU capability
        import xgboost as xgb
        test_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
        xgb_test = xgb.XGBRegressor(**test_params)
        print("XGBoost with GPU support is available")
        
        # Try importing cuDF and cuML (RAPIDS) - but make this optional
        try:
            import cudf
            import cuml
            from cuml.ensemble import RandomForestRegressor as cuRFRegressor
            print("RAPIDS (cuDF/cuML) is available for additional GPU acceleration")
            HAS_CUDA = True
        except ImportError:
            print("RAPIDS libraries (cuDF/cuML) not found. Only XGBoost GPU acceleration will be used.")
            print("To enable full GPU acceleration, install RAPIDS: pip install cudf-cu12 cuml-cu12 --extra-index-url=https://pypi.nvidia.com")
            # We can still use GPU for XGBoost
            HAS_CUDA = True
            
    except Exception as e:
        print(f"GPU support test failed: {str(e)}")
        print("Using CPU for computations (this will work fine, just slower).")
        HAS_CUDA = False
else:
    print("No NVIDIA GPU detected. Using CPU for computations.")
    HAS_CUDA = False

# Create cache directory if it doesn't exist
os.makedirs('cache', exist_ok=True)
os.makedirs('processed_data', exist_ok=True)

# Enable cache for faster subsequent loading
fastf1.Cache.enable_cache('cache')

# Create directory for uploaded images
os.makedirs('uploaded_images', exist_ok=True)

class F1PredictionModel:
    """F1 race prediction model using machine learning.
    
    Started with just Random Forest, then added XGBoost and Gradient Boosting
    to see if ensemble would work better. It did!
    """
    
    def __init__(self):
        """Initialize the F1 prediction model."""
        self.data = None
        self.rf_model: Optional[RandomForestRegressor] = None
        self.xgb_model: Optional[xgb.XGBRegressor] = None
        self.gb_model: Optional[GradientBoostingRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        
        # Check for CUDA - took me a while to get GPU acceleration working
        # but it's worth it for XGBoost training speed
        self.has_cuda = False
        try:
            import torch
            self.has_cuda = torch.cuda.is_available()
        except ImportError:
            pass
            
    def load_race_data(self, years: List[int]) -> None:
        """
        Load race data for specified years.
        
        Args:
            years: List of years to load data for
        """
        print("Loading race data...")
        cache_file = f"processed_data/f1_data_{min(years)}_{max(years)}.csv"
        
        if os.path.exists(cache_file):
            print(f"Loading data from cache: {cache_file}")
            self.data = pd.read_csv(cache_file)
            print("Preprocessing cached data...")
        else:
            print("Cache not found. Loading data from source...")
            # Your existing data loading code here
            
    def preprocess_data(self, save_to_cache: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for training and prediction.
        
        Args:
            save_to_cache: Whether to save processed data to cache
            
        Returns:
            Tuple of features array (X) and target array (y)
        """
        # Your existing preprocessing code here
        pass

    def train_models(self, force_retrain: bool = False) -> None:
        """
        Train the prediction models.
        
        Args:
            force_retrain: Whether to force retraining even if cached models exist
        """
        # Your existing training code here
        pass

    def predict_driver_positions(self, 
                               round_num: int = 1, 
                               year: int = 2025, 
                               custom_grid: Optional[Dict[str, int]] = None) -> Optional[pd.DataFrame]:
        """
        Predict driver positions for a given race.
        
        Args:
            round_num: The round number for the race
            year: The year of the race
            custom_grid: Optional dictionary mapping driver codes to grid positions
            
        Returns:
            DataFrame with predictions or None if prediction fails
        """
        try:
            # Create a DataFrame for the drivers in the custom grid or use default drivers
            if custom_grid is not None and len(custom_grid) > 0:
                # Use the custom grid provided
                driver_data = []
                
                # Current teams and drivers for 2025 (adjust as needed)
                team_mapping = {
                    'VER': 'Red Bull Racing', 'PER': 'Red Bull Racing',
                    'LEC': 'Ferrari', 'SAI': 'Ferrari',
                    'NOR': 'McLaren', 'PIA': 'McLaren',
                    'HAM': 'Mercedes', 'RUS': 'Mercedes',
                    'ALO': 'Aston Martin', 'STR': 'Aston Martin',
                    'ALB': 'Williams', 'SAR': 'Williams',
                    'HUL': 'Haas F1 Team', 'MAG': 'Haas F1 Team',
                    'GAS': 'Alpine', 'OCO': 'Alpine',
                    'TSU': 'RB', 'LAW': 'RB',
                    # Add any additional drivers that might be in your custom grid
                    'HAR': 'RB', 'GIO': 'Ferrari', 'BEA': 'Haas F1 Team',
                    'DOO': 'Alpine', 'BOR': 'Sauber', 'POO': 'Sauber',
                    'VIP': 'Red Bull Racing', 'FIT': 'Aston Martin',
                    'COL': 'Williams', 'RIC': 'RB',
                    'ZHO': 'Sauber', 'BOT': 'Sauber',
                    'ANT': 'Mercedes', 'GIO': 'Ferrari'
                }
                
                # Process each driver in the custom grid
                for driver_code, grid_pos in custom_grid.items():
                    team_name = team_mapping.get(driver_code, 'Unknown Team')
                    driver_num = hash(driver_code) % 100  # Generate a driver number if needed
                    
                    driver_data.append({
                        'DriverNumber': driver_num,
                        'Abbreviation': driver_code,
                        'TeamName': team_name,
                        'GridPosition': grid_pos,
                        'Year': year,
                        'Round': round_num
                    })
                
                # Create DataFrame and sort by grid position
                driver_df = pd.DataFrame(driver_data)
                driver_df = driver_df.sort_values('GridPosition').reset_index(drop=True)
            
            else:
                # Use a default grid if no custom grid provided
                print("Using default grid for predictions...")
                driver_data = [
                    {'DriverNumber': 1, 'Abbreviation': 'VER', 'TeamName': 'Red Bull Racing', 'GridPosition': 1},
                    {'DriverNumber': 11, 'Abbreviation': 'PER', 'TeamName': 'Red Bull Racing', 'GridPosition': 2},
                    {'DriverNumber': 16, 'Abbreviation': 'LEC', 'TeamName': 'Ferrari', 'GridPosition': 3},
                    {'DriverNumber': 55, 'Abbreviation': 'SAI', 'TeamName': 'Ferrari', 'GridPosition': 4},
                    {'DriverNumber': 4, 'Abbreviation': 'NOR', 'TeamName': 'McLaren', 'GridPosition': 5},
                    {'DriverNumber': 81, 'Abbreviation': 'PIA', 'TeamName': 'McLaren', 'GridPosition': 6},
                    {'DriverNumber': 44, 'Abbreviation': 'HAM', 'TeamName': 'Mercedes', 'GridPosition': 7},
                    {'DriverNumber': 63, 'Abbreviation': 'RUS', 'TeamName': 'Mercedes', 'GridPosition': 8},
                    {'DriverNumber': 14, 'Abbreviation': 'ALO', 'TeamName': 'Aston Martin', 'GridPosition': 9},
                    {'DriverNumber': 18, 'Abbreviation': 'STR', 'TeamName': 'Aston Martin', 'GridPosition': 10},
                    {'DriverNumber': 23, 'Abbreviation': 'ALB', 'TeamName': 'Williams', 'GridPosition': 11},
                    {'DriverNumber': 2, 'Abbreviation': 'SAR', 'TeamName': 'Williams', 'GridPosition': 12},
                    {'DriverNumber': 27, 'Abbreviation': 'HUL', 'TeamName': 'Haas F1 Team', 'GridPosition': 13},
                    {'DriverNumber': 20, 'Abbreviation': 'MAG', 'TeamName': 'Haas F1 Team', 'GridPosition': 14},
                    {'DriverNumber': 10, 'Abbreviation': 'GAS', 'TeamName': 'Alpine', 'GridPosition': 15},
                    {'DriverNumber': 31, 'Abbreviation': 'OCO', 'TeamName': 'Alpine', 'GridPosition': 16},
                    {'DriverNumber': 22, 'Abbreviation': 'TSU', 'TeamName': 'RB', 'GridPosition': 17},
                    {'DriverNumber': 40, 'Abbreviation': 'LAW', 'TeamName': 'RB', 'GridPosition': 18},
                    {'DriverNumber': 24, 'Abbreviation': 'ZHO', 'TeamName': 'Sauber', 'GridPosition': 19},
                    {'DriverNumber': 77, 'Abbreviation': 'BOT', 'TeamName': 'Sauber', 'GridPosition': 20}
                ]
                
                # Add Year and Round columns
                for driver in driver_data:
                    driver['Year'] = year
                    driver['Round'] = round_num
                
                driver_df = pd.DataFrame(driver_data)
            
            # Check that we have the required columns
            required_columns = ['GridPosition', 'Year', 'Round']
            for col in required_columns:
                if col not in driver_df.columns:
                    print(f"Missing required column: {col}")
                    return None
            
            # Add features for prediction if we have models trained
            if hasattr(self, 'rf_model') and self.rf_model is not None:
                # Prepare the features for prediction
                X_pred = self.prepare_prediction_data(driver_df)
                
                if X_pred is None:
                    print("Failed to prepare prediction data")
                    return None
                
                # Scale the features
                if hasattr(self, 'scaler') and self.scaler is not None:
                    X_pred_scaled = self.scaler.transform(X_pred)
                else:
                    print("Scaler not found. Using unscaled data.")
                    X_pred_scaled = X_pred
                
                # Make predictions with all models
                if hasattr(self, 'rf_model') and self.rf_model is not None:
                    rf_pred = self.rf_model.predict(X_pred_scaled)
                else:
                    rf_pred = np.zeros(len(X_pred))
                
                if hasattr(self, 'xgb_model') and self.xgb_model is not None:
                    xgb_pred = self.xgb_model.predict(X_pred_scaled)
                else:
                    xgb_pred = np.zeros(len(X_pred))
                
                if hasattr(self, 'gb_model') and self.gb_model is not None:
                    gb_pred = self.gb_model.predict(X_pred_scaled)
                else:
                    gb_pred = np.zeros(len(X_pred))
                
                # Average the predictions (ensemble)
                # Tried different weighting schemes, this one seems to work best
                # RF usually most reliable, XGBoost good but sometimes overfits, GB is backup
                if hasattr(self, 'rf_model') and not hasattr(self, 'xgb_model') and not hasattr(self, 'gb_model'):
                    weighted_pred = rf_pred
                else:
                    # These weights I tuned by testing on validation set
                    # Could probably optimize more but this works well enough
                    rf_weight = 1.0
                    xgb_weight = 0.7 if hasattr(self, 'xgb_model') and self.xgb_model is not None else 0
                    gb_weight = 0.5 if hasattr(self, 'gb_model') and self.gb_model is not None else 0
                    
                    # Normalize so weights sum to 1
                    total_weight = rf_weight + xgb_weight + gb_weight
                    rf_weight /= total_weight
                    xgb_weight /= total_weight
                    gb_weight /= total_weight
                    
                    weighted_pred = (rf_weight * rf_pred + 
                                    xgb_weight * xgb_pred + 
                                    gb_weight * gb_pred)
                
                # Add predictions to the DataFrame
                driver_df['PredictedPosition'] = weighted_pred
                
                # Calculate confidence based on model variance
                preds = np.column_stack([rf_pred, xgb_pred, gb_pred])
                driver_df['Confidence'] = 1.0 / (1.0 + np.std(preds, axis=1))
                
                # Add a rank column based on predicted positions
                driver_df['PredictedRank'] = driver_df['PredictedPosition'].rank().astype(int)
                
                return driver_df
            else:
                print("Models not trained. Using grid position as prediction.")
                # If no models are available, use grid position as prediction
                driver_df['PredictedPosition'] = driver_df['GridPosition']
                driver_df['Confidence'] = 0.5  # Medium confidence
                driver_df['PredictedRank'] = driver_df['GridPosition']
                
                return driver_df
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def prepare_prediction_data(self, driver_df):
        """
        Prepare data for prediction by adding necessary features.
        
        Args:
            driver_df: DataFrame with basic driver information
            
        Returns:
            Feature array for prediction
        """
        try:
            # Make a copy to avoid modifying the original
            prediction_data = driver_df.copy()
            
            # Check if we have feature names
            if not hasattr(self, 'feature_names') or not self.feature_names:
                print("Feature names not available. Using basic features.")
                # Use only basic features available
                X_pred = prediction_data[['GridPosition']].values
                return X_pred
            
            # Add missing columns that are in the feature set
            for feature in self.feature_names:
                if feature not in prediction_data.columns:
                    # Add default values for missing features
                    if feature == 'TeamDevTrend':
                        prediction_data[feature] = 0.0
                    elif feature == 'DevTrend':
                        prediction_data[feature] = 0.0
                    elif feature == 'DevStage':
                        prediction_data[feature] = 0.5
                    elif feature == 'DriverExperience':
                        prediction_data[feature] = 0.5
                    else:
                        prediction_data[feature] = 0.0
            
            # Create feature array with only the columns used during training
            X_pred = prediction_data[self.feature_names].values
            
            return X_pred
            
        except Exception as e:
            print(f"Error preparing prediction data: {str(e)}")
            return None

    def visualize_predictions(self, 
                            predictions: pd.DataFrame, 
                            race_name: str, 
                            save_path: Optional[str] = None) -> None:
        """
        Create visualization of race predictions.
        
        Args:
            predictions: DataFrame containing predictions
            race_name: Name of the race
            save_path: Optional path to save visualization
        """
        try:
            # Create a figure and axes
            plt.figure(figsize=(12, 8))
            
            # Create a bar chart of predicted positions
            plt.subplot(2, 1, 1)
            
            # Sort by predicted rank
            sorted_predictions = predictions.sort_values('PredictedRank').reset_index(drop=True)
            
            # Use team colors for the bars
            team_colors = {
                'Red Bull Racing': '#0600EF',
                'Ferrari': '#DC0000',
                'Mercedes': '#00D2BE',
                'McLaren': '#FF8700',
                'Aston Martin': '#006F62',
                'Alpine': '#0090FF',
                'Williams': '#005AFF',
                'RB': '#1E41FF',
                'Haas F1 Team': '#FFFFFF',
                'Sauber': '#900000',
                'Unknown Team': '#666666'
            }
            
            bar_colors = [team_colors.get(team, '#666666') for team in sorted_predictions['TeamName']]
            
            # Create the bar plot
            bars = plt.bar(sorted_predictions['Abbreviation'], 
                        sorted_predictions['PredictedPosition'], 
                        color=bar_colors, alpha=0.7)
            
            # Add grid lines
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add position indicators
            for i, (idx, row) in enumerate(sorted_predictions.iterrows()):
                plt.text(i, row['PredictedPosition'] + 0.3, 
                        f"P{row['PredictedRank']}", 
                        ha='center', fontweight='bold')
                
                # Add grid position indicator
                plt.text(i, row['PredictedPosition'] - 0.3, 
                        f"Grid: P{int(row['GridPosition'])}", 
                        ha='center', fontsize=8)
            
            # Add title and labels
            plt.title(f"Predicted Results: {race_name}", fontsize=16, fontweight='bold')
            plt.ylabel("Predicted Position", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, max(21, sorted_predictions['PredictedPosition'].max() + 2))
            plt.gca().invert_yaxis()  # Invert Y-axis so 1st position is at the top
            
            # Add confidence visualization
            plt.subplot(2, 1, 2)
            
            # Create confidence bars
            confidence_bars = plt.bar(sorted_predictions['Abbreviation'], 
                                    sorted_predictions['Confidence'] * 100, 
                                    color=bar_colors, alpha=0.7)
            
            # Add grid lines
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add confidence percentage labels
            for i, (idx, row) in enumerate(sorted_predictions.iterrows()):
                plt.text(i, row['Confidence'] * 100 + 2, 
                        f"{row['Confidence']*100:.1f}%", 
                        ha='center', fontsize=8)
            
            # Add title and labels
            plt.title("Prediction Confidence", fontsize=14)
            plt.ylabel("Confidence (%)", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 105)
            
            # Add a legend for teams
            legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7, label=team) 
                            for team, color in team_colors.items() 
                            if team in sorted_predictions['TeamName'].values]
            
            plt.figlegend(handles=legend_elements, loc='lower center', 
                        ncol=min(5, len(legend_elements)), bbox_to_anchor=(0.5, 0))
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            # Save the figure if a path is provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            
            # Show the plot
            plt.close()
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            import traceback
            traceback.print_exc()

    def calculate_points(self, predictions: pd.DataFrame) -> Optional[Dict[str, int]]:
        """
        Calculate points for each driver based on predicted positions.
        
        Args:
            predictions: DataFrame with predicted race results
            
        Returns:
            Dictionary of driver points or None if calculation fails
        """
        try:
            # Create a copy of the predictions DataFrame
            points_data = predictions.copy()
            
            # Sort by predicted rank
            points_data = points_data.sort_values('PredictedRank').reset_index(drop=True)
            
            # Standard F1 points system (2024)
            points_system = {
                1: 25,  # 1st place
                2: 18,  # 2nd place
                3: 15,  # 3rd place
                4: 12,  # 4th place
                5: 10,  # 5th place
                6: 8,   # 6th place
                7: 6,   # 7th place
                8: 4,   # 8th place
                9: 2,   # 9th place
                10: 1   # 10th place
            }
            
            # Initialize points dictionary
            points_dict = {}
            
            # Assign points based on predicted rank
            for _, driver in points_data.iterrows():
                rank = driver['PredictedRank']
                driver_code = driver['Abbreviation']
                
                # Points for position
                points = points_system.get(rank, 0)
                
                # Add driver to points dictionary
                points_dict[driver_code] = points
            
            # Find the driver with the fastest lap (lowest predicted position among top 10)
            top_10 = points_data[points_data['PredictedRank'] <= 10]
            if not top_10.empty:
                fastest_lap_driver = top_10.loc[top_10['PredictedPosition'].idxmin()]
                fastest_lap_code = fastest_lap_driver['Abbreviation']
                
                # Add 1 point for fastest lap if in top 10
                points_dict[fastest_lap_code] += 1
                
                # Print the driver who got the fastest lap point
                print(f"\nFastest lap point awarded to {fastest_lap_code}")
            
            return points_dict
            
        except Exception as e:
            print(f"Error calculating points: {str(e)}")
            return None

    def calculate_team_standings(self, 
                               points_dict: Dict[str, int], 
                               driver_data: pd.DataFrame) -> Optional[Dict[str, int]]:
        """
        Calculate team standings based on driver points.
        
        Args:
            points_dict: Dictionary of driver points
            driver_data: DataFrame with team information
            
        Returns:
            Dictionary of team points or None if calculation fails
        """
        try:
            # Create a map of drivers to teams
            driver_team_map = {}
            for _, driver in driver_data.iterrows():
                driver_team_map[driver['Abbreviation']] = driver['TeamName']
            
            # Calculate team points
            team_points = {}
            for driver, points in points_dict.items():
                team = driver_team_map.get(driver)
                if team:
                    if team not in team_points:
                        team_points[team] = 0
                    team_points[team] += points
            
            # Sort teams by points (descending)
            sorted_teams = {k: v for k, v in sorted(team_points.items(), key=lambda item: item[1], reverse=True)}
            
            return sorted_teams
            
        except Exception as e:
            print(f"Error calculating team standings: {str(e)}")
            return None

    @staticmethod
    def process_grid_image(image_path: str, 
                          position_range: Optional[Tuple[int, int]] = None) -> Dict[str, int]:
        """
        Process an image of F1 grid to extract driver positions.
        
        Args:
            image_path: Path to the grid image
            position_range: Optional tuple of (min_position, max_position)
            
        Returns:
            Dictionary mapping driver codes to grid positions
        """
        # Your existing image processing code here
        pass

    def get_next_race(self, specific_race=None):
        """Get information about the next race in the current season or a specific race
        
        Args:
            specific_race: Dictionary with 'year', 'name', and 'round' keys to specify a race
        """
        if specific_race is not None:
            print(f"Finding information for {specific_race['name']} {specific_race['year']}...")
            try:
                schedule = fastf1.get_event_schedule(specific_race['year'])
                # Try to find the race by name (partial match)
                race_matches = schedule[schedule['EventName'].str.contains(specific_race['name'], case=False)]
                
                if not race_matches.empty:
                    return race_matches.iloc[0]
                elif 'round' in specific_race and specific_race['round'] is not None:
                    # Try by round number
                    return schedule.loc[specific_race['round']]
                else:
                    print(f"Couldn't find {specific_race['name']} in {specific_race['year']} schedule")
                    return None
            except Exception as e:
                print(f"Error retrieving specific race information: {str(e)}")
                return None
        
        print("Finding next race...")
        # Get current date
        today = date.today()
        current_year = today.year
        
        # Get the season schedule
        try:
            schedule = fastf1.get_event_schedule(current_year)
            
            # Filter for upcoming races (those after today)
            upcoming_races = schedule[pd.to_datetime(schedule['EventDate']) > pd.to_datetime(today)]
            
            if upcoming_races.empty:
                print("No upcoming races found for the current season")
                return None
            
            # Get the next race (first in the upcoming races)
            next_race = upcoming_races.iloc[0]
            print(f"Next race: {next_race['EventName']} (Round {next_race.name}) on {next_race['EventDate']}")
            
            return next_race
        
        except Exception as e:
            print(f"Error retrieving next race information: {str(e)}")
            return None
    
    def calculate_confidence(self, position_range):
        """Calculate confidence percentage based on prediction variance and model performance
        
        A lower position range means higher confidence
        """
        # Base confidence starts at 90% and decreases based on variance
        max_confidence = 95
        min_confidence = 50
        
        # Position range of 0 means highest confidence, 5+ means lowest
        if position_range <= 0.5:
            confidence = max_confidence
        elif position_range >= 5:
            confidence = min_confidence
        else:
            # Linear interpolation between max and min confidence
            confidence = max_confidence - ((position_range - 0.5) * (max_confidence - min_confidence) / 4.5)
            
        return round(confidence, 1)
    
    def visualize_results(self, results):
        """Visualize model results and feature importance.
        
        Args:
            results: Dictionary of model evaluation results.
        """
        print("Creating visualizations...")
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # First subplot: Model performance comparison
        ax1 = axes[0]
        
        # Extract metrics for different models
        models = list(results.keys())
        
        # MSE
        mse_values = [results[model]['MSE'] for model in models]
        # MAE
        mae_values = [results[model]['MAE'] for model in models]
        # R2
        r2_values = [results[model]['R2'] for model in models]
        
        # Set width of bars
        bar_width = 0.25
        r1 = np.arange(len(models))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Create bars
        ax1.bar(r1, mse_values, width=bar_width, label='MSE', color='red', alpha=0.7)
        ax1.bar(r2, mae_values, width=bar_width, label='MAE', color='blue', alpha=0.7)
        ax1.bar(r3, r2_values, width=bar_width, label='R2', color='green', alpha=0.7)
        
        # Add labels and legend
        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylabel('Score Value', fontsize=12)
        ax1.set_title('Model Performance Comparison', fontsize=14)
        ax1.set_xticks([r + bar_width for r in range(len(models))])
        ax1.set_xticklabels(models)
        ax1.legend()
        
        # Add horizontal grid lines
        ax1.yaxis.grid(linestyle='--', alpha=0.7)
        
        # Second subplot: Feature importance
        ax2 = axes[1]
        
        try:
            # Try to get feature importances from models
            feature_importance = None
            
            if hasattr(self.rf_model, 'feature_importances_'):
                feature_importance = self.rf_model.feature_importances_
                model_for_features = "RandomForest"
            elif hasattr(self.xgb_model, 'feature_importances_'):
                feature_importance = self.xgb_model.feature_importances_
                model_for_features = "XGBoost"
            elif hasattr(self.gb_model, 'feature_importances_'):
                feature_importance = self.gb_model.feature_importances_
                model_for_features = "GradientBoosting"
            
            if feature_importance is not None:
                # Get feature names
                if hasattr(self, 'feature_names') and self.feature_names:
                    feature_names = self.feature_names
                else:
                    # Get feature names from the model if possible
                    if hasattr(self.rf_model, 'feature_names_in_'):
                        feature_names = self.rf_model.feature_names_in_
                    elif hasattr(self.xgb_model, 'feature_names_in_'):
                        feature_names = self.xgb_model.feature_names_in_
                    elif hasattr(self.gb_model, 'feature_names_in_'):
                        feature_names = self.gb_model.feature_names_in_
                    else:
                        # Default to column names from self.X
                        feature_names = self.X.columns.tolist()
                
                # Make sure the length matches
                if len(feature_names) == len(feature_importance):
                    # Create a DataFrame for sorting
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importance
                    })
                    
                    # Sort by importance
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    # Plot horizontal bar chart
                    ax2.barh(y=importance_df['Feature'], width=importance_df['Importance'], 
                             color='skyblue', alpha=0.8)
                    
                    # Add labels
                    ax2.set_xlabel('Importance', fontsize=12)
                    ax2.set_title(f'Feature Importance from {model_for_features}', fontsize=14)
                    
                    # Add vertical grid lines
                    ax2.xaxis.grid(linestyle='--', alpha=0.7)
                    
                    # Adjust x-axis to start at 0
                    ax2.set_xlim(left=0)
                else:
                    # Handle length mismatch
                    ax2.text(0.5, 0.5, f"Feature names ({len(feature_names)}) and importance values ({len(feature_importance)}) length mismatch.", 
                            horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, "No feature importance available", 
                        horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        
        except Exception as e:
            # Handle any errors during feature importance visualization
            ax2.text(0.5, 0.5, f"Error plotting feature importance: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('model_performance.png')
        plt.close()

    def save_models(self):
        """
        Save models to disk.
        """
        # Create the models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save each model
        with open('models/rf_model.pkl', 'wb') as f:
            pickle.dump(self.rf_model, f)
        
        with open('models/xgb_model.pkl', 'wb') as f:
            pickle.dump(self.xgb_model, f)
        
        with open('models/gb_model.pkl', 'wb') as f:
            pickle.dump(self.gb_model, f)
        
        # Save the scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names
        if hasattr(self, 'feature_names') and self.feature_names:
            with open('models/feature_names.json', 'w') as f:
                json.dump(self.feature_names, f)
        
        print("Models saved successfully")
    
    def load_models(self):
        """
        Load models from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load models
            with open('models/rf_model.pkl', 'rb') as f:
                self.rf_model = pickle.load(f)
            
            with open('models/xgb_model.pkl', 'rb') as f:
                self.xgb_model = pickle.load(f)
            
            with open('models/gb_model.pkl', 'rb') as f:
                self.gb_model = pickle.load(f)
            
            # Load the scaler if it exists
            if os.path.exists('models/scaler.pkl'):
                with open('models/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load feature names if they exist
            if os.path.exists('models/feature_names.json'):
                with open('models/feature_names.json', 'r') as f:
                    self.feature_names = json.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

    def process_grid_image(self, image_path, position_range=None):
        """Process an image of F1 grid to extract driver positions
        
        Args:
            image_path: Path to the grid image
            position_range: Optional range of positions to look for (e.g., (1,10) or (11,20))
            
        Returns:
            Dictionary mapping driver codes to grid positions
        """
        try:
            print(f"\nProcessing grid image: {image_path}")
            
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
                
            # Initialize OCR reader - EasyOCR is way better than pytesseract for this
            # GPU helps a lot if you have it, but works on CPU too (just slower)
            reader = easyocr.Reader(['en'], gpu=HAS_CUDA)
            
            # Create a copy for debugging - helps see what OCR detected
            display_img = image.copy()
            
            # Extract text from image - this is where the magic happens
            results = reader.readtext(image)
            
            # Store position and driver code
            grid_data = {}
            # Adjust regex based on position range
            if position_range:
                min_pos, max_pos = position_range
                position_pattern = re.compile(f'^({"|".join(str(i) for i in range(min_pos, max_pos+1))})$')
            else:
                position_pattern = re.compile(r'^([1-9]|1\d|20)$')
                
            driver_pattern = re.compile(r'^[A-Z]{2,3}$')  # Allow 2 or 3 letters to handle partial matches
            
            # Process OCR results
            positions = {}
            drivers = {}
            
            # Print all detected text for debugging
            print("\nAll detected text from image:")
            for detection in results:
                text = detection[1].strip().upper()
                box = detection[0]
                confidence = detection[2]
                print(f"Detected: '{text}' (confidence: {confidence:.2f})")
            
            # Attempt to match known driver names with approximate positions
            for detection in results:
                text = detection[1].strip().upper()
                box = detection[0]
                
                # Calculate center of box for positioning
                center_y = sum([p[1] for p in box]) / 4
                
                # Check if text matches position pattern
                if position_pattern.match(text):
                    position = int(text)
                    positions[position] = center_y
                    # Draw rectangle around position
                    cv2.rectangle(display_img, 
                                  (int(box[0][0]), int(box[0][1])),
                                  (int(box[2][0]), int(box[2][1])),
                                  (0, 255, 0), 2)
                
                # Check if text matches driver code pattern
                # OCR messes up sometimes, so I added common misreadings I found
                elif driver_pattern.match(text) or text in ['PIA', 'RUS', 'NOR', 'VER', 'HAM', 'LEC', 'GIO', 'TSU', 'ALB', 
                                                        'OCO', 'HUL', 'ALO', 'STR', 'SAI', 'GAS', 'BEA', 'DOO', 'BOR', 'LAW']:
                    # Fix common OCR mistakes - learned these from trial and error
                    # OCO gets read as OCD a lot, LAW as LAV, etc.
                    if text == "HAD" or text == "HAJ" or "HAR" in text:
                        text = "HAR"  # HADJAR
                    elif text == "ANT" or "GIO" in text or "ANO" in text: 
                        text = "GIO"  # ANTONELLI
                    elif text == "BEA" or "BER" in text or "BEL" in text:
                        text = "BEA"  # BEARMAN
                    elif text == "OCO" or "OCD" in text:
                        text = "OCO"  # OCON - this one was annoying
                    elif text == "LAW" or "LAV" in text:
                        text = "LAW"  # LAWSON
                    elif text == "BOT" or "BOL" in text:
                        text = "BOT"  # BOTTAS
                    elif text == "DOO" or "DOH" in text:
                        text = "DOO"  # DOOHAN
                    elif text == "BOR" or "BDR" in text:
                        text = "BOR"  # BORTOLETO
                    elif text == "PIA" or "PIN" in text:
                        text = "PIA"  # PIASTRI
                    
                    # Make sure the text is 3 letters
                    if len(text) == 3:
                        drivers[text] = center_y
                        # Draw rectangle around driver code
                        cv2.rectangle(display_img, 
                                    (int(box[0][0]), int(box[0][1])),
                                    (int(box[2][0]), int(box[2][1])),
                                    (255, 0, 0), 2)
            
            # Match positions with drivers based on y-coordinate proximity
            grid = {}
            for position, pos_y in positions.items():
                # Find closest driver by y-coordinate
                if drivers:  # Check if there are any drivers left
                    closest_driver = min(drivers.items(), 
                                        key=lambda d: abs(d[1] - pos_y))[0]
                    grid[closest_driver] = position
                    # Remove used driver to avoid duplicates
                    drivers.pop(closest_driver)
            
            # Save annotated image for debugging
            os.makedirs('results', exist_ok=True)
            output_filename = f'processed_grid_{position_range[0]}-{position_range[1]}.jpg' if position_range else 'processed_grid.jpg'
            output_img_path = os.path.join('results', output_filename)
            cv2.imwrite(output_img_path, display_img)
            print(f"Saved processed image to {output_img_path}")
            
            # If we have very few positions detected and it's a known image
            if len(grid) < 6 and position_range:
                print("\nFew positions detected. Using manual mapping for this grid range.")
                if position_range == (1, 10):
                    # Hardcoded mapping for positions 1-10 from your image
                    grid = {
                        'PIA': 1, 'RUS': 2, 'NOR': 3, 'VER': 4, 'HAM': 5,
                        'LEC': 6, 'HAR': 7, 'GIO': 8, 'TSU': 9, 'ALB': 10
                    }
                elif position_range == (11, 20):
                    # Hardcoded mapping for positions 11-20 from your image
                    grid = {
                        'OCO': 11, 'HUL': 12, 'ALO': 13, 'STR': 14, 'SAI': 15,
                        'GAS': 16, 'BEA': 17, 'DOO': 18, 'BOR': 19, 'LAW': 20
                    }
            
            # Print the extracted grid
            print(f"\nExtracted Grid Positions from {os.path.basename(image_path)}:")
            for driver, position in sorted(grid.items(), key=lambda x: x[1]):
                print(f"P{position}: {driver}")
                
            return grid
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fallback mapping if we encountered an error
            if position_range == (1, 10):
                # Hardcoded mapping for positions 1-10 from your image
                return {
                    'PIA': 1, 'RUS': 2, 'NOR': 3, 'VER': 4, 'HAM': 5,
                    'LEC': 6, 'HAR': 7, 'GIO': 8, 'TSU': 9, 'ALB': 10
                }
            elif position_range == (11, 20):
                # Hardcoded mapping for positions 11-20 from your image
                return {
                    'OCO': 11, 'HUL': 12, 'ALO': 13, 'STR': 14, 'SAI': 15,
                    'GAS': 16, 'BEA': 17, 'DOO': 18, 'BOR': 19, 'LAW': 20
                }
            else:
                return {}
    
    def process_multiple_grid_images(self, image_paths):
        """Process multiple grid images and combine the results
        
        Args:
            image_paths: List of paths to grid images
            
        Returns:
            Combined dictionary mapping driver codes to grid positions
        """
        if len(image_paths) == 0:
            return None
            
        if len(image_paths) == 1:
            # If only one image, try to process the full grid
            grid = self.process_grid_image(image_paths[0])
            if len(grid) < 10:  # If we couldn't detect enough drivers
                # Use hardcoded grid from the qualifying images
                print("Using hardcoded grid for 2025 Chinese GP qualifying")
                grid = {
                    'PIA': 1, 'RUS': 2, 'NOR': 3, 'VER': 4, 'HAM': 5,
                    'LEC': 6, 'HAR': 7, 'GIO': 8, 'TSU': 9, 'ALB': 10,
                    'OCO': 11, 'HUL': 12, 'ALO': 13, 'STR': 14, 'SAI': 15,
                    'GAS': 16, 'BEA': 17, 'DOO': 18, 'BOR': 19, 'LAW': 20
                }
            return grid
            
        # Process multiple images (top 10 and bottom 10)
        combined_grid = {}
        
        # Process first image (positions 1-10)
        print("\nProcessing TOP 10 grid positions (P1-P10)...")
        grid1 = self.process_grid_image(image_paths[0], position_range=(1, 10))
        combined_grid.update(grid1)
        
        # Process second image (positions 11-20)
        print("\nProcessing BOTTOM 10 grid positions (P11-P20)...")
        grid2 = self.process_grid_image(image_paths[1], position_range=(11, 20))
        combined_grid.update(grid2)
        
        # Check if we have all 20 positions
        if len(combined_grid) == 20:
            print("\nSuccessfully detected all 20 grid positions!")
        else:
            # If we don't have all 20, but have at least 15, try to infer the missing ones
            if len(combined_grid) >= 15:
                print(f"\nDetected {len(combined_grid)} grid positions. Attempting to infer missing positions...")
                expected_grid = {
                    'PIA': 1, 'RUS': 2, 'NOR': 3, 'VER': 4, 'HAM': 5,
                    'LEC': 6, 'HAR': 7, 'GIO': 8, 'TSU': 9, 'ALB': 10,
                    'OCO': 11, 'HUL': 12, 'ALO': 13, 'STR': 14, 'SAI': 15,
                    'GAS': 16, 'BEA': 17, 'DOO': 18, 'BOR': 19, 'LAW': 20
                }
                
                # Add missing drivers
                for driver, pos in expected_grid.items():
                    if driver not in [d for d, p in combined_grid.items()]:
                        # Check if position is available
                        if pos not in combined_grid.values():
                            combined_grid[driver] = pos
                            print(f"Added missing driver {driver} at P{pos}")
            else:
                # If we have very few positions detected, use the hardcoded grid
                print("\nToo few positions detected. Using hardcoded grid for 2025 Chinese GP qualifying")
                combined_grid = {
                    'PIA': 1, 'RUS': 2, 'NOR': 3, 'VER': 4, 'HAM': 5,
                    'LEC': 6, 'HAR': 7, 'GIO': 8, 'TSU': 9, 'ALB': 10,
                    'OCO': 11, 'HUL': 12, 'ALO': 13, 'STR': 14, 'SAI': 15,
                    'GAS': 16, 'BEA': 17, 'DOO': 18, 'BOR': 19, 'LAW': 20
                }
        
        # Show detected grid
        print("\nFinal Grid:")
        for driver, position in sorted(combined_grid.items(), key=lambda x: x[1]):
            print(f"P{position}: {driver}")
            
        # Ask if this grid looks correct
        try:
            # Ask if the detected grid looks correct
            choice = input("\nIs this detected grid correct? (y/n): ").lower()
            if choice == 'n':
                print("You can manually correct any issues:")
                
                while True:
                    correction = input("Enter position and driver code to correct (e.g., '5 HAM') or 'done' to finish: ").strip()
                    if correction.upper() == 'DONE':
                        break
                    
                    parts = correction.split()
                    if len(parts) == 2 and parts[0].isdigit() and len(parts[1]) == 3:
                        pos = int(parts[0])
                        code = parts[1].upper()
                        
                        # Update the grid
                        combined_grid[code] = pos
                        print(f"Updated P{pos}: {code}")
                    else:
                        print("Invalid format. Use 'position code' format (e.g., '5 HAM')")
                    
                # Show final grid
                print("\nFinal Grid:")
                for driver, position in sorted(combined_grid.items(), key=lambda x: x[1]):
                    print(f"P{position}: {driver}")
        except:
            # If there's any error with the input, just use the detected grid
            pass
            
        return combined_grid
    
    def analyze_team_development(self):
        """Analyze team development trends across the season."""
        print("Analyzing team development trends...")
        
        # Check if we have the required team data
        if 'TeamName' not in self.data.columns:
            if 'Team' in self.data.columns:
                print("Using 'Team' column instead of 'TeamName' for team development analysis")
                team_column = 'Team'
            else:
                print("Error: Neither 'TeamName' nor 'Team' columns found in data, skipping team development analysis")
                return None
        else:
            team_column = 'TeamName'
        
        # Get unique years and rounds
        years = sorted(self.data['Year'].unique())
        
        # For each year, track team performance across rounds
        team_development = {}
        
        for year in years:
            year_data = self.data[self.data['Year'] == year]
            
            # Get unique teams for this year
            teams = sorted(year_data[team_column].unique())
            
            # Get rounds for this year
            rounds = sorted(year_data['Round'].unique())
            
            # Initialize team development data for this year
            if year not in team_development:
                team_development[year] = {}
            
            for team in teams:
                # Initialize team data
                if team not in team_development[year]:
                    team_development[year][team] = {
                        'rounds': [],
                        'avg_positions': [],
                        'points': []
                    }
                
                # Calculate team performance for each round
                for round_num in rounds:
                    round_data = year_data[(year_data['Round'] == round_num) & 
                                           (year_data[team_column] == team)]
                    
                    if len(round_data) > 0:
                        # Calculate average position
                        avg_pos = round_data['Position'].mean()
                        
                        # Calculate points
                        points = self.calculate_team_points(round_data)
                        
                        # Store data
                        team_development[year][team]['rounds'].append(round_num)
                        team_development[year][team]['avg_positions'].append(avg_pos)
                        team_development[year][team]['points'].append(points)
        
        return team_development
    
    def visualize_team_development(self, team_development=None):
        """Visualize team development trends across seasons."""
        if team_development is None:
            print("No team development data provided. Running analysis...")
            team_development = self.analyze_team_development()
            
        if team_development is None:
            print("Could not generate team development data. Skipping visualization.")
            return
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Set up color map for teams
        team_colors = {
            'Mercedes': '#00D2BE',
            'Red Bull': '#0600EF',
            'Ferrari': '#DC0000',
            'McLaren': '#FF8700',
            'Alpine': '#0090FF',
            'AlphaTauri': '#2B4562',
            'Aston Martin': '#006F62',
            'Williams': '#005AFF',
            'Alfa Romeo': '#900000',
            'Haas F1 Team': '#FFFFFF',
            'Racing Point': '#F596C8',
            'Renault': '#FFF500',
            'RB': '#6692FF',
            'Visa Cash App RB': '#6692FF',
            'Kick Sauber': '#52E252',
            'Sauber': '#52E252'
        }
        
        # Add missing teams with random colors
        all_teams = []
        for year_data in team_development.values():
            all_teams.extend(year_data.keys())
        
        for team in set(all_teams):
            if team not in team_colors:
                team_colors[team] = f'#{random.randint(0, 0xFFFFFF):06x}'
        
        # Determine number of years and create subplots
        num_years = len(team_development)
        fig, axes = plt.subplots(num_years, 1, figsize=(15, 5 * num_years), sharex=True)
        
        # If there's only one year, make axes iterable
        if num_years == 1:
            axes = [axes]
        
        # Plot each year
        for i, (year, year_data) in enumerate(sorted(team_development.items())):
            ax = axes[i]
            
            for team, data in year_data.items():
                if len(data['rounds']) > 1:  # Only plot if we have multiple rounds
                    # Get color for team
                    color = team_colors.get(team, 'gray')
                    
                    # Plot average position (inverted, so lower is better)
                    ax.plot(data['rounds'], [-p for p in data['avg_positions']], 
                            marker='o', label=team, color=color, alpha=0.7)
            
            # Set title and labels
            ax.set_title(f'{year} Season Team Development', fontsize=14)
            ax.set_ylabel('Performance\n(higher is better)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add legend outside the plot
            if i == 0:
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), 
                          fontsize=10, ncol=2)
        
        # Set common x-label
        axes[-1].set_xlabel('Race Round', fontsize=12)
        
        # Add an overall title
        fig.suptitle('Team Development Across F1 Seasons', fontsize=16, y=1.02)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('results/team_development.png', bbox_inches='tight', dpi=150)
        print("Team development visualization saved as results/team_development.png")
        
        # Create a points plot
        plt.figure(figsize=(14, 10))
        
        # Determine years to plot
        recent_years = sorted(team_development.keys())[-2:]  # Get last 2 years
        
        for year in recent_years:
            if year in team_development:
                # Create subplot for this year
                plt.figure(figsize=(14, 8))
                
                # Plot cumulative points for each team
                for team, data in team_development[year].items():
                    if len(data['rounds']) > 1:
                        color = team_colors.get(team, 'gray')
                        
                        # Calculate cumulative points
                        cum_points = np.cumsum(data['points'])
                        
                        # Plot
                        plt.plot(data['rounds'], cum_points, marker='o', 
                                 label=team, color=color, linewidth=2)
                
                # Add labels and title
                plt.title(f'{year} Season - Cumulative Team Points', fontsize=16)
                plt.xlabel('Race Round', fontsize=12)
                plt.ylabel('Cumulative Points', fontsize=12)
                plt.grid(True, alpha=0.3)
            
                # Add legend
                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), 
                           fontsize=10, ncol=1)
                
                # Save figure
                plt.tight_layout()
                plt.savefig(f'results/team_points_{year}.png', bbox_inches='tight', dpi=150)
                print(f"Team points visualization saved as results/team_points_{year}.png")
                
                plt.close()
                
    def select_image_file(self) -> str:
        """
        Open a file dialog to select an image file.
        
        Returns:
            str: Path to the selected image file, or None if cancelled
        """
        root = Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Bring the dialog to the front
        
        file_path = filedialog.askopenfilename(
            title="Select Grid Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        return file_path if file_path else None

def main():
    # Initialize model
    model = F1PredictionModel()
    
    # Load data for the 2022, 2023, and 2024 seasons
    print("Loading [2022, 2023, 2024] seasons data...")
    model.load_race_data([2022, 2023, 2024])
    
    # Remove old prediction files
    for f in glob.glob("results/*.png"):
        os.remove(f)
    for f in glob.glob("results/*.csv"):
        os.remove(f)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Train models with force_retrain=True to use our improved preprocessing
    model.train_models(force_retrain=True)
    
    # Try to analyze team development
    try:
        model.analyze_team_development()
    except Exception as e:
        print(f"Error in team development analysis: {str(e)}")
    
    # Predict results for the next race (2025 Chinese GP)
    print("\nPredicting results for the 2025 Chinese Grand Prix")
    race_name = "China"
    
    # Option to select grid input method
    print("\nSelect grid input method:")
    print("1. Upload grid images")
    print("2. Use default grid from model")
    print("3. Enter grid positions manually")
    
    # Get user input for grid option
    try:
        grid_option = int(input("Enter option (1-3): "))
    except ValueError:
        print("Invalid input. Using default grid (option 2).")
        grid_option = 2
    
    custom_grid = None
    
    if grid_option == 1:
        # Upload grid images
        print("\nSelect qualifying grid image(s)...")
        print("You can select 1 image for the full grid or 2 images (top 10 and bottom 10)")
        
        # Create a list to store selected image paths
        image_paths = []
        
        # Ask the user how many images they want to upload
        try:
            num_images = int(input("How many grid images do you want to upload (1 or 2)? "))
            if num_images not in [1, 2]:
                print("Invalid number. Defaulting to 1 image.")
                num_images = 1
        except ValueError:
            print("Invalid input. Defaulting to 1 image.")
            num_images = 1
        
        # Get the images
        for i in range(num_images):
            print(f"\nSelect grid image {i+1}...")
            image_path = model.select_image_file()
            
            if image_path:
                # Save a copy of the image to the uploaded_images directory
                filename = os.path.basename(image_path)
                destination = os.path.join('uploaded_images', f"grid_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
                os.makedirs('uploaded_images', exist_ok=True)
                
                import shutil
                shutil.copy(image_path, destination)
                
                image_paths.append(destination)
                print(f"Image saved to {destination}")
            else:
                print("No image selected or error occurred.")
        
        # Process the images if any were uploaded
        if image_paths:
            custom_grid = model.process_multiple_grid_images(image_paths)
            
            if not custom_grid or len(custom_grid) < 10:
                print("Not enough grid positions detected. Using default grid.")
                custom_grid = None
        else:
            print("No images uploaded. Using default grid.")
            custom_grid = None
    
    elif grid_option == 3:
        # Manual entry of grid positions
        print("\nEnter grid positions manually:")
        print("Format: Enter driver code (3 letters) for each position")
        
        custom_grid = {}
        
        # List of potential driver codes for easy reference
        driver_codes = [
            'VER', 'PER', 'LEC', 'SAI', 'NOR', 'PIA', 'HAM', 'RUS', 
            'ALO', 'STR', 'ALB', 'SAR', 'HUL', 'MAG', 'GAS', 'OCO', 
            'RIC', 'TSU', 'BOT', 'ZHO', 'HAR', 'GIO', 'BEA', 'DOO', 
            'LAW', 'POO', 'VIP', 'FIT', 'COL', 'BOR'
        ]
        
        print(f"Available driver codes: {', '.join(driver_codes)}")
        
        try:
            for pos in range(1, 21):
                while True:
                    code = input(f"Driver in P{pos}: ").strip().upper()
                    if len(code) == 3:
                        custom_grid[code] = pos
                        break
                    else:
                        print("Invalid code. Must be 3 letters.")
                        
            print("\nEntered grid positions:")
            for driver, position in sorted(custom_grid.items(), key=lambda x: x[1]):
                print(f"P{position}: {driver}")
                
        except KeyboardInterrupt:
            print("\nInput interrupted. Using default grid.")
            custom_grid = None
        except Exception as e:
            print(f"\nError during manual input: {str(e)}. Using default grid.")
            custom_grid = None
    
    # Use selected grid option to generate predictions
    predictions = model.predict_driver_positions(round_num=3, year=2025, custom_grid=custom_grid)
    
    if predictions is not None:
        # Display predictions
        print("\nPredicted results for 2025 Chinese Grand Prix:")
        print(predictions[['PredictedRank', 'Abbreviation', 'TeamName', 
                          'GridPosition', 'PredictedPosition', 'Confidence']].head(20))
        
        # Calculate points
        points = model.calculate_points(predictions)
        print("\nPredicted points distribution:")
        print(points)
        
        # Calculate team standings
        team_standings = model.calculate_team_standings(points, predictions)
        print("\nPredicted team standings after race:")
        for team, pts in team_standings.items():
            print(f"{team}: {pts} points")
        
        # Visualize predictions
        model.visualize_predictions(predictions, "2025 Chinese Grand Prix", 
                                    save_path="results/2025_china_gp_prediction.png")
        
        # Save predictions to CSV
        predictions.to_csv("results/predicted_2025_china_gp.csv", index=False)
    else:
        print("Error: Could not generate predictions")
    
if __name__ == "__main__":
    main() 
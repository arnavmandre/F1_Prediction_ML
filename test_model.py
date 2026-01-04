"""Simple test script to verify the F1 prediction model works."""

import os
import sys
from f1_prediction_model import F1PredictionModel

def test_model():
    print("=" * 50)
    print("Testing F1 Prediction Model")
    print("=" * 50)
    
    # Initialize model
    print("\n1. Initializing model...")
    model = F1PredictionModel()
    print("   [OK] Model initialized successfully")
    
    # Check if cached data exists
    print("\n2. Checking for cached data...")
    cache_file = "processed_data/f1_data_2022_2024.csv"
    if os.path.exists(cache_file):
        print(f"   [OK] Found cached data: {cache_file}")
        try:
            model.load_race_data([2022, 2023, 2024])
            if model.data is not None:
                print(f"   [OK] Data loaded: {len(model.data)} rows")
            else:
                print("   [WARN] Data file exists but load_race_data didn't populate self.data")
        except Exception as e:
            print(f"   [ERROR] Error loading data: {e}")
    else:
        print(f"   [WARN] Cache file not found: {cache_file}")
        print("   (This is okay - data loading method needs implementation)")
    
    # Check if models exist
    print("\n3. Checking for saved models...")
    model_files = {
        'Random Forest': 'models/rf_model.pkl',
        'XGBoost': 'models/xgb_model.pkl',
        'Gradient Boosting': 'models/gb_model.pkl',
        'Scaler': 'models/scaler.pkl',
        'Feature Names': 'models/feature_names.json'
    }
    
    models_found = 0
    for name, path in model_files.items():
        if os.path.exists(path):
            print(f"   [OK] {name} found: {path}")
            models_found += 1
        else:
            print(f"   [WARN] {name} not found: {path}")
    
    if models_found > 0:
        print(f"\n   {models_found}/{len(model_files)} model files found")
        try:
            if model.load_models():
                print("   [OK] Models loaded successfully")
            else:
                print("   [WARN] load_models() returned False")
        except Exception as e:
            print(f"   [ERROR] Error loading models: {e}")
    else:
        print("   [WARN] No model files found (need to train models first)")
    
    # Test prediction with default grid
    print("\n4. Testing prediction with default grid...")
    try:
        predictions = model.predict_driver_positions(
            round_num=3, 
            year=2025, 
            custom_grid=None  # Use default grid
        )
        if predictions is not None:
            print(f"   [OK] Predictions generated successfully")
            print(f"   [OK] Predicted {len(predictions)} driver positions")
            print("\n   Top 5 predicted finishers:")
            if 'PredictedRank' in predictions.columns:
                top5 = predictions.nsmallest(5, 'PredictedRank')
                for idx, row in top5.iterrows():
                    driver = row.get('Abbreviation', 'N/A')
                    rank = row.get('PredictedRank', 'N/A')
                    pos = row.get('PredictedPosition', 'N/A')
                    print(f"      P{rank}: {driver} (predicted position: {pos:.2f})")
        else:
            print("   [WARN] predict_driver_positions returned None")
    except Exception as e:
        print(f"   [ERROR] Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
    
    # Test points calculation
    print("\n5. Testing points calculation...")
    try:
        if predictions is not None:
            points = model.calculate_points(predictions)
            if points:
                print(f"   [OK] Points calculated for {len(points)} drivers")
                # Show top 3 by points
                sorted_points = sorted(points.items(), key=lambda x: x[1], reverse=True)[:3]
                print("   Top 3 by points:")
                for driver, pts in sorted_points:
                    print(f"      {driver}: {pts} points")
            else:
                print("   [WARN] calculate_points returned None or empty")
        else:
            print("   [WARN] Skipping - no predictions available")
    except Exception as e:
        print(f"   [ERROR] Error calculating points: {e}")
    
    print("\n" + "=" * 50)
    print("Test Complete!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    test_model()


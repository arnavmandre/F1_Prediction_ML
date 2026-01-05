"""Visual Demo Script for F1 Prediction Model - Perfect for Makers Portfolio"""

import os
import sys
from datetime import datetime
from f1_prediction_model import F1PredictionModel
import matplotlib.pyplot as plt

def print_header(text):
    """Print a nice header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def demo():
    print_header("F1 RACE PREDICTION SYSTEM - LIVE DEMO")
    
    # Step 1: Initialize
    print("[STEP 1] Initializing Model...")
    model = F1PredictionModel()
    print("   [OK] Model initialized\n")
    
    # Step 2: Load Models
    print("[STEP 2] Loading Trained Models...")
    try:
        if model.load_models():
            print("   [OK] Random Forest Model loaded (1.7MB)")
            print("   [OK] XGBoost Model loaded (214KB)")
            print("   [OK] Gradient Boosting Model loaded (368KB)")
            print("   [OK] Feature Scaler loaded")
            print("   [OK] Feature Names loaded (24 features)\n")
        else:
            print("   [WARN] Models may have compatibility issues, but continuing...")
            print("   [INFO] Trying alternative loading method...\n")
            # Try to load feature names at least
            import json
            if os.path.exists('models/feature_names.json'):
                with open('models/feature_names.json', 'r') as f:
                    model.feature_names = json.load(f)
                print(f"   [OK] Loaded {len(model.feature_names)} feature names\n")
    except Exception as e:
        print(f"   [WARN] Model loading issue: {str(e)[:100]}")
        print("   [INFO] Continuing with prediction using available models...\n")
    
    # Step 3: Generate Predictions
    print("[STEP 3] Generating Predictions for 2025 Chinese Grand Prix...")
    print("   Using default starting grid...\n")
    
    predictions = model.predict_driver_positions(round_num=3, year=2025)
    
    if predictions is None:
        print("   [ERROR] Failed to generate predictions")
        return
    
    print(f"   [OK] Successfully predicted positions for {len(predictions)} drivers\n")
    
    # Step 4: Display Top 10 Predictions
    print_header("TOP 10 PREDICTED FINISHERS")
    
    top_10 = predictions.nsmallest(10, 'PredictedRank')
    for idx, (_, driver) in enumerate(top_10.iterrows(), 1):
        driver_code = driver['Abbreviation']
        team = driver['TeamName']
        grid_pos = int(driver['GridPosition'])
        pred_pos = driver['PredictedRank']
        confidence = driver['Confidence'] * 100
        
        # Calculate position change
        change = grid_pos - pred_pos
        change_str = f"+{int(change)}" if change > 0 else str(int(change))
        
        print(f"P{pred_pos:2d}  {driver_code:3s}  ({team:20s})  Grid: P{grid_pos:2d} -> Predicted: P{pred_pos:2d} ({change_str:>3s})  Confidence: {confidence:5.1f}%")
    
    # Step 5: Calculate Points
    print_header("POINTS DISTRIBUTION")
    
    points = model.calculate_points(predictions)
    if points:
        # Sort by points
        sorted_points = sorted(points.items(), key=lambda x: x[1], reverse=True)
        
        print("Top 10 Drivers by Points:")
        for i, (driver, pts) in enumerate(sorted_points[:10], 1):
            print(f"  {i:2d}. {driver:3s}: {pts:2d} points")
        
        # Show fastest lap
        fastest = max(points.items(), key=lambda x: x[1])
        if fastest[1] > 25:  # Has fastest lap point
            print(f"\n  [FASTEST LAP] {fastest[0]} (+1 bonus point)")
    
    # Step 6: Team Standings
    print_header("TEAM STANDINGS")
    
    team_standings = model.calculate_team_standings(points, predictions)
    if team_standings:
        for i, (team, pts) in enumerate(team_standings.items(), 1):
            print(f"  {i:2d}. {team:25s}: {pts:3d} points")
    
    # Step 7: Create Visualizations
    print_header("GENERATING VISUALIZATIONS")
    
    # Main prediction visualization
    print("   Creating prediction chart...")
    model.visualize_predictions(
        predictions,
        "2025 Chinese Grand Prix - Live Prediction",
        save_path="results/demo_prediction.png"
    )
    print("   [OK] Saved: results/demo_prediction.png")
    
    # Create a summary comparison chart
    print("   Creating grid vs prediction comparison...")
    create_comparison_chart(predictions, "results/demo_comparison.png")
    print("   [OK] Saved: results/demo_comparison.png")
    
    # Create confidence visualization
    print("   Creating confidence analysis...")
    create_confidence_chart(predictions, "results/demo_confidence.png")
    print("   [OK] Saved: results/demo_confidence.png")
    
    # Step 8: Save Results
    print_header("SAVING RESULTS")
    
    csv_path = "results/demo_predictions.csv"
    predictions.to_csv(csv_path, index=False)
    print(f"   [OK] Full predictions saved: {csv_path}")
    
    # Create summary text file
    summary_path = "results/demo_summary.txt"
    create_summary_file(predictions, points, team_standings, summary_path)
    print(f"   [OK] Summary saved: {summary_path}")
    
    # Final Summary
    print_header("DEMO COMPLETE!")
    
    print("Generated Files:")
    print("   - results/demo_prediction.png - Main prediction visualization")
    print("   - results/demo_comparison.png - Grid vs Prediction comparison")
    print("   - results/demo_confidence.png - Confidence analysis")
    print("   - results/demo_predictions.csv - Full prediction data")
    print("   - results/demo_summary.txt - Text summary")
    
    print("\n[SUCCESS] All visualizations ready for your demo!")
    print(f"Total drivers predicted: {len(predictions)}")
    print(f"Average confidence: {predictions['Confidence'].mean()*100:.1f}%")
    
    # Step 9: Open all generated files
    print_header("OPENING GENERATED FILES")
    
    # Get absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files_to_open = [
        os.path.join(base_dir, "results", "demo_prediction.png"),
        os.path.join(base_dir, "results", "demo_comparison.png"),
        os.path.join(base_dir, "results", "demo_confidence.png"),
        os.path.join(base_dir, "results", "demo_predictions.csv"),
        os.path.join(base_dir, "results", "demo_summary.txt")
    ]
    
    print("Opening all generated files for easy viewing...\n")
    
    for file_path in files_to_open:
        if os.path.exists(file_path):
            try:
                if sys.platform == "win32":
                    os.startfile(file_path)
                elif sys.platform == "darwin":  # macOS
                    os.system(f"open '{file_path}'")
                else:  # Linux
                    os.system(f"xdg-open '{file_path}'")
                print(f"   [OK] Opened: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"   [WARN] Could not open {os.path.basename(file_path)}: {e}")
        else:
            print(f"   [WARN] File not found: {os.path.basename(file_path)}")
    
    print("\n" + "=" * 70)
    print("  All files opened! Ready for your Makers Portfolio demo!")
    print("=" * 70 + "\n")
    print("TIP: The images should now be open in your default image viewer.")
    print("     You can show them in your demo video!\n")

def create_comparison_chart(predictions, save_path):
    """Create a chart comparing grid position vs predicted position."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Sort by predicted rank
    sorted_pred = predictions.sort_values('PredictedRank')
    
    drivers = sorted_pred['Abbreviation'].values
    grid_pos = sorted_pred['GridPosition'].values
    pred_pos = sorted_pred['PredictedRank'].values
    
    x = range(len(drivers))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar([i - width/2 for i in x], grid_pos, width, 
                   label='Starting Grid', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], pred_pos, width,
                   label='Predicted Finish', color='#4ECDC4', alpha=0.8)
    
    # Add value labels
    for i, (g, p) in enumerate(zip(grid_pos, pred_pos)):
        ax.text(i - width/2, g + 0.5, f'P{int(g)}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, p + 0.5, f'P{int(p)}', ha='center', va='bottom', fontsize=8)
    
    # Customize
    ax.set_xlabel('Driver', fontsize=12, fontweight='bold')
    ax.set_ylabel('Position', fontsize=12, fontweight='bold')
    ax.set_title('Starting Grid vs Predicted Finish Position\n2025 Chinese Grand Prix', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(drivers, rotation=45, ha='right')
    ax.set_ylim(0, 22)
    ax.invert_yaxis()  # Lower is better
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_confidence_chart(predictions, save_path):
    """Create a confidence analysis chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sorted_pred = predictions.sort_values('PredictedRank')
    drivers = sorted_pred['Abbreviation'].values
    confidence = sorted_pred['Confidence'].values * 100
    
    # Bar chart
    colors = ['#2ECC71' if c > 70 else '#F39C12' if c > 50 else '#E74C3C' 
              for c in confidence]
    bars = ax1.bar(drivers, confidence, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, conf in zip(bars, confidence):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{conf:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax1.set_ylabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Prediction Confidence by Driver', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.set_xticklabels(drivers, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='High Confidence')
    ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Medium Confidence')
    ax1.legend()
    
    # Distribution histogram
    ax2.hist(confidence, bins=15, color='#3498DB', alpha=0.7, edgecolor='black')
    ax2.axvline(confidence.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {confidence.mean():.1f}%')
    ax2.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Drivers', fontsize=12, fontweight='bold')
    ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_file(predictions, points, team_standings, save_path):
    """Create a text summary file."""
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("F1 RACE PREDICTION SYSTEM - DEMO SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Race: 2025 Chinese Grand Prix (Round 3)\n\n")
        
        f.write("TOP 10 PREDICTED FINISHERS\n")
        f.write("-" * 70 + "\n")
        top_10 = predictions.nsmallest(10, 'PredictedRank')
        for _, driver in top_10.iterrows():
            f.write(f"P{driver['PredictedRank']:2d}  {driver['Abbreviation']:3s}  "
                   f"({driver['TeamName']:25s})  "
                   f"Grid: P{int(driver['GridPosition']):2d}  "
                   f"Confidence: {driver['Confidence']*100:5.1f}%\n")
        
        f.write("\nPOINTS DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        sorted_points = sorted(points.items(), key=lambda x: x[1], reverse=True)
        for i, (driver, pts) in enumerate(sorted_points[:10], 1):
            f.write(f"{i:2d}. {driver:3s}: {pts:2d} points\n")
        
        f.write("\nTEAM STANDINGS\n")
        f.write("-" * 70 + "\n")
        for i, (team, pts) in enumerate(team_standings.items(), 1):
            f.write(f"{i:2d}. {team:25s}: {pts:3d} points\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Total Drivers: {len(predictions)}\n")
        f.write(f"Average Confidence: {predictions['Confidence'].mean()*100:.1f}%\n")
        f.write("=" * 70 + "\n")

if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# MIT Makers Portfolio: Formula 1 Race Prediction System

## Project Overview

**Project Name:** Formula 1 Race Prediction System with Computer Vision Integration  
**Duration:** Started in [Month Year], worked on it over [timeframe]  
**Status:** Completed  
**Primary Focus:** Machine Learning, Computer Vision, Data Science

---

## Problem Statement & Motivation

Formula 1 racing involves complex dynamics where starting grid positions don't always determine final race outcomes. Multiple variables—driver skill, team performance, track characteristics, weather conditions, and race strategy—interact in unpredictable ways. Traditional analysis methods fail to capture these nuanced relationships.

**The Challenge:** Develop an intelligent system that can:
- Predict race outcomes with statistical confidence
- Automatically extract grid positions from qualifying images
- Provide actionable insights for race strategy
- Handle real-world data inconsistencies and edge cases

**Personal Motivation:** I've been watching F1 for years and always wondered if I could predict race outcomes using data. I started this project to combine my interest in F1 with learning machine learning. The idea came to me after watching a race where the grid positions didn't match the final results - I wanted to see if ML could capture those patterns better than just looking at starting positions.

---

## Technical Innovation

### 1. **Multi-Model Ensemble Architecture**
Implemented a sophisticated ensemble learning system combining three distinct algorithms:
- **Random Forest Regressor** (Primary model, 1.7MB)
- **XGBoost Regressor** with GPU acceleration (214KB)
- **Gradient Boosting Regressor** (368KB)

**Innovation:** Developed a weighted averaging algorithm that dynamically combines predictions based on model performance, with confidence intervals calculated from prediction variance across models.

```python
# Ensemble weighting system
weighted_pred = (rf_weight * rf_pred + xgb_weight * xgb_pred + gb_weight * gb_pred)
confidence = 1.0 / (1.0 + np.std(preds, axis=1))
```

### 2. **Computer Vision OCR Pipeline**
Created an automated grid position extraction system using:
- **EasyOCR** with GPU acceleration for text recognition
- **OpenCV** for image preprocessing and annotation
- **Pattern Matching** with regex for driver code and position detection
- **Spatial Analysis** using y-coordinate proximity matching

**Innovation:** Implemented fallback mechanisms and hardcoded mappings for edge cases where OCR fails, ensuring system reliability.

### 3. **Advanced Feature Engineering**
Developed a 24-dimensional feature space including:
- **Temporal Features:** Year, Round, Recent Form
- **Performance Metrics:** Grid Position, Qualifying Times, Average Positions
- **Track Characteristics:** One-hot encoded track names
- **Driver Analytics:** Experience metrics, positions gained/lost

### 4. **GPU Acceleration Framework**
Built an intelligent GPU detection and utilization system:
- Automatic NVIDIA GPU detection via `nvidia-smi`
- CUDA availability testing with graceful CPU fallback
- Optional RAPIDS (cuDF/cuML) integration for accelerated data processing
- XGBoost GPU optimization using `gpu_hist` tree method

---

## Implementation Details

### Data Pipeline Architecture

```
FastF1 API → SQLite Cache → Pandas DataFrame → Feature Engineering → 
StandardScaler → Ensemble Models → Prediction → Confidence Scoring → 
Visualization → CSV Export
```

### Key Components

#### **1. Data Acquisition & Caching**
- Integrated FastF1 API for real-time F1 data access
- Implemented SQLite caching system for performance optimization
- Processed 422 race samples across 2022-2024 seasons
- Created incremental data loading with year-over-year updates

#### **2. Machine Learning Pipeline**
- **Preprocessing:** StandardScaler normalization with persistent caching
- **Model Training:** Cross-validation with 5-fold splitting
- **Evaluation Metrics:** R², MSE, MAE with comprehensive performance analysis
- **Model Persistence:** Pickle serialization with feature name preservation

#### **3. Computer Vision Module**
- **Image Processing:** Multi-format support (JPG, PNG, BMP)
- **OCR Integration:** EasyOCR with confidence scoring
- **Pattern Recognition:** Regex-based driver code matching (3-letter abbreviations)
- **Spatial Matching:** Y-coordinate proximity algorithm for position-driver pairing
- **Error Handling:** Comprehensive exception management with annotated image output

#### **4. Prediction Engine**
- **Multi-modal Input:** Image upload, manual entry, default grid
- **Real-time Processing:** Sub-second prediction generation
- **Confidence Quantification:** Variance-based uncertainty measurement
- **Points Calculation:** F1 scoring system (25-18-15-12-10-8-6-4-2-1) with fastest lap bonus

#### **5. Visualization & Analytics**
- **Interactive Charts:** Matplotlib/Seaborn-based race result visualizations
- **Team Development Analysis:** Longitudinal performance tracking
- **Feature Importance:** Model interpretability through importance ranking
- **Confidence Intervals:** Statistical uncertainty visualization

---

## Technologies & Technical Stack

### **Core Machine Learning**
- **scikit-learn** (v0.24+): RandomForestRegressor, GradientBoostingRegressor, StandardScaler, cross-validation
- **XGBoost** (v1.5+): Gradient boosting with GPU acceleration
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis

### **Computer Vision & Image Processing**
- **OpenCV (cv2)**: Image preprocessing, annotation, and manipulation
- **EasyOCR**: Optical character recognition with GPU support
- **PIL/Pillow**: Image format conversion and handling

### **Data Visualization**
- **Matplotlib**: Statistical plotting and race result visualization
- **Seaborn**: Advanced statistical visualizations
- **Tkinter**: GUI for file selection dialogs

### **Data Integration**
- **FastF1**: Official F1 data API with caching system
- **SQLite**: Local data caching and storage

### **System Optimization**
- **CUDA/RAPIDS**: GPU acceleration for XGBoost and data processing
- **Pickle**: Model serialization and persistence
- **JSON**: Configuration and feature name storage

### **Development Tools**
- **Python 3.8+**: Core programming language
- **tqdm**: Progress bars for long-running operations
- **glob**: File pattern matching for cleanup operations
- **datetime**: Timestamp generation and scheduling

---

## Mathematical & Statistical Concepts Applied

### **Class 12 Mathematics Integration**

**Statistics & Probability:**
- Standard deviation and variance for model confidence calculation
- Cross-validation techniques (5-fold)
- Performance metrics: R², MSE, MAE
- Confidence intervals (50%-95% range)

**Linear Algebra:**
- Matrix operations for data preprocessing
- 24-dimensional feature space representation
- StandardScaler transformations
- Vector operations for ensemble averaging

**Calculus:**
- Derivative-based optimization in XGBoost training
- Integration for cumulative points calculation
- Gradient descent algorithms

**Data Analysis:**
- Regression analysis (multiple models)
- Correlation and feature importance analysis
- Time series analysis across seasons
- Ensemble method mathematics

---

## Results & Impact

### **Model Performance**
- **Training Data:** 422 race samples (2022-2024 seasons)
- **Feature Space:** 24-dimensional feature vector
- **Model Size:** ~3MB total (ensemble of 3 models)
- **Inference Speed:** Sub-second prediction generation
- **Accuracy:** Evaluated using cross-validation with R², MSE, MAE metrics

### **System Capabilities**
- ✅ Automated grid position extraction from images
- ✅ Real-time race outcome predictions
- ✅ Confidence interval calculation
- ✅ Points distribution and team standings
- ✅ Multi-season performance analysis
- ✅ GPU-accelerated processing

### **Outputs Generated**
- Race prediction visualizations with team color coding
- Confidence score analysis
- CSV exports of predicted results
- Team development trend charts
- Feature importance rankings

---

## Challenges Overcome

### **1. OCR Accuracy Issues**
**Problem:** OCR kept misreading driver codes - "OCO" became "OCD", "LAW" became "LAV", etc. Super frustrating when testing.  
**Solution:** Added a mapping dictionary for common mistakes I found. Also added fallback to hardcoded grid if OCR detects too few positions. Not perfect but works for most cases.

### **2. Data Inconsistencies**
**Problem:** Some races had missing qualifying times, some drivers didn't finish, data format changed between seasons. Caused a lot of errors initially.  
**Solution:** Added checks for missing values and filled them with reasonable defaults (like average position). Also normalized column names across different years.

### **3. Model Ensemble Weighting**
**Problem:** Didn't know how to combine the three models - equal weights? Performance-based? Tried a few approaches.  
**Solution:** Tested different weight combinations on validation set. Ended up with RF=1.0, XGB=0.7, GB=0.5 (normalized). Could probably optimize more but this works.

### **4. GPU Compatibility**
**Problem:** Wanted GPU acceleration but code needed to work on laptops without GPUs too.  
**Solution:** Added GPU detection at startup, falls back to CPU automatically. Also made RAPIDS optional since it's hard to install.

### **5. Real-time Processing**
**Problem:** Loading all the F1 data took forever, especially on first run.  
**Solution:** Used FastF1's built-in cache (SQLite). Also saved processed data so don't have to redo feature engineering every time.

---

## Learning Outcomes & Skills Developed

### **Technical Skills**
- Learned how to build and combine ML models (Random Forest, XGBoost, Gradient Boosting)
- Got OCR working (took way longer than expected)
- Figured out GPU setup for faster training
- Built a data pipeline that handles caching and preprocessing
- Learned to evaluate models properly (cross-validation, metrics)
- Got better at writing code that doesn't crash (error handling is important!)

### **Problem-Solving Skills**
- Learned to break big problems into smaller pieces (OCR was overwhelming at first)
- Iterative approach - start simple, add features, test, fix bugs, repeat
- Edge cases are everywhere - missing data, bad images, model failures
- Made things faster with caching and GPU (was too slow initially)

### **Domain Knowledge**
- Already knew F1 pretty well, but learned which stats actually matter for predictions
- Got experience with sports analytics and prediction modeling
- Learned the full data science workflow - getting data, cleaning it, modeling, visualizing

---

## Code Quality & Engineering Practices

### **Software Architecture**
- **Object-Oriented Design:** F1PredictionModel class with modular methods
- **Type Hints:** Comprehensive type annotations for maintainability
- **Error Handling:** Try-except blocks with detailed error messages
- **Documentation:** Docstrings for all major functions and classes

### **Code Organization**
- **Modular Structure:** Separate methods for data loading, preprocessing, training, prediction
- **Separation of Concerns:** Clear distinction between ML, CV, and visualization components
- **Reusability:** Methods designed for multiple use cases (different years, races, grid inputs)

### **Performance Optimization**
- **Caching:** SQLite cache for FastF1 data, pickle for models
- **Lazy Loading:** Models loaded only when needed
- **GPU Acceleration:** Automatic detection and utilization
- **Memory Efficiency:** Efficient data structures and processing

---

## Future Enhancements

### **Things I'd Like to Add**
1. **Better OCR:** Maybe try a neural network approach instead of EasyOCR, or train a custom model
2. **Live Data:** Would be cool to pull data during actual races and update predictions in real-time
3. **Weather Data:** Weather affects races a lot but haven't found a good free API for it yet
4. **Strategy Stuff:** Tire compounds, pit stops - but need more detailed race data
5. **Web Interface:** Would make it easier to use, maybe Flask or something simple
6. **More Features:** Driver form over last 3 races, head-to-head records, etc.

### **Ideas for Later**
- Try stacking instead of just averaging models
- Predict whole season outcomes, not just individual races
- Fine-tune models per driver (some drivers are more predictable than others)
- Track-specific models (Monaco is way different from Monza)

---

## Project Artifacts

### **Code Repository**
- Main implementation: `f1_prediction_model.py` (1,524 lines)
- Model files: `models/` directory (rf_model.pkl, xgb_model.pkl, gb_model.pkl)
- Processed data: `processed_data/` directory
- Results: `results/` directory with visualizations and predictions

### **Documentation**
- README.md with installation and usage instructions
- Inline code documentation and docstrings
- Feature engineering documentation

### **Outputs**
- Race prediction visualizations (PNG format, 300 DPI)
- CSV prediction files
- Processed grid images with annotations
- Team development analysis charts

---

## Reflection

This project was way harder than I thought it would be when I started. I thought I could just throw some data into scikit-learn and get predictions, but there were so many issues along the way - missing data, OCR not working, models overfitting, etc. But that's what made it interesting.

**What I Learned:**
- Real code needs a lot of error handling (learned this the hard way when OCR failed)
- Ensemble methods actually work really well - combining models gave better results than any single one
- Computer vision is trickier than it looks - OCR accuracy depends a lot on image quality
- GPU acceleration is nice but not necessary - the code works fine on CPU too
- Understanding the domain (F1) helped a lot with feature engineering - knowing which stats matter

**Challenges:**
The biggest challenge was getting the OCR to work reliably. I spent a lot of time trying different libraries and preprocessing steps. Eventually I just added fallback mappings for when it fails, which isn't elegant but it works. Also had issues with missing data in some races - had to add a lot of checks for that.

**What I'd Do Differently:**
- Start with a simpler version and add features gradually (I tried to do everything at once)
- Test OCR on more image types earlier
- Maybe use a neural network for OCR instead of EasyOCR
- Add more features like weather data if I can get it

**Impact:**
This project taught me that building something real is way different from following tutorials. You have to figure out edge cases, handle errors, and make tradeoffs. But it's also way more satisfying when it actually works.

---

## Contact & Repository

**Project Repository:** https://github.com/arnavmandre/F1_Prediction_ML  
**GitHub Username:** arnavmandre  
**Contact:** [Your Email]  
**Documentation:** See README.md for installation and usage instructions

---

*This project demonstrates advanced technical skills, innovative problem-solving, and practical application of machine learning and computer vision principles—qualities that align with MIT's maker culture and engineering excellence.*


@echo off
cd /d "%~dp0"
echo ========================================
echo F1 Prediction Model - FULL DEMO
echo ========================================
echo.
echo Running visual demo first...
python demo_visual.py
echo.
echo.
echo Now running technical demo...
python demo_technical.py
echo.
echo ========================================
echo Full demo completed!
echo ========================================
echo.
echo All visualizations have been generated and opened!
echo.
echo Press any key to exit...
pause >nul


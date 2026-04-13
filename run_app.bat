@echo off
echo =====================================
echo Starting Log Classification System...
echo =====================================

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies (safe if already installed)
echo Installing dependencies...
python -m pip install -r requirements.txt

REM Check if model exists
IF NOT EXIST model\model.pkl (
    echo Model not found. Training model first...
    python src/train.py
) ELSE (
    echo Model already exists. Skipping training.
)

REM Start Flask app
echo Launching Web App...
echo Open browser at: http://127.0.0.1:5000/
python app.py

pause
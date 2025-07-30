# TrafficTelligence: AI-Powered Traffic Analysis

Advanced traffic volume estimation system using machine learning and computer vision.

## Features

- **Real-time Vehicle Detection**: YOLOv8-powered vehicle detection from video feeds
- **Traffic Volume Prediction**: ML-based forecasting using Random Forest
- **Video Analysis**: Upload and analyze traffic videos automatically  
- **Smart Dashboard**: Real-time analytics and predictions
- **REST API**: Easy integration with other systems

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open http://localhost:5000 in your browser

## API Endpoints

- `GET /analyze` - Current traffic analysis
- `GET /predict` - Next 24 hours predictions  
- `POST /upload_video` - Analyze traffic video
- `GET /plot` - Traffic volume visualization

## Technology Stack

- **Backend**: Flask, Python
- **ML/AI**: YOLOv8, Random Forest, scikit-learn
- **Computer Vision**: OpenCV, Ultralytics
- **Data**: Pandas, NumPy, Matplotlib

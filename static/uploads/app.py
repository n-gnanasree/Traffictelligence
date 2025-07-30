from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from traffic_detector import TrafficDetector
from ml_predictor import TrafficPredictor
from traffic_insights import TrafficInsights

app = Flask(__name__)

# Initialize components
df = pd.read_csv("traffic volume.csv")
detector = TrafficDetector()
predictor = TrafficPredictor()
insights = TrafficInsights("traffic volume.csv")
training_result = predictor.train("traffic volume.csv")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    average_volume = df['traffic_volume'].mean()
    peak_hour = df['Time'].str[:2].value_counts().idxmax()
    current_hour = pd.Timestamp.now().hour
    
    # Weather-aware prediction
    predicted_volume = predictor.predict_volume(
        current_hour, 
        pd.Timestamp.now().weekday(), 
        pd.Timestamp.now().month,
        weather='Clear',  # Default weather
        temp=280  # Default temp
    )
    
    # Traffic insights
    weather_impact = df.groupby('weather')['traffic_volume'].mean().to_dict()
    
    return jsonify({
        'average_volume': round(average_volume, 2),
        'peak_hour': peak_hour + ':00',
        'current_prediction': predicted_volume,
        'current_hour': current_hour,
        'weather_impact': {k: round(v, 0) for k, v in weather_impact.items() if pd.notna(k)}
    })

@app.route('/predict')
def predict():
    predictions = predictor.predict_next_hours(12)
    return jsonify(predictions)

@app.route('/insights')
def get_insights():
    congestion = insights.get_congestion_analysis()
    weather_impact = insights.weather_impact_analysis()
    optimal_times = insights.predict_optimal_times()
    
    return jsonify({
        'congestion_analysis': congestion,
        'weather_impact': weather_impact,
        'optimal_times': optimal_times,
        'model_accuracy': round(training_result.get('accuracy', 0) * 100, 1)
    })

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    filepath = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)
    
    try:
        result = detector.count_traffic(filepath)
        os.remove(filepath)  # Clean up
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/plot')
def plot():
    df['hour'] = df['Time'].str[:2]
    hourly_avg = df.groupby('hour')['traffic_volume'].mean()

    plt.figure(figsize=(12, 8))
    
    # Create subplots
    plt.subplot(2, 2, 1)
    hourly_avg.plot(kind='bar', color='skyblue')
    plt.xlabel('Hour')
    plt.ylabel('Avg Traffic Volume')
    plt.title('Hourly Traffic Pattern')
    plt.xticks(rotation=45)
    
    # Weather impact
    plt.subplot(2, 2, 2)
    weather_avg = df.groupby('weather')['traffic_volume'].mean().dropna()
    weather_avg.plot(kind='bar', color='lightcoral')
    plt.xlabel('Weather')
    plt.ylabel('Avg Traffic Volume')
    plt.title('Weather Impact on Traffic')
    plt.xticks(rotation=45)
    
    # Temperature vs Traffic
    plt.subplot(2, 2, 3)
    temp_clean = df.dropna(subset=['temp'])
    plt.scatter(temp_clean['temp'], temp_clean['traffic_volume'], alpha=0.5, color='green')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Traffic Volume')
    plt.title('Temperature vs Traffic')
    
    # Day of week pattern
    plt.subplot(2, 2, 4)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['Time'])
    df['day_name'] = df['datetime'].dt.day_name()
    day_avg = df.groupby('day_name')['traffic_volume'].mean()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_avg = day_avg.reindex([d for d in day_order if d in day_avg.index])
    day_avg.plot(kind='bar', color='orange')
    plt.xlabel('Day of Week')
    plt.ylabel('Avg Traffic Volume')
    plt.title('Weekly Traffic Pattern')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plot_path = 'static/traffic_plot.png'
    os.makedirs('static', exist_ok=True)
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    return send_file(plot_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import datetime

class TrafficPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df):
        # Combine date and time columns
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['Time'],dayfirst=True,errors='coerce')

        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_holiday'] = (df['holiday'] != 'None').astype(int)
        
        # Weather encoding
        weather_map = {'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Drizzle': 3, 'Mist': 4, 'Fog': 5, 'Snow': 6, 'Thunderstorm': 7, 'Haze': 8}
        df['weather_encoded'] = df['weather'].map(weather_map).fillna(1)
        
        # Temperature (fill missing values with mean)
        df['temp_filled'] = df['temp'].fillna(df['temp'].mean())
        
        features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday', 'weather_encoded', 'temp_filled']
        return df[features]
    
    def train(self, csv_path):
        df = pd.read_csv(csv_path)
        X = self.prepare_features(df)
        y = df['traffic_volume']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        score = self.model.score(X_test_scaled, y_test)
        return {'accuracy': score, 'test_samples': len(X_test)}
    
    def predict_volume(self, hour, day_of_week, month, is_weekend=None, is_holiday=0, weather='Clear', temp=280):
        if not self.is_trained:
            return None
            
        if is_weekend is None:
            is_weekend = 1 if day_of_week >= 5 else 0
            
        weather_map = {'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Drizzle': 3, 'Mist': 4, 'Fog': 5, 'Snow': 6, 'Thunderstorm': 7, 'Haze': 8}
        weather_encoded = weather_map.get(weather, 1)
        
        features = np.array([[hour, day_of_week, month, is_weekend, is_holiday, weather_encoded, temp]])
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        return max(0, int(prediction))
    
    def predict_next_hours(self, hours=24):
        now = datetime.datetime.now()
        predictions = []
        
        for i in range(hours):
            future_time = now + datetime.timedelta(hours=i)
            pred = self.predict_volume(
                future_time.hour,
                future_time.weekday(),
                future_time.month
            )
            predictions.append({
                'hour': future_time.strftime('%H:00'),
                'predicted_volume': pred
            })
        
        return predictions

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TrafficInsights:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['datetime'] = pd.to_datetime(
    self.df['date'] + ' ' + self.df['Time'],
    dayfirst=True,
    errors='coerce'
)

        
    def get_congestion_analysis(self):
        """Analyze traffic congestion patterns"""
        hourly_avg = self.df.groupby(self.df['Time'].str[:2])['traffic_volume'].mean()
        
        # Define congestion levels
        low_threshold = hourly_avg.quantile(0.33)
        high_threshold = hourly_avg.quantile(0.67)
        
        congestion_hours = {
            'low': hourly_avg[hourly_avg <= low_threshold].index.tolist(),
            'medium': hourly_avg[(hourly_avg > low_threshold) & (hourly_avg <= high_threshold)].index.tolist(),
            'high': hourly_avg[hourly_avg > high_threshold].index.tolist()
        }
        
        return {
            'peak_hours': hourly_avg.nlargest(3).to_dict(),
            'off_peak_hours': hourly_avg.nsmallest(3).to_dict(),
            'congestion_levels': congestion_hours
        }
    
    def weather_impact_analysis(self):
        """Analyze how weather affects traffic"""
        weather_stats = self.df.groupby('weather')['traffic_volume'].agg(['mean', 'std', 'count']).round(2)
        
        # Calculate weather impact relative to clear weather
        clear_avg = weather_stats.loc['Clear', 'mean'] if 'Clear' in weather_stats.index else weather_stats['mean'].mean()
        weather_stats['impact_factor'] = (weather_stats['mean'] / clear_avg).round(2)
        
        return weather_stats.to_dict('index')
    
    def predict_optimal_times(self, target_date=None):
        """Suggest optimal travel times"""
        if target_date is None:
            target_date = datetime.now().date()
        
        day_of_week = pd.to_datetime(target_date).weekday()
        
        # Get hourly averages for this day type
        self.df['day_of_week'] = self.df['datetime'].dt.weekday
        day_data = self.df[self.df['day_of_week'] == day_of_week]
        
        if len(day_data) > 0:
            hourly_avg = day_data.groupby(day_data['Time'].str[:2])['traffic_volume'].mean()
        else:
            hourly_avg = self.df.groupby(self.df['Time'].str[:2])['traffic_volume'].mean()
        
        # Find best and worst times
        best_times = hourly_avg.nsmallest(5).index.tolist()
        worst_times = hourly_avg.nlargest(5).index.tolist()
        
        return {
            'best_travel_times': [f"{hour}:00-{int(hour)+1}:00" for hour in best_times],
            'avoid_times': [f"{hour}:00-{int(hour)+1}:00" for hour in worst_times],
            'day_type': 'Weekend' if day_of_week >= 5 else 'Weekday'
        }

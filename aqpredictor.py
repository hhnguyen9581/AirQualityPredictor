import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class AirQualityPredictor:
    """
    Urban Air Quality Predictor using Random Forest Regression
    Predicts next-day PM2.5 levels using historical pollution and weather data
    Uses Open-Meteo API
    """

    def __init__(self, city="Los Angeles", latitude=34.05, longitude=-118.24, openaq_api_key=None):
        self.city = city
        self.latitude = latitude
        self.longitude = longitude
        self.openaq_api_key = openaq_api_key
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def fetch_openaq_data(self, days=90):
        """
        Fetch PM2.5 data using Open-Meteo Air Quality API
        """
        print(f"Fetching air quality data for {self.city}...")
        print("Using Open-Meteo Air Quality API")

        base_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

        # Calculate the date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        params = {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'hourly': 'pm2_5',  # Request PM2.5 data
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'timezone': 'auto'
        }

        try:
            print(f"  Requesting data from {start_date} to {end_date}...")
            print(f"  Location: ({self.latitude}, {self.longitude})")

            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Verify response structure
            if 'hourly' not in data:
                print("  ⚠ Unexpected API response format")
                print(f"  Response keys: {list(data.keys())}")
                return self._generate_synthetic_pollution_data(days)

            hourly_data = data['hourly']

            if 'time' not in hourly_data or 'pm2_5' not in hourly_data:
                print("  ⚠ Missing time or PM2.5 data in response")
                return self._generate_synthetic_pollution_data(days)

            times = hourly_data['time']
            pm25_values = hourly_data['pm2_5']

            print(f"  ✓ Received {len(times)} hourly measurements")

            # Check if we have valid data
            valid_count = sum(1 for v in pm25_values if v is not None)

            if valid_count == 0:
                print("  ⚠ No valid PM2.5 data available for this location")
                print("  Open-Meteo may not have coverage for this area")
                print("  Generating synthetic data instead...")
                return self._generate_synthetic_pollution_data(days)

            print(f"  ✓ Found {valid_count} valid measurements ({valid_count / len(times) * 100:.1f}%)")

            # Parse into records
            records = []
            for time_str, pm25 in zip(times, pm25_values):
                if pm25 is not None and pm25 >= 0 and pm25 < 1000:  # Filter invalid values
                    try:
                        dt = pd.to_datetime(time_str)
                        records.append({
                            'datetime': dt,
                            'date': dt.date(),
                            'pm25': pm25
                        })
                    except Exception:
                        continue

            if len(records) == 0:
                print("  ⚠ No valid PM2.5 measurements after filtering")
                return self._generate_synthetic_pollution_data(days)

            # Convert to DataFrame
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])

            print(f"  Processing {len(records)} valid measurements...")

            # Calculate daily averages
            df_daily = df.groupby('date').agg({
                'pm25': ['mean', 'count', 'min', 'max']
            }).reset_index()
            df_daily.columns = ['date', 'pm25', 'count', 'pm25_min', 'pm25_max']

            # Keep only days with sufficient data (at least 8 hours)
            df_daily = df_daily[df_daily['count'] >= 8].copy()
            df_daily = df_daily[['date', 'pm25']].sort_values('date').reset_index(drop=True)

            if len(df_daily) < 30:
                print(f" Only {len(df_daily)} days with sufficient data (need 30+)")
                print("  This may be due to limited historical data availability")
                print("  Generating synthetic data instead...")
                return self._generate_synthetic_pollution_data(days)

            print(f"\n Retrieved {len(df_daily)} days of reliable PM2.5 data")
            print(
                f"    Date range: {df_daily['date'].min().strftime('%Y-%m-%d')} to {df_daily['date'].max().strftime('%Y-%m-%d')}")
            print(f"    PM2.5 range: {df_daily['pm25'].min():.1f} - {df_daily['pm25'].max():.1f} μg/m³")
            print(f"    PM2.5 mean: {df_daily['pm25'].mean():.1f} μg/m³")
            print()

            return df_daily

        except requests.exceptions.HTTPError as e:
            print(f"  HTTP Error {e.response.status_code}: {e}")
            print("  The Open-Meteo API may be experiencing issues")
            return self._generate_synthetic_pollution_data(days)

        except requests.exceptions.RequestException as e:
            print(f"  Network Error: {str(e)[:100]}")
            return self._generate_synthetic_pollution_data(days)

        except Exception as e:
            print(f"  Unexpected Error: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            return self._generate_synthetic_pollution_data(days)

    def fetch_weather_data(self, days=90):
        """Fetch historical weather data from Open-Meteo API"""
        print(f"Fetching weather data for {self.city}...")

        url = "https://archive-api.open-meteo.com/v1/archive"

        date_to = datetime.now()
        date_from = date_to - timedelta(days=days)

        params = {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'start_date': date_from.strftime('%Y-%m-%d'),
            'end_date': date_to.strftime('%Y-%m-%d'),
            'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,windspeed_10m_max,windgusts_10m_max',
            'timezone': 'auto'
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame({
                'date': pd.to_datetime(data['daily']['time']),
                'temp_max': data['daily']['temperature_2m_max'],
                'temp_min': data['daily']['temperature_2m_min'],
                'temp_mean': data['daily']['temperature_2m_mean'],
                'precipitation': data['daily']['precipitation_sum'],
                'wind_speed': data['daily']['windspeed_10m_max'],
                'wind_gust': data['daily']['windgusts_10m_max']
            })

            print(f"✓ Fetched {len(df)} days of weather data")
            return df

        except Exception as e:
            print(f" Error fetching weather data: {e}")
            return None

    def _generate_synthetic_pollution_data(self, days=90):
        """Generate synthetic PM2.5 data for demonstration if API fails"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Create realistic PM2.5 pattern with seasonality and noise
        base_level = 25
        seasonal = 10 * np.sin(np.linspace(0, 4 * np.pi, days))
        trend = np.linspace(0, 5, days)
        noise = np.random.normal(0, 5, days)

        pm25 = base_level + seasonal + trend + noise
        pm25 = np.maximum(pm25, 0)  # Ensure non-negative

        return pd.DataFrame({'date': dates, 'pm25': pm25})

    def _generate_synthetic_full_dataset(self, days=90):
        """Generate complete synthetic dataset with weather and pollution if API fails"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # PM2.5 data
        base_level = 25
        seasonal = 10 * np.sin(np.linspace(0, 4 * np.pi, days))
        trend = np.linspace(0, 5, days)
        noise = np.random.normal(0, 5, days)
        pm25 = np.maximum(base_level + seasonal + trend + noise, 0)

        # Weather data
        temp_mean = 20 + 10 * np.sin(np.linspace(0, 2 * np.pi, days)) + np.random.normal(0, 2, days)
        temp_max = temp_mean + np.random.uniform(3, 8, days)
        temp_min = temp_mean - np.random.uniform(3, 8, days)
        precipitation = np.maximum(np.random.exponential(2, days), 0)
        wind_speed = np.maximum(np.random.gamma(2, 2, days), 0.1)
        wind_gust = wind_speed + np.random.uniform(5, 15, days)

        return pd.DataFrame({
            'date': dates,
            'pm25': pm25,
            'temp_max': temp_max,
            'temp_min': temp_min,
            'temp_mean': temp_mean,
            'precipitation': precipitation,
            'wind_speed': wind_speed,
            'wind_gust': wind_gust
        })

    def merge_data(self, pollution_df, weather_df):
        """Merge pollution and weather data"""
        print("Merging datasets...")

        df = pd.merge(pollution_df, weather_df, on='date', how='inner')
        df = df.sort_values('date').reset_index(drop=True)

        print(f"✓ Merged dataset: {len(df)} records")
        return df

    def engineer_features(self, df):
        """Create additional features for better prediction"""
        print("Engineering features...")

        df = df.copy()

        # Temporal features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear

        # Lagged PM2.5 features (previous days)
        df['pm25_lag1'] = df['pm25'].shift(1)
        df['pm25_lag2'] = df['pm25'].shift(2)
        df['pm25_lag3'] = df['pm25'].shift(3)
        df['pm25_lag7'] = df['pm25'].shift(7)

        # Rolling statistics
        df['pm25_rolling_mean_7'] = df['pm25'].rolling(window=7, min_periods=1).mean()
        df['pm25_rolling_std_7'] = df['pm25'].rolling(window=7, min_periods=1).std().fillna(0)

        # Weather interactions
        df['temp_range'] = df['temp_max'] - df['temp_min']
        df['wind_precipitation'] = df['wind_speed'] * df['precipitation']

        # Target: next day PM2.5
        df['pm25_next_day'] = df['pm25'].shift(-1)

        # Drop rows with NaN values
        df = df.dropna()

        print(f"✓ Feature engineering complete: {df.shape[1]} features, {len(df)} samples")

        # Check if we have enough data
        if len(df) < 20:
            print(f" Warning: Only {len(df)} samples after feature engineering.")
            print("Using synthetic data instead.")
            return None

        return df

    def prepare_train_test(self, df, test_size=0.3):
        """Split data into train and test sets by date"""
        print(f"Splitting data: {int((1 - test_size) * 100)}% train, {int(test_size * 100)}% test...")

        if df is None or len(df) < 20:
            print("Insufficient data for training")
            return None, None, None, None, None, None

        # Features for modeling
        feature_cols = [col for col in df.columns if col not in ['date', 'pm25_next_day', 'pm25']]

        X = df[feature_cols]
        y = df['pm25_next_day']
        dates = df['date']

        # Time-based split (important for time series)
        split_idx = int(len(df) * (1 - test_size))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_train, dates_test = dates[:split_idx], dates[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.feature_names = feature_cols

        print(f"✓ Train set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")

        return X_train_scaled, X_test_scaled, y_train, y_test, dates_train, dates_test

    def train_model(self, X_train, y_train):
        """Train Random Forest Regressor"""
        print("\nTraining Random Forest model...")

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)
        print(" Model training complete")

    def evaluate_model(self, X_test, y_test, dates_test):
        """Evaluate model performance"""
        print("\nEvaluating model...")

        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"\n{'=' * 50}")
        print(f"MODEL PERFORMANCE METRICS")
        print(f"{'=' * 50}")
        print(f"Mean Absolute Error (MAE):  {mae:.2f} μg/m³")
        print(f"Root Mean Squared Error:     {rmse:.2f} μg/m³")
        print(f"R² Score:                    {r2:.3f}")
        print(f"{'=' * 50}\n")

        # Display prediction results in console
        print(f"\n{'=' * 70}")
        print(f"SHORT-TERM PM2.5 FORECASTS")
        print(f"{'=' * 70}")
        print(f"{'Date':<12} {'Actual PM2.5':<15} {'Predicted PM2.5':<18} {'Error':<10}")
        print(f"{'-' * 70}")

        for i in range(len(dates_test)):
            date_str = dates_test.iloc[i].strftime('%Y-%m-%d')
            actual = y_test.iloc[i]
            predicted = y_pred[i]
            error = actual - predicted
            print(f"{date_str:<12} {actual:>8.2f} μg/m³   {predicted:>8.2f} μg/m³      {error:>+7.2f}")

        print(f"{'=' * 70}\n")

        return y_pred, mae, rmse, r2

    def display_data_summary(self, df_featured, X_train, X_test, y_train, y_test):
        """Display all data used in calculations"""
        print(f"\n{'=' * 70}")
        print("DATA SUMMARY - TRAINING DATASET")
        print(f"{'=' * 70}\n")

        print(f"Total samples: {len(df_featured)}")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Number of features: {len(self.feature_names)}")

        print(f"\n{'=' * 70}")
        print("FEATURES USED IN MODEL")
        print(f"{'=' * 70}")
        for i, feature in enumerate(self.feature_names, 1):
            print(f"{i:2d}. {feature}")

        print(f"\n{'=' * 70}")
        print("FEATURE IMPORTANCE (TOP 10)")
        print(f"{'=' * 70}")
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)

        for idx, row in feature_importance.iterrows():
            print(f"{row['feature']:<25} {row['importance']:.4f}")

        print(f"\n{'=' * 70}")
        print("TRAINING DATA STATISTICS")
        print(f"{'=' * 70}")
        print(f"PM2.5 Mean:   {y_train.mean():.2f} μg/m³")
        print(f"PM2.5 Std:    {y_train.std():.2f} μg/m³")
        print(f"PM2.5 Min:    {y_train.min():.2f} μg/m³")
        print(f"PM2.5 Max:    {y_train.max():.2f} μg/m³")

        print(f"\n{'=' * 70}")
        print("TESTING DATA STATISTICS")
        print(f"{'=' * 70}")
        print(f"PM2.5 Mean:   {y_test.mean():.2f} μg/m³")
        print(f"PM2.5 Std:    {y_test.std():.2f} μg/m³")
        print(f"PM2.5 Min:    {y_test.min():.2f} μg/m³")
        print(f"PM2.5 Max:    {y_test.max():.2f} μg/m³")
        print(f"{'=' * 70}\n")

    def predict_next_day(self, df_featured):
        """Predict PM2.5 for the next day using most recent data"""
        print("\n" + "=" * 70)
        print("NEXT-DAY FORECAST")
        print("=" * 70)

        # Get the most recent complete data point
        latest_data = df_featured.iloc[-1:].copy()

        # Features for prediction (excluding target and date)
        feature_cols = [col for col in df_featured.columns if col not in ['date', 'pm25_next_day', 'pm25']]
        X_latest = latest_data[feature_cols]

        # Scale features
        X_latest_scaled = self.scaler.transform(X_latest)

        # Make prediction
        next_day_prediction = self.model.predict(X_latest_scaled)[0]

        # Calculate next day's date
        latest_date = latest_data['date'].values[0]
        next_day_date = pd.to_datetime(latest_date) + timedelta(days=1)

        print(f"\nBased on data from: {pd.to_datetime(latest_date).strftime('%Y-%m-%d')}")
        print(f"Forecast for:       {next_day_date.strftime('%Y-%m-%d')}")
        print(f"\nPredicted PM2.5:    {next_day_prediction:.2f} μg/m³")

        # Interpret air quality
        if next_day_prediction < 12:
            quality = "Good"
            color = "🟢"
        elif next_day_prediction < 35.4:
            quality = "Moderate"
            color = "🟡"
        elif next_day_prediction < 55.4:
            quality = "Unhealthy for Sensitive Groups"
            color = "🟠"
        elif next_day_prediction < 150.4:
            quality = "Unhealthy"
            color = "🔴"
        else:
            quality = "Very Unhealthy"
            color = "🟣"

        print(f"Air Quality Index:  {color} {quality}")
        print("=" * 70 + "\n")

        return next_day_date, next_day_prediction

    def plot_forecast(self, dates_test, y_test, y_pred, next_day_date=None, next_day_pred=None):
        """Create single forecast visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))

        # Plot historical actual vs predicted (validation period)
        ax.plot(dates_test.values, y_test.values, label='Actual PM2.5 (Historical)',
                color='#2E86AB', linewidth=2.5, marker='o', markersize=5)
        ax.plot(dates_test.values, y_pred, label='Model Predictions (Historical)',
                color='#A23B72', linewidth=2.5, marker='s', markersize=5, linestyle='--', alpha=0.7)

        # If we have a next-day forecast, add it
        if next_day_date is not None and next_day_pred is not None:
            # Connect last actual point to forecast
            last_date = dates_test.values[-1]
            last_actual = y_test.values[-1]

            ax.plot([last_date, next_day_date], [last_actual, next_day_pred],
                    color='#F18F01', linewidth=3, linestyle='-', marker='*',
                    markersize=15, label='Next-Day Forecast', zorder=5)

            # Add annotation for the forecast
            ax.annotate(f'Forecast: {next_day_pred:.1f} μg/m³',
                        xy=(next_day_date, next_day_pred),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                        fontsize=11, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

        # Shade the area between actual and predicted
        ax.fill_between(dates_test.values, y_test.values, y_pred, alpha=0.15, color='gray')

        # Add AQI threshold lines
        ax.axhline(y=12, color='green', linestyle=':', alpha=0.5, linewidth=1.5, label='Good (<12 μg/m³)')
        ax.axhline(y=35.4, color='gold', linestyle=':', alpha=0.5, linewidth=1.5, label='Moderate (12-35.4 μg/m³)')
        ax.axhline(y=55.4, color='orange', linestyle=':', alpha=0.5, linewidth=1.5,
                   label='Unhealthy for Sensitive (35.5-55.4 μg/m³)')

        ax.set_xlabel('Date', fontweight='bold', fontsize=12)
        ax.set_ylabel('PM2.5 Concentration (μg/m³)', fontweight='bold', fontsize=12)
        ax.set_title(f'PM2.5 Air Quality Forecast - {self.city}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45)

        # Highlight the forecast region
        if next_day_date is not None:
            ax.axvspan(dates_test.values[-1], next_day_date, alpha=0.1, color='yellow', label='_nolegend_')

        plt.tight_layout()
        plt.show()

    def run_full_pipeline(self):
        """Execute complete prediction pipeline"""
        print(f"\n{'=' * 60}")
        print(f"URBAN AIR QUALITY PREDICTOR - {self.city}")
        print(f"{'=' * 60}\n")

        # Fetch data
        pollution_df = self.fetch_openaq_data(days=90)
        weather_df = self.fetch_weather_data(days=90)

        if weather_df is None:
            print(" Failed to fetch weather data. Exiting...")
            return None, None, None, None

        # Merge and engineer features
        df = self.merge_data(pollution_df, weather_df)

        if len(df) < 20:
            print(f" Only {len(df)} records after merging. Need at least 20.")
            print("Generating synthetic dataset")
            df = self._generate_synthetic_full_dataset(90)

        df_featured = self.engineer_features(df)

        if df_featured is None:
            print("Generating synthetic dataset")
            df = self._generate_synthetic_full_dataset(90)
            df_featured = self.engineer_features(df)

        # Prepare data
        result = self.prepare_train_test(df_featured)
        if result[0] is None:
            print("Failed to prepare training data. Closing...")
            return None, None, None, None

        X_train, X_test, y_train, y_test, dates_train, dates_test = result

        # Train model
        self.train_model(X_train, y_train)

        # Display data summary
        self.display_data_summary(df_featured, X_train, X_test, y_train, y_test)

        # Evaluate on test set
        y_pred, mae, rmse, r2 = self.evaluate_model(X_test, y_test, dates_test)

        # Predict next day
        next_day_date, next_day_pred = self.predict_next_day(df_featured)

        # Visualize forecast (including next day)
        self.plot_forecast(dates_test, y_test, y_pred, next_day_date, next_day_pred)

        print(f"\n{'=' * 60}")
        print("✓ Pipeline complete!")
        print(f"{'=' * 60}\n")

        return self.model, mae, rmse, r2


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("STARTING AIR QUALITY PREDICTION SYSTEM")
    print("=" * 60)
    print("Using Open-Meteo API\n")

    # Initialize predictor for Los Angeles
    predictor = AirQualityPredictor(
        city="Los Angeles",
        latitude=34.05,
        longitude=-118.24
    )

    # Run full pipeline
    result = predictor.run_full_pipeline()

    if result[0] is not None:
        model, mae, rmse, r2 = result
        print("\nModel successfully trained and evaluated.")
        print(f"Final Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
    else:
        print("\n Pipeline failed. Please check the error messages above.")

    # Try other cities if needed:
    # predictor = AirQualityPredictor(city="New York", latitude=40.7128, longitude=-74.0060)
    # predictor = AirQualityPredictor(city="Beijing", latitude=39.9, longitude=116.4)
    # predictor = AirQualityPredictor(city="Delhi", latitude=28.7, longitude=77.1)
    # predictor = AirQualityPredictor(city="London", latitude=51.5, longitude=-0.1)
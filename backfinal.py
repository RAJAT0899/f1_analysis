from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
from typing import List, Dict
import requests
import logging
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from statistical_analysis import (
    perform_anova, perform_correlation_analysis, perform_regression_analysis, 
    calculate_effect_size, perform_pca, perform_time_series_analysis, 
    perform_cluster_analysis, perform_hypothesis_test
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Update CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "F1 Data Analysis API is running"}

@app.get("/race_results/{year}")
async def get_race_results(year: int) -> List[Dict]:
    url = f"http://ergast.com/api/f1/{year}/results.json?limit=1000"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch race data: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to fetch race data: {str(e)}")
    
    data = response.json()
    races = data['MRData']['RaceTable']['Races']
    
    results = []
    for race in races:
        for result in race['Results']:
            results.append({
                'season': race['season'],
                'round': race['round'],
                'raceName': race['raceName'],
                'circuitName': race['Circuit']['circuitName'],
                'driverId': result['Driver']['driverId'],
                'givenName': result['Driver']['givenName'],
                'familyName': result['Driver']['familyName'],
                'constructor': result['Constructor']['name'],
                'position': result['position'],
                'points': result['points']
            })
    
    return results

def generate_enhanced_telemetry(num_points=1000):
    # Base distance and time
    distance = np.linspace(0, 5000, num_points)  # 5km lap
    timestamp_ms = np.linspace(0, 90000, num_points)  # 90 second lap
    
    # Track and environment parameters
    track_length = 5000  # meters
    elevation_change = 50 * np.sin(distance / 1000) # Simulated elevation changes
    weather_conditions = np.random.choice(['Sunny', 'Cloudy', 'Light Rain'], num_points)
    temperature = 25 + np.random.normal(0, 2, num_points)  # Celsius
    track_grip = 90 + np.random.normal(0, 5, num_points)  # percentage
    
    # Vehicle dynamics
    speed = 200 + 50 * np.sin(distance / 500)
    speed = np.clip(speed, 50, 350)
    
    engine_rpm = 8000 + (speed * 30) + np.random.normal(0, 200, num_points)
    engine_rpm = np.clip(engine_rpm, 4000, 15000)
    
    # Forces and performance metrics
    braking_force = np.where(np.gradient(speed) < 0, 
                            -np.gradient(speed) * 1000, 
                            0)
    engine_temp = 90 + (speed / 10) + np.random.normal(0, 3, num_points)
    aero_drag = 0.5 * 1.225 * (speed ** 2) * 0.7  # Simple drag equation
    
    # Tire dynamics
    base_tire_wear = np.linspace(100, 90, num_points)  # Linear wear
    tire_temps = 80 + (speed / 5) + np.random.normal(0, 5, num_points)
    
    def generate_wheel_data(base_speed):
        return base_speed + np.random.normal(0, 0.5, num_points)
    
    # Add new constraints
    normalized_suspension_travel = {
        'front_left': np.random.uniform(0, 1, num_points),
        'front_right': np.random.uniform(0, 1, num_points),
        'rear_left': np.random.uniform(0, 1, num_points),
        'rear_right': np.random.uniform(0, 1, num_points)
    }

    suspension_travel_meters = {
        'front_left': np.random.uniform(0, 0.2, num_points),
        'front_right': np.random.uniform(0, 0.2, num_points),
        'rear_left': np.random.uniform(0, 0.2, num_points),
        'rear_right': np.random.uniform(0, 0.2, num_points)
    }

    position = {
        'x': np.cumsum(np.random.normal(0, 1, num_points)),
        'y': np.cumsum(np.random.normal(0, 1, num_points)),
        'z': elevation_change
    }

    acceleration = {
        'x': np.random.normal(0, 2, num_points),
        'y': np.random.normal(0, 2, num_points),
        'z': np.random.normal(0, 1, num_points)
    }

    velocity = {
        'x': np.cumsum(acceleration['x']) * 0.01,
        'y': np.cumsum(acceleration['y']) * 0.01,
        'z': np.cumsum(acceleration['z']) * 0.01
    }

    angular_velocity = {
        'x': np.random.normal(0, 0.1, num_points),
        'y': np.random.normal(0, 0.1, num_points),
        'z': np.random.normal(0, 0.2, num_points)
    }

    # Generate telemetry DataFrame
    telemetry = pd.DataFrame({
        'timestamp_ms': timestamp_ms,
        'Distance': distance,
        'current_engine_rpm': engine_rpm,
        'CarModel': 'F1-2024',
        'TrackLength_km': track_length / 1000,
        'ElevationChange_m': elevation_change,
        'Weather': weather_conditions,
        'Temperature_C': temperature,
        'TrackGrip': track_grip,
        'MaxSpeed_kmh': speed,
        'AvgSpeed_kmh': np.cumsum(speed) / np.arange(1, num_points + 1),
        'BrakingForce_N': braking_force,
        'EngineTemp_C': engine_temp,
        'EngineRPM': engine_rpm,
        'AeroDrag_N': aero_drag,
        'TireWear_percent': base_tire_wear,
        'GForce_lateral': np.random.normal(0, 2, num_points),
        'GForce_longitudinal': np.random.normal(0, 3, num_points),
        'FuelConsumption_lph': 100 * (engine_rpm / 15000) + np.random.normal(0, 2, num_points),
        'DownforceBalance_percent': 50 + np.random.normal(0, 3, num_points),
        'normalized_suspension_travel_front_left': normalized_suspension_travel['front_left'],
        'normalized_suspension_travel_front_right': normalized_suspension_travel['front_right'],
        'normalized_suspension_travel_rear_left': normalized_suspension_travel['rear_left'],
        'normalized_suspension_travel_rear_right': normalized_suspension_travel['rear_right'],
        'suspension_travel_meters_front_left': suspension_travel_meters['front_left'],
        'suspension_travel_meters_front_right': suspension_travel_meters['front_right'],
        'suspension_travel_meters_rear_left': suspension_travel_meters['rear_left'],
        'suspension_travel_meters_rear_right': suspension_travel_meters['rear_right'],
        'position_x': position['x'],
        'position_y': position['y'],
        'position_z': position['z'],
        'acceleration_x': acceleration['x'],
        'acceleration_y': acceleration['y'],
        'acceleration_z': acceleration['z'],
        'velocity_x': velocity['x'],
        'velocity_y': velocity['y'],
        'velocity_z': velocity['z'],
        'angular_velocity_x': angular_velocity['x'],
        'angular_velocity_y': angular_velocity['y'],
        'angular_velocity_z': angular_velocity['z'],
        'yaw': np.cumsum(angular_velocity['z']) * 0.01,
        'pitch': np.cumsum(angular_velocity['y']) * 0.01,
        'roll': np.cumsum(angular_velocity['x']) * 0.01,
        'speed': np.sqrt(velocity['x']**2 + velocity['y']**2 + velocity['z']**2),
        'power': engine_rpm * 0.1 + np.random.normal(0, 10, num_points),
        'torque': engine_rpm * 0.05 + np.random.normal(0, 5, num_points),
        'boost': np.random.uniform(0, 2, num_points),
        'fuel': 100 - np.linspace(0, 20, num_points) + np.random.normal(0, 1, num_points),
        'distance_traveled': np.cumsum(speed * 0.001),
        'acceleration': np.sqrt(acceleration['x']**2 + acceleration['y']**2 + acceleration['z']**2),
        'brake': np.random.uniform(0, 1, num_points),
        'clutch': np.random.choice([0, 1], num_points, p=[0.95, 0.05]),
        'handbrake': np.random.choice([0, 1], num_points, p=[0.99, 0.01]),
        'gear': np.random.randint(1, 9, num_points),
        'steer': np.random.uniform(-1, 1, num_points),
        'lap_number': np.random.randint(1, 60, num_points),
        'best_lap_time': np.random.uniform(80, 90, num_points),
        'last_lap_time': np.random.uniform(80, 95, num_points),
        'current_lap_time': np.linspace(0, 90, num_points) + np.random.normal(0, 1, num_points),
        'current_race_time': np.linspace(0, 5400, num_points),
        'race_position': np.random.randint(1, 21, num_points)
    })
    
    for wheel in ['front_left', 'front_right', 'rear_left', 'rear_right']:
        telemetry[f'wheel_rotation_speed_{wheel}'] = generate_wheel_data(speed)
        telemetry[f'wheel_on_rumble_strip_{wheel}'] = np.random.choice([0, 1], num_points, p=[0.9, 0.1])
        telemetry[f'wheel_in_puddle_depth_{wheel}'] = np.random.exponential(0.1, num_points)
        telemetry[f'tire_slip_rotation_{wheel}'] = np.random.normal(0, 0.1, num_points)
        telemetry[f'tire_slip_angle_{wheel}'] = np.random.normal(0, 2, num_points)
        telemetry[f'tire_combined_slip_{wheel}'] = np.random.normal(0, 0.15, num_points)
        telemetry[f'tire_temp_{wheel}'] = tire_temps + np.random.normal(0, 2, num_points)
    
    return telemetry

@app.get("/telemetry/{year}/{grand_prix}")
async def get_telemetry(year: int, grand_prix: str):
    try:
        telemetry = generate_enhanced_telemetry()
        return telemetry.to_dict(orient='list')
    except Exception as e:
        logger.error(f"Failed to generate telemetry data: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to generate telemetry data: {str(e)}")

@app.get("/download_excel/{year}/{grand_prix}")
async def download_excel(year: int, grand_prix: str):
    try:
        # Fetch race results
        race_results = await get_race_results(year)
        race_results_df = pd.DataFrame(race_results)
        
        # Generate telemetry data
        telemetry_df = generate_enhanced_telemetry()
        
        # Create a directory for temporary files if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"f1_data_{year}_{grand_prix.replace(' ', '_')}_{timestamp}.xlsx"
        filepath = os.path.join("temp", filename)
        
        # Write data to Excel with enhanced telemetry
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            race_results_df.to_excel(writer, sheet_name='Race Results', index=False)
            telemetry_df.to_excel(writer, sheet_name='Enhanced Telemetry', index=False)
            
            # Add summary statistics
            summary_stats = pd.DataFrame({
                'Metric': ['Average Speed', 'Max Speed', 'Average Engine Temp', 'Average Tire Wear',
                          'Max G-Force Lateral', 'Max G-Force Longitudinal', 'Average Fuel Consumption'],
                'Value': [
                    telemetry_df['AvgSpeed_kmh'].mean(),
                    telemetry_df['MaxSpeed_kmh'].max(),
                    telemetry_df['EngineTemp_C'].mean(),
                    telemetry_df['TireWear_percent'].mean(),
                    telemetry_df['GForce_lateral'].abs().max(),
                    telemetry_df['GForce_longitudinal'].abs().max(),
                    telemetry_df['FuelConsumption_lph'].mean()
                ]
            })
            summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)
        
        # Log file creation
        logger.info(f"Excel file created: {filepath}")
        
        # Serve the file
        headers = {
            'Content-Disposition': f'attachment; filename="{filename}"'
        }
        return FileResponse(filepath, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as e:
        logger.error(f"Failed to generate Excel file: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to generate Excel file: {str(e)}")

@app.get("/telemetry_statistics/{year}/{grand_prix}")
async def get_telemetry_statistics(year: int, grand_prix: str):
    try:
        telemetry = generate_enhanced_telemetry()
        
        # Calculate basic statistics for numerical columns
        numeric_cols = telemetry.select_dtypes(include=[np.number]).columns
        statistics = {
            'basic_stats': telemetry[numeric_cols].describe().to_dict(),
            'correlations': {
                'speed_vs_engine': float(telemetry['MaxSpeed_kmh'].corr(telemetry['current_engine_rpm'])),
                'speed_vs_temp': float(telemetry['MaxSpeed_kmh'].corr(telemetry['EngineTemp_C'])),
                'tire_wear_vs_grip': float(telemetry['TireWear_percent'].corr(telemetry['TrackGrip']))
            },
            'weather_distribution': telemetry['Weather'].value_counts().to_dict()
        }
        
        return statistics
    except Exception as e:
        logger.error(f"Failed to generate telemetry statistics: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to generate telemetry statistics: {str(e)}")

@app.get("/train_model/{year}/{grand_prix}")
async def train_model(year: int, grand_prix: str):
    try:
        telemetry = generate_enhanced_telemetry()
        
        # Prepare data for modeling
        features = ['current_engine_rpm', 'EngineTemp_C', 'TireWear_percent', 'TrackGrip']
        target = 'MaxSpeed_kmh'
        
        X = telemetry[features]
        y = telemetry[target]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save the model
        model_filename = f"f1_model_{year}_{grand_prix}.joblib"
        joblib.dump(model, model_filename)
        
        return {
            "message": "Model trained successfully",
            "mse": mse,
            "r2": r2,
            "model_filename": model_filename
        }
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to train model: {str(e)}")

@app.get("/advanced_analysis/{analysis_type}")
async def get_advanced_analysis(analysis_type: str, year: int, grand_prix: str):
    try:
        telemetry = generate_enhanced_telemetry()
        
        if analysis_type == "pca":
            variables = ['MaxSpeed_kmh', 'current_engine_rpm', 'EngineTemp_C', 'TireWear_percent']
            pca_result, explained_variance_ratio, loadings = perform_pca(telemetry, variables)
            return {
                "pca_result": pca_result.tolist(),
                "explained_variance_ratio": explained_variance_ratio.tolist(),
                "loadings": loadings.tolist(),
                "variables": variables
            }
        
        elif analysis_type == "time_series":
            time_column = 'timestamp_ms'
            value_column = 'MaxSpeed_kmh'
            ts_results = perform_time_series_analysis(telemetry, time_column, value_column)
            return {
                "rolling_mean": ts_results['rolling_mean'].tolist(),
                "rolling_std": ts_results['rolling_std'].tolist(),
                "autocorrelation": ts_results['autocorrelation'].tolist(),
                "time": telemetry[time_column].tolist(),
                "value": telemetry[value_column].tolist()
            }
        
        elif analysis_type == "cluster":
            variables = ['MaxSpeed_kmh', 'current_engine_rpm', 'EngineTemp_C', 'TireWear_percent']
            cluster_labels, silhouette_avg = perform_cluster_analysis(telemetry, variables)
            return {
                "cluster_labels": cluster_labels.tolist(),
                "silhouette_score": silhouette_avg,
                "data": telemetry[variables].to_dict(orient='list')
            }
        
        elif analysis_type == "hypothesis_test":
            variable = 'MaxSpeed_kmh'
            mid_point = len(telemetry) // 2
            group1 = telemetry[variable][:mid_point]
            group2 = telemetry[variable][mid_point:]
            test_results = perform_hypothesis_test(group1, group2)
            return {
                "test_statistic": float(test_results['test_statistic']),
                "p_value": float(test_results['p_value']),
                "group1": group1.tolist(),
                "group2": group2.tolist()
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported analysis type: {analysis_type}")
    
    except Exception as e:
        logger.error(f"Failed to perform advanced analysis: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to perform advanced analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting F1 Data Analysis API")
    uvicorn.run(app, host="0.0.0.0", port=8001)
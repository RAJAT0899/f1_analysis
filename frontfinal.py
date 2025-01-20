import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statistical_analysis import (
    perform_anova, perform_correlation_analysis, perform_regression_analysis, 
    calculate_effect_size, perform_pca, perform_time_series_analysis, 
    perform_cluster_analysis, perform_hypothesis_test
)
import base64

# Page configuration
st.set_page_config(page_title="F1 Research Analysis", layout="wide")

st.title("Formula 1 Research Analysis Platform")

# Sidebar for comprehensive filtering
st.sidebar.header("Research Parameters")
year_range = st.sidebar.slider("Select Year Range", 2013, 2023, (2018, 2023))

# Backend URL
BACKEND_URL = "http://localhost:8001"

# Check backend connection
@st.cache_data(ttl=60)
def check_backend_connection():
    try:
        response = requests.get(f"{BACKEND_URL}/")
        return response.status_code == 200
    except requests.RequestException:
        return False

if not check_backend_connection():
    st.error("Unable to connect to the backend. Please make sure the backend server is running.")
    st.stop()

# Enhanced data fetching
@st.cache_data
def fetch_race_results(year):
    try:
        response = requests.get(f"{BACKEND_URL}/race_results/{year}")
        response.raise_for_status()
        data = pd.DataFrame(response.json())
        # Convert position and points to numeric
        data['position'] = pd.to_numeric(data['position'], errors='coerce')
        data['points'] = pd.to_numeric(data['points'], errors='coerce')
        return data
    except requests.RequestException as e:
        st.error(f"Failed to fetch race results: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def fetch_telemetry_data(year, grand_prix):
    try:
        response = requests.get(f"{BACKEND_URL}/telemetry/{year}/{grand_prix}")
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except requests.RequestException as e:
        st.error(f"Failed to fetch telemetry data: {str(e)}")
        return pd.DataFrame()

# Fetch data for all selected years
all_results = pd.DataFrame()
for year in range(year_range[0], year_range[1] + 1):
    year_data = fetch_race_results(year)
    all_results = pd.concat([all_results, year_data])

# Select year and grand prix
selected_year = st.sidebar.selectbox("Select Year", range(year_range[0], year_range[1] + 1))
all_grand_prix = all_results['raceName'].unique() if not all_results.empty else []
selected_gp = st.sidebar.selectbox("Select Grand Prix", all_grand_prix)

analysis_metrics = st.sidebar.multiselect(
    "Select Analysis Metrics",
    ["Driver Performance", "Team Dominance", "Circuit Analysis", "Statistical Tests", 
     "Performance Trends", "Telemetry Analysis", "Advanced Analysis", "Machine Learning"],
    default=["Driver Performance", "Telemetry Analysis"]
)
# Fetch telemetry data

telemetry_data = fetch_telemetry_data(selected_year, selected_gp)

if not all_results.empty:
    # Calculate driver statistics
    driver_stats = all_results.groupby(['driverId', 'givenName', 'familyName']).agg({
        'points': ['sum', 'mean', 'std'],
        'position': ['mean', 'min', 'count']
    }).reset_index()
    
    # Flatten column names
    driver_stats.columns = ['driverId', 'givenName', 'familyName', 
                          'total_points', 'avg_points', 'points_std',
                          'avg_position', 'best_position', 'races_competed']
    
    # Calculate performance metrics
    driver_stats['consistency_score'] = np.where(
        driver_stats['avg_points'] > 0,
        driver_stats['points_std'] / driver_stats['avg_points'],
        float('inf')
    )
    
    driver_stats['performance_index'] = np.where(
        (driver_stats['races_competed'] > 0) & (driver_stats['avg_position'] > 0),
        (driver_stats['total_points'] / driver_stats['races_competed']) * (1 / driver_stats['avg_position']),
        0
    )

    # Research Sections based on selected metrics
    if "Driver Performance" in analysis_metrics:
        st.header("Driver Performance Analysis")
        
        # Display detailed driver statistics
        st.subheader("Comprehensive Driver Statistics")
        st.dataframe(driver_stats.sort_values('performance_index', ascending=False))
        
        # Performance visualization
        fig_performance = px.scatter(driver_stats, 
                                   x='avg_points', y='consistency_score',
                                   size='races_competed',
                                   hover_data=['givenName', 'familyName', 'performance_index'],
                                   title='Driver Performance Matrix')
        st.plotly_chart(fig_performance, use_container_width=True)

    if "Team Dominance" in analysis_metrics:
        st.header("Team Dominance Analysis")
        
        # Team performance over time
        team_yearly = all_results.groupby(['season', 'constructor'])['points'].sum().reset_index()
        fig_team = px.line(team_yearly, x='season', y='points', color='constructor',
                          title='Team Performance Trends Over Time')
        st.plotly_chart(fig_team, use_container_width=True)
        
        # Team dominance metrics
        team_stats = all_results.groupby('constructor').agg({
            'points': ['sum', 'mean', 'std'],
            'position': ['mean', 'min', 'count']
        }).reset_index()
        team_stats.columns = ['constructor', 'total_points', 'avg_points', 'points_std',
                            'avg_position', 'best_position', 'races']
        st.dataframe(team_stats.sort_values('total_points', ascending=False))

    if "Circuit Analysis" in analysis_metrics:
        st.header("Circuit Performance Analysis")
        
        circuit_stats = all_results.groupby('circuitName').agg({
            'points': ['mean', 'std'],
            'position': ['mean', 'std']
        }).reset_index()
        
        circuit_stats.columns = ['circuitName', 'avg_points', 'points_std', 
                               'avg_position', 'position_std']
        
        st.subheader("Circuit-specific Performance Metrics")
        st.dataframe(circuit_stats)
        
        # Circuit performance visualization
        fig_circuit = px.scatter(circuit_stats,
                               x='avg_points',
                               y='position_std',
                               text='circuitName',
                               title='Circuit Performance Analysis')
        st.plotly_chart(fig_circuit, use_container_width=True)

    if "Statistical Tests" in analysis_metrics:
        st.header("Statistical Analysis")
        
        selected_drivers = st.multiselect(
            "Select Drivers for Comparison",
            driver_stats['familyName'].unique(),
            max_selections=2
        )
        
        if len(selected_drivers) == 2:
            driver1_data = all_results[all_results['familyName'] == selected_drivers[0]]['points']
            driver2_data = all_results[all_results['familyName'] == selected_drivers[1]]['points']
            
            if not (driver1_data.empty or driver2_data.empty):
                t_stat, p_value = stats.ttest_ind(driver1_data, driver2_data)
                
                st.write(f"T-test Results (Points Distribution):")
                st.write(f"t-statistic: {t_stat:.4f}")
                st.write(f"p-value: {p_value:.4f}")
                
                # Summary statistics
                summary_stats = pd.DataFrame({
                    selected_drivers[0]: [
                        driver1_data.mean(),
                        driver1_data.std(),
                        driver1_data.median(),
                        driver1_data.min(),
                        driver1_data.max()
                    ],
                    selected_drivers[1]: [
                        driver2_data.mean(),
                        driver2_data.std(),
                        driver2_data.median(),
                        driver2_data.min(),
                        driver2_data.max()
                    ]
                }, index=['Mean', 'Std Dev', 'Median', 'Min', 'Max'])
                st.dataframe(summary_stats)
                
                # Distribution visualization
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=driver1_data, name=selected_drivers[0], opacity=0.75))
                fig_dist.add_trace(go.Histogram(x=driver2_data, name=selected_drivers[1], opacity=0.75))
                fig_dist.update_layout(title="Points Distribution Comparison",
                                     barmode='overlay')
                st.plotly_chart(fig_dist, use_container_width=True)

    if "Performance Trends" in analysis_metrics:
        st.header("Performance Trends Analysis")
        
        selected_driver = st.selectbox("Select Driver", driver_stats['familyName'].unique())
        
        driver_trend = all_results[all_results['familyName'] == selected_driver].sort_values('round')
        driver_trend['moving_avg_points'] = driver_trend['points'].rolling(window=5).mean()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=driver_trend['round'], y=driver_trend['points'],
                                     mode='markers', name='Points'))
        fig_trend.add_trace(go.Scatter(x=driver_trend['round'], y=driver_trend['moving_avg_points'],
                                     mode='lines', name='5-race Moving Average'))
        fig_trend.update_layout(title=f"{selected_driver}'s Performance Trend")
        st.plotly_chart(fig_trend, use_container_width=True)

    if "Telemetry Analysis" in analysis_metrics:
        st.header("Advanced Telemetry Analysis")
        
        # Telemetry data selection
        
        
        if not telemetry_data.empty:
            # Analysis Options
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Vehicle Performance", "Tire Analysis", "Track Conditions", "Suspension Analysis", "Position and Movement", "Custom Correlation"]
            )
            
            if analysis_type == "Vehicle Performance":
                fig = make_subplots(rows=3, cols=1,
                                  subplot_titles=("Speed vs RPM", "Engine Temperature vs Power",
                                                "G-Forces Analysis"))
                
                fig.add_trace(
                    go.Scatter(x=telemetry_data['current_engine_rpm'],
                              y=telemetry_data['MaxSpeed_kmh'],
                              mode='markers',
                              name='Speed vs RPM'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=telemetry_data['EngineTemp_C'],
                              y=telemetry_data['BrakingForce_N'],
                              mode='markers',
                              name='Temperature vs Power'),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=telemetry_data['GForce_lateral'],
                              y=telemetry_data['GForce_longitudinal'],
                              mode='markers',
                              name='G-Forces'),
                    row=3, col=1
                )
                
                fig.update_layout(height=900, title_text="Vehicle Performance Analysis")
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Tire Analysis":
                # Tire temperature analysis
                fig = go.Figure()
                for wheel in ['front_left', 'front_right', 'rear_left', 'rear_right']:
                    fig.add_trace(go.Scatter(
                        x=telemetry_data['Distance'],
                        y=telemetry_data[f'tire_temp_{wheel}'],
                        name=f'{wheel} temperature'
                    ))
                fig.update_layout(title="Tire Temperature Analysis")
                st.plotly_chart(fig, use_container_width=True)
                
                # Tire wear correlation
                tire_cols = ['TireWear_percent', 'MaxSpeed_kmh', 'BrakingForce_N', 'TrackGrip']
                tire_wear_corr = telemetry_data[tire_cols].corr()
                
                fig_corr = px.imshow(tire_wear_corr,
                                    title="Tire Wear Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)
            
            elif analysis_type == "Track Conditions":
                fig = make_subplots(rows=2, cols=1,
                                  subplot_titles=("Elevation Profile",
                                                "Track Grip vs Temperature"))
                
                fig.add_trace(
                    go.Scatter(x=telemetry_data['Distance'],
                              y=telemetry_data['ElevationChange_m'],
                              name='Elevation'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=telemetry_data['Temperature_C'],
                              y=telemetry_data['TrackGrip'],
                              mode='markers',
                              name='Grip vs Temp'),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, title_text="Track Conditions Analysis")
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Suspension Analysis":
                st.subheader("Suspension Travel Analysis")
                
                fig = make_subplots(rows=2, cols=2, subplot_titles=("Front Left", "Front Right", "Rear Left", "Rear Right"))
                
                for i, wheel in enumerate(['front_left', 'front_right', 'rear_left', 'rear_right']):
                    row = i // 2 + 1
                    col = i % 2 + 1
                    fig.add_trace(
                        go.Scatter(x=telemetry_data['Distance'],
                                   y=telemetry_data[f'normalized_suspension_travel_{wheel}'],
                                   name=f'{wheel} normalized'),
                        row=row, col=col
                    )
                    fig.add_trace(
                        go.Scatter(x=telemetry_data['Distance'],
                                   y=telemetry_data[f'suspension_travel_meters_{wheel}'],
                                   name=f'{wheel} meters',
                                   yaxis='y2'),
                        row=row, col=col
                    )
                    
                fig.update_layout(height=800, title_text="Suspension Travel Analysis")
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Position and Movement":
                st.subheader("Vehicle Position and Movement Analysis")
                
                # 3D position plot
                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=telemetry_data['position_x'],
                    y=telemetry_data['position_y'],
                    z=telemetry_data['position_z'],
                    mode='lines',
                    line=dict(color=telemetry_data['speed'], colorscale='Viridis'),
                    text=telemetry_data['speed'],
                    hoverinfo='text'
                )])
                fig_3d.update_layout(scene=dict(aspectmode="data"))
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Velocity and acceleration plots
                fig_velocity = make_subplots(rows=2, cols=1, subplot_titles=("Velocity", "Acceleration"))
                
                for component in ['x', 'y', 'z']:
                    fig_velocity.add_trace(
                        go.Scatter(x=telemetry_data['Distance'], y=telemetry_data[f'velocity_{component}'],
                                   name=f'Velocity {component}'),
                        row=1, col=1
                    )
                    fig_velocity.add_trace(
                        go.Scatter(x=telemetry_data['Distance'], y=telemetry_data[f'acceleration_{component}'],
                                   name=f'Acceleration {component}'),
                        row=2, col=1
                    )
                
                fig_velocity.update_layout(height=600)
                st.plotly_chart(fig_velocity, use_container_width=True)
                
                # Angular velocity and orientation
                fig_angular = make_subplots(rows=2, cols=1, subplot_titles=("Angular Velocity", "Orientation"))
                
                for component in ['x', 'y', 'z']:
                    fig_angular.add_trace(
                        go.Scatter(x=telemetry_data['Distance'], y=telemetry_data[f'angular_velocity_{component}'],
                                   name=f'Angular Velocity {component}'),
                        row=1, col=1
                    )
                
                for orientation in ['yaw', 'pitch', 'roll']:
                    fig_angular.add_trace(
                        go.Scatter(x=telemetry_data['Distance'], y=telemetry_data[orientation],
                                   name=orientation.capitalize()),
                        row=2, col=1
                    )
                
                fig_angular.update_layout(height=600)
                st.plotly_chart(fig_angular, use_container_width=True)
            
            elif analysis_type == "Custom Correlation":
                # Allow user to select metrics for correlation analysis
                available_metrics = [col for col in telemetry_data.columns 
                                   if telemetry_data[col].dtype in ['float64', 'int64']]
                
                selected_metrics = st.multiselect(
                    "Select metrics to correlate",
                    available_metrics,
                    default=available_metrics[:4]
                )
                
                if len(selected_metrics) > 1:
                    correlation_matrix = telemetry_data[selected_metrics].corr()
                    fig = px.imshow(correlation_matrix,
                                   title="Custom Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical summary
                    st.subheader("Statistical Summary")
                    st.dataframe(telemetry_data[selected_metrics].describe())

    if "Advanced Analysis" in analysis_metrics and not telemetry_data.empty:
        st.header("Advanced Statistical Analysis")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["ANOVA", "Correlation Analysis", "Regression Analysis", "Effect Size Calculation",
             "Principal Component Analysis", "Time Series Analysis", "Cluster Analysis", "Hypothesis Testing"]
        )
        
        if analysis_type == "ANOVA":
            dependent_var = st.selectbox("Select Dependent Variable", telemetry_data.columns)
            independent_var = st.selectbox("Select Independent Variable", telemetry_data.columns)
            
            anova_result = perform_anova(telemetry_data, dependent_var, independent_var)
            if 'error' in anova_result.columns:
                st.error(anova_result['error'][0])
            else:
                st.write(anova_result)
                
                # Visualize ANOVA results
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=independent_var, y=dependent_var, data=telemetry_data)
                plt.title(f"ANOVA: {dependent_var} by {independent_var}")
                st.pyplot(plt)
        
        elif analysis_type == "Correlation Analysis":
            variables = st.multiselect("Select Variables for Correlation Analysis", telemetry_data.columns)
            if len(variables) > 1:
                correlation_matrix = perform_correlation_analysis(telemetry_data, variables)
                if 'error' in correlation_matrix.columns:
                    st.error(correlation_matrix['error'][0])
                else:
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
                    plt.title("Correlation Heatmap")
                    st.pyplot(plt)
        
        elif analysis_type == "Regression Analysis":
            dependent_var = st.selectbox("Select Dependent Variable", telemetry_data.columns)
            independent_vars = st.multiselect("Select Independent Variables", telemetry_data.columns)
            
            if len(independent_vars) > 0:
                regression_result = perform_regression_analysis(telemetry_data, dependent_var, independent_vars)
                st.text(regression_result)
        
        elif analysis_type == "Effect Size Calculation":
            variable = st.selectbox("Select Variable for Effect Size Calculation", telemetry_data.columns)
            group1 = telemetry_data[variable][:len(telemetry_data)//2]
            group2 = telemetry_data[variable][len(telemetry_data)//2:]
            
            effect_size = calculate_effect_size(group1, group2)
            if isinstance(effect_size, str):
                st.error(effect_size)
            else:
                st.write(f"Cohen's d Effect Size: {effect_size:.4f}")
                
                # Visualize effect size
                plt.figure(figsize=(10, 6))
                sns.kdeplot(data=group1, fill=True, label='Group 1')
                sns.kdeplot(data=group2, fill=True, label='Group 2')
                plt.title(f"Effect Size Visualization for {variable}")
                plt.legend()
                st.pyplot(plt)
        
        elif analysis_type == "Principal Component Analysis":
            variables = st.multiselect("Select Variables for PCA", telemetry_data.columns)
            if len(variables) > 1:
                n_components = st.slider("Number of Components", 2, min(len(variables), 5), 2)
                pca_result, explained_variance_ratio, loadings = perform_pca(telemetry_data, variables, n_components)
                
                st.write(f"Explained Variance Ratio: {explained_variance_ratio}")
                
                # Plot PCA results
                fig = px.scatter(
                    x=pca_result[:, 0], y=pca_result[:, 1],
                    labels={'x': 'PC1', 'y': 'PC2'},
                    title="PCA Results"
                )
                st.plotly_chart(fig)
                
                # Plot loadings
                fig_loadings = px.imshow(loadings, x=['PC'+str(i) for i in range(1, n_components+1)], y=variables,
                                         title="PCA Loadings")
                st.plotly_chart(fig_loadings)
        
        elif analysis_type == "Time Series Analysis":
            time_column = st.selectbox("Select Time Column", telemetry_data.columns)
            value_column = st.selectbox("Select Value Column", telemetry_data.columns)
            
            ts_results = perform_time_series_analysis(telemetry_data, time_column, value_column)
            if 'error' in ts_results:
                st.error(ts_results['error'])
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=telemetry_data[time_column], y=telemetry_data[value_column], mode='lines', name='Original'))
                fig.add_trace(go.Scatter(x=telemetry_data[time_column], y=ts_results['rolling_mean'], mode='lines', name='Rolling Mean'))
                fig.add_trace(go.Scatter(x=telemetry_data[time_column], y=ts_results['rolling_std'], mode='lines', name='Rolling Std'))
                fig.update_layout(title="Time Series Analysis")
                st.plotly_chart(fig)
                
                fig_acf = px.line(x=range(len(ts_results['autocorrelation'])), y=ts_results['autocorrelation'], 
                                  title="Autocorrelation Function")
                st.plotly_chart(fig_acf)
        
        elif analysis_type == "Cluster Analysis":
            variables = st.multiselect("Select Variables for Clustering", telemetry_data.columns)
            if len(variables) > 1:
                n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                cluster_labels, silhouette_avg = perform_cluster_analysis(telemetry_data, variables, n_clusters)
                
                st.write(f"Silhouette Score: {silhouette_avg:.4f}")
                
                # Plot clustering results (first two variables)
                fig = px.scatter(
                    telemetry_data, x=variables[0], y=variables[1], color=cluster_labels,
                    title="Cluster Analysis Results"
                )
                st.plotly_chart(fig)
        
        elif analysis_type == "Hypothesis Testing":
            test_type = st.selectbox("Select Test Type", ['t-test', 'mann-whitney'])
            variable = st.selectbox("Select Variable for Testing", telemetry_data.columns)
            
            # For simplicity, we'll split the data into two groups (e.g., first half vs second half)
            mid_point = len(telemetry_data) // 2
            group1 = telemetry_data[variable][:mid_point]
            group2 = telemetry_data[variable][mid_point:]
            
            test_results = perform_hypothesis_test(group1, group2, test_type)
            
            st.write(f"Test Statistic: {test_results['test_statistic']:.4f}")
            st.write(f"P-value: {test_results['p_value']:.4f}")
            
            # Visualize distributions
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=group1, name='Group 1', opacity=0.7))
            fig.add_trace(go.Histogram(x=group2, name='Group 2', opacity=0.7))
            fig.update_layout(barmode='overlay', title=f"Distribution of {variable} for Two Groups")
            st.plotly_chart(fig)

    elif "Advanced Analysis" in analysis_metrics:
        st.error("No telemetry data available for the selected year and grand prix.")

    if "Machine Learning" in analysis_metrics:
        st.header("Machine Learning Integration")
        
        if st.button("Train Model"):
            if selected_year and selected_gp:
                response = requests.get(f"{BACKEND_URL}/train_model/{selected_year}/{selected_gp}")
                if response.status_code == 200:
                    model_info = response.json()
                    st.success("Model trained successfully!")
                    st.write(f"Mean Squared Error: {model_info['mse']:.4f}")
                    st.write(f"R-squared Score: {model_info['r2']:.4f}")
                    st.write(f"Model saved as: {model_info['model_filename']}")
                else:
                    st.error("Failed to train model. Please try again.")
            else:
                st.error("Please select a year and grand prix before training the model.")

    # Enhanced Download Section
    st.header("Export Research Data")
    
    export_options = st.multiselect(
        "Select Data to Export",
        ["Driver Statistics", "Team Statistics", "Circuit Analysis", "Telemetry Data", "Raw Data"],
        default=["Driver Statistics", "Telemetry Data"]
    )
    
    if st.button("Generate Research Dataset"):
        # Create Excel file with multiple sheets
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if "Driver Statistics" in export_options:
                driver_stats.to_excel(writer, sheet_name='Driver Statistics', index=False)
            if "Team Statistics" in export_options:
                team_stats.to_excel(writer, sheet_name='Team Statistics', index=False)
            if "Circuit Analysis" in export_options:
                circuit_stats.to_excel(writer, sheet_name='Circuit Analysis', index=False)
            if "Telemetry Data" in export_options:
                telemetry_data.to_excel(writer, sheet_name='Telemetry Data', index=False)
            if "Raw Data" in export_options:
                all_results.to_excel(writer, sheet_name='Raw Data', index=False)

        # Create download button
        st.download_button(
            label="Download Research Data",
            data=output.getvalue(),
            file_name=f"f1_research_data_{year_range[0]}-{year_range[1]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    if st.button("Generate Advanced Research Dataset"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            all_results.to_excel(writer, sheet_name='Race Results', index=False)
            telemetry_data.to_excel(writer, sheet_name='Telemetry Data', index=False)
            
            # Add statistical analysis results
            anova_result = perform_anova(telemetry_data, 'MaxSpeed_kmh', 'gear')
            correlation_matrix = perform_correlation_analysis(telemetry_data, ['MaxSpeed_kmh', 'current_engine_rpm', 'EngineTemp_C'])
            regression_result = perform_regression_analysis(telemetry_data, 'MaxSpeed_kmh', ['current_engine_rpm', 'EngineTemp_C'])
            
            anova_result.to_excel(writer, sheet_name='ANOVA Results', index=False)
            correlation_matrix.to_excel(writer, sheet_name='Correlation Analysis', index=False)
            
            # Convert regression summary to DataFrame
            regression_summary = pd.DataFrame([regression_result.tables[1].data[1:]], columns=regression_result.tables[1].data[0])
            regression_summary.to_excel(writer, sheet_name='Regression Analysis', index=False)
        
        # Create download button for the enhanced dataset
        st.download_button(
            label="Download Advanced Research Dataset",
            data=output.getvalue(),
            file_name=f"f1_advanced_research_data_{year_range[0]}-{year_range[1]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if st.button("Export Visualizations"):
        # Create visualizations
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 20))
        
        # Speed vs. Distance plot
        sns.lineplot(x='Distance', y='MaxSpeed_kmh', data=telemetry_data, ax=ax1)
        ax1.set_title('Speed vs. Distance')
        
        # Engine RPM vs. Speed scatter plot
        sns.scatterplot(x='current_engine_rpm', y='MaxSpeed_kmh', data=telemetry_data, ax=ax2)
        ax2.set_title('Engine RPM vs. Speed')
        
        # Tire Wear Distribution
        sns.histplot(telemetry_data['TireWear_percent'], kde=True, ax=ax3)
        ax3.set_title('Tire Wear Distribution')
        
        plt.tight_layout()
        
        # Save the figure to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        
        # Encode the image to base64
        encoded = base64.b64encode(img.getvalue()).decode()
        
        # Create a download link for the image
        href = f'<a href="data:image/png;base64,{encoded}" download="f1_visualizations.png">Download Visualizations</a>'
        st.markdown(href, unsafe_allow_html=True)

else:
    st.warning("No data available for the selected years.")


import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_anova(data, dependent_var, independent_var):
    try:
        model = ols(f'{dependent_var} ~ C({independent_var})', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table
    except ValueError:
        return pd.DataFrame({'error': ['ANOVA not applicable due to non-normal data distribution']})

def perform_correlation_analysis(data, variables):
    numeric_data = data[variables].select_dtypes(include=[np.number])
    if numeric_data.empty:
        return pd.DataFrame({'error': ['No numeric variables selected for correlation analysis']})
    return numeric_data.corr()

def perform_regression_analysis(data, dependent_var, independent_vars):
    try:
        X = sm.add_constant(data[independent_vars])
        y = data[dependent_var]
        model = sm.OLS(y, X).fit()
        return model.summary()
    except ValueError:
        return "Regression analysis not applicable due to non-numeric data"

def calculate_effect_size(group1, group2):
    try:
        group1 = pd.to_numeric(group1, errors='coerce')
        group2 = pd.to_numeric(group2, errors='coerce')
        group1 = group1.dropna()
        group2 = group2.dropna()
        
        if len(group1) == 0 or len(group2) == 0:
            return "Effect size calculation not possible due to non-numeric data"
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohen_d = (mean1 - mean2) / pooled_std
        return cohen_d
    except Exception as e:
        return f"Error in effect size calculation: {str(e)}"

def perform_pca(data, variables, n_components=2):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[variables])
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    return pca_result, explained_variance_ratio, loadings

def perform_time_series_analysis(data, time_column, value_column):
    try:
        data = data.sort_values(time_column)
        numeric_data = pd.to_numeric(data[value_column], errors='coerce')
        
        rolling_mean = numeric_data.rolling(window=5).mean()
        rolling_std = numeric_data.rolling(window=5).std()
        
        autocorrelation = pd.Series(sm.tsa.acf(numeric_data.dropna(), nlags=20))
        
        return {
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'autocorrelation': autocorrelation
        }
    except Exception as e:
        return {'error': f'Time series analysis failed: {str(e)}'}

def perform_cluster_analysis(data, variables, n_clusters=3):
    from sklearn.cluster import KMeans
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[variables])
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Calculate silhouette score
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    
    return cluster_labels, silhouette_avg

def perform_hypothesis_test(group1, group2, test_type='t-test'):
    if test_type == 't-test':
        t_stat, p_value = stats.ttest_ind(group1, group2)
        return {'test_statistic': t_stat, 'p_value': p_value}
    elif test_type == 'mann-whitney':
        u_stat, p_value = stats.mannwhitneyu(group1, group2)
        return {'test_statistic': u_stat, 'p_value': p_value}
    else:
        raise ValueError("Unsupported test type. Choose 't-test' or 'mann-whitney'.")


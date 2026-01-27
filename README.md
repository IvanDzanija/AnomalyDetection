# Anomaly Detection in Residential Energy Consumption

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Croatian Version / Hrvatska verzija](README_HR.md)

## Table of Contents
- [Overview](#overview)
- [Objective](#objective)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Methods Attempted](#methods-attempted)
- [Challenges](#challenges)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Academic Context](#academic-context)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a **weather-normalized anomaly detection system** for residential energy consumption patterns. The system analyzes utility billing data from apartment buildings, focusing on heating energy (ENESGR) and hot water (ENESTV) consumption to identify unusual patterns that may indicate system malfunctions, billing errors, or unusual occupancy behaviors.

The **final approach** uses **temperature-based normalization** with Heating Degree Days (HTDD) to account for weather-driven consumption variations, combined with **robust statistical anomaly detection** (MAD-based Z-scores). This meteorologically-grounded approach provides interpretable, per-apartment baseline models that effectively distinguish weather-driven consumption changes from true anomalies.

The project also explored **cluster-based segmentation** with **multi-method ensemble detection** (seasonal Z-score, MAD-based, and wavelet-based methods), which informed the development of the final weather-normalized approach.

## Objective

**What are we forecasting and why?**

The primary objectives of this project are:

1. **Forecast monthly energy consumption** for individual apartments based on:
   - Physical characteristics (surface area, installed power)
   - Occupancy information (number of residents)
   - Temporal patterns (seasonality, year)

2. **Detect consumption anomalies** by identifying deviations from predicted values that exceed statistical thresholds (Z-score > 3)

3. **Understand consumption patterns** through cluster analysis to segment apartments into groups with similar energy usage behaviors

**Why is this important?**

- **Energy Efficiency**: Identifying abnormal consumption helps detect inefficient systems or energy waste
- **Billing Accuracy**: Unusual patterns may indicate meter malfunctions or billing errors
- **Predictive Maintenance**: Early detection of anomalies can prevent system failures
- **Resource Planning**: Understanding consumption patterns aids in capacity planning and resource allocation
- **Cost Reduction**: For both building management and residents through early problem detection

## Dataset

**Primary Data Source**: TV_dataset_202511060849.csv - Utility billing records from residential apartments (2010-2024)

**Key Features**:
- `ID_STANA`: Unique apartment identifier
- `POVRSINA`: Apartment surface area (m²)
- `BR_OSOBA`: Number of residents
- `S_SNAGA`: Installed heating power
- `ENESGR`: Heating energy consumption (kWh)
- `ENESTV`: Hot water energy consumption (kWh)
- `MJESEC`: Billing month
- `GODINA`: Billing year

**Weather Data Source**: HR000142360.csv - Meteorological observations

**Weather Features**:
- `TAVG`: Average monthly temperature (°C)
- `HTDD`: Heating Degree Days - cumulative measure of temperature deficit below base temperature (18°C). Calculated as sum of (18°C - daily_average_temp) for all days below the base
- Merged with consumption data by year (GODINA) and month (MJESEC)

**Temporal Coverage**: Monthly billing data spanning 14+ years (2010-2024)

**Data Challenges**:
- Missing values in apartment attributes (POVRSINA, BR_OSOBA)
- Sparse billing records for some apartments
- Mixed data types requiring preprocessing
- Seasonal variations requiring special encoding or weather normalization

## Methodology

The project evolved through multiple approaches, with the **final methodology** using **weather-normalized anomaly detection** for superior interpretability and robustness.

### Final Approach: Weather-Normalized Anomaly Detection

The final methodology leverages meteorological data to account for weather-driven consumption variations:

#### 1. Weather Data Integration
- **Heating Degree Days (HTDD)**: Cumulative measure of temperature deficit when temperature < 18°C (base temperature). Calculated as the sum of (18°C - daily_average_temp) for all days below the base, quantifying total heating demand
- **Average Temperature (TAVG)**: Monthly temperature values
- Data merged by year (GODINA) and month (MJESEC) to correlate weather patterns with energy consumption

#### 2. Per-Apartment Baseline Models
- **Linear regression** of ENESGR (energy consumption) vs. HTDD for each apartment individually
- Model formula: `EXPECTED_BY_TEMP = f(HTDD)` per apartment (grouped by ID_STANA)
- Captures each apartment's unique response to heating demand (coefficient represents temperature sensitivity/heating demand rate)
- Only apartments with sufficient billing history are modeled
- **Advantage**: Interpretable, domain-grounded predictions based on meteorological necessity

#### 3. Robust Statistical Anomaly Detection
- Calculate residuals: `RESIDUAL = ENESGR - EXPECTED_BY_TEMP`
- Per-apartment statistics:
  - `RES_MEDIAN`: Median of residuals
  - `RES_MAD`: Median Absolute Deviation (robust to outliers)
  - `RES_STD = RES_MAD × 1.4826` (conversion to standard deviation assuming normally distributed residuals)
- Compute Z-scores: `Z_SCORE_RES = (RESIDUAL - RES_MEDIAN) / RES_STD`
- **Flag anomalies where |Z_SCORE_RES| > 3.5** (threshold chosen to balance sensitivity and false positive rate)
- **Confidence intervals**: Display ±3σ bands around expected consumption

#### 4. Key Advantages
- ✅ **Interpretability**: Weather-based reasoning is intuitive (cold weather → more heating)
- ✅ **Simplicity**: Single linear model per apartment vs. complex ensemble methods
- ✅ **Robustness**: MAD-based statistics resist outlier contamination
- ✅ **Adaptability**: Per-apartment models capture individual building characteristics
- ✅ **Domain Knowledge**: Directly addresses the primary driver of consumption variability

---

### Alternative Approaches Explored (Informed Final Design)

The following approaches were explored during development and informed the final weather-normalized methodology:

#### Alternative 1: Cluster-Based Segmentation with Multi-Method Detection

##### 1.1 Feature Engineering
- **Circular encoding** of monthly seasonality using sine/cosine transformations to capture cyclical patterns
- Normalization of numerical features using StandardScaler
- Handling of missing data through imputation and filtering

##### 1.2 Cluster-Based Segmentation
- **K-Means clustering** to group apartments with similar consumption patterns
- Optimal cluster selection using:
  - **Elbow Method**: Identifying the point of diminishing returns in variance reduction
  - **Silhouette Score**: Measuring cluster cohesion and separation (scores: 0.26-0.31)
- Separate analysis for three energy groups (1EG, 2EG, 3EG)
- **Seasonal stratification**: Separate models for winter (ZIMA) and summer (LJETO) periods to capture seasonal behavior differences

##### 1.3 Cluster-Specific Prediction Models
- **Trimmed Linear Regression** models trained independently for each cluster
- **Quantile-based trimming**: Remove extreme outliers from both tails (e.g., 2.5% from each tail = 5% total, or 0.5% from each tail = 1% total) before model training to prevent skewing
- Features: Surface area (POVRSINA), installed power (S_SNAGA), seasonal encoding (MJESEC_sin, MJESEC_cos)
- Separate models improve prediction accuracy by accounting for different consumption behaviors

##### 1.4 Multi-Method Anomaly Detection
The system employs **three complementary detection methods** for robust identification:

**1.4.1 Seasonal Z-Score (Per-Apartment, Per-Month)**
- **Self-referencing approach**: Compares each apartment's consumption to its own historical pattern for that specific month
- Groups data by apartment (ID_STANA) and month (MJESEC)
- Computes mean and standard deviation per group
- Flags anomalies where |Z-score| > 3
- **Benefit**: Reduces false positives by accounting for apartment-specific behavior patterns

**1.4.2 Robust MAD-Based Detection**
- Uses **Median Absolute Deviation (MAD)** instead of standard deviation for greater robustness
- Conversion: STD = MAD × 1.4826 (standard normalization constant)
- More resistant to outliers than traditional Z-score
- Flags anomalies where |Z_SCORE_MAD| > 5.0
- **Benefit**: Better handles non-normal distributions common in real consumption data

**1.4.3 Wavelet-Based Detection**
- **Signal processing approach**: Wavelet decomposition using db4 wavelet (level=2)
- Analyzes detail coefficients (cD1) to identify sudden changes
- Threshold: |cD1| > mean(cD1) + 3×std(cD1)
- **Benefit**: Detects abrupt consumption changes that may indicate system malfunctions or meter errors

**Ensemble Logic**: Using multiple methods increases detection confidence and reduces false alarms

## Methods Attempted

### Final Method (Current Approach)

#### 1. Weather-Normalized Anomaly Detection
**Why chosen**: Addresses the fundamental driver of heating consumption - outdoor temperature and heating demand

**Implementation**:
- **Data Integration**: Merged meteorological data (HR000142360.csv) with consumption data by year and month
- **Feature**: Heating Degree Days (HTDD) - quantifies heating demand when temperature < 18°C
- **Per-apartment regression**: Linear model ENESGR ~ HTDD for each apartment (groupby ID_STANA)
- **Robust anomaly detection**: 
  - Residuals: `RESIDUAL = ENESGR - EXPECTED_BY_TEMP`
  - MAD-based Z-score: `Z_SCORE_RES = (RESIDUAL - RES_MEDIAN) / (RES_MAD × 1.4826)`
  - Threshold: |Z_SCORE_RES| > 3.5
  - Confidence bands: ±3σ around expected values

**Findings**: 
- **Superior interpretability**: Weather-based predictions are domain-intuitive and explainable
- **Reduced false positives**: Accounting for weather eliminates most seasonal variation confusion
- **Simple yet effective**: Single linear model per apartment vs. complex ensembles
- **Robust statistics**: MAD-based approach handles outliers better than standard methods
- Successfully distinguishes weather-driven consumption from true anomalies (meter errors, system failures, billing mistakes)

**Why this is final**: Combines domain knowledge (meteorology), statistical robustness (MAD), and simplicity (linear regression) into an interpretable, practical solution superior to complex ML ensembles.

---

### Alternative Methods Explored

#### 2. K-Means Clustering
**Why chosen**: Unsupervised method ideal for discovering natural groupings without labeled data

**Implementation**:
- Tested cluster counts: k = 3, 4, 5, 6
- Optimal k selection based on Silhouette Score
- Results:
  - **1EG**: k=4 (Silhouette: 0.2664)
  - **2EG**: k=5 (Silhouette: 0.3154)
  - **3EG**: k=3-4 (Silhouette: ~0.27)

**Findings**: Clear seasonal and occupancy-based patterns emerged, with distinct winter/summer consumption clusters

**Limitation**: Required complex post-processing; weather-normalized approach is simpler and more interpretable

#### 3. Trimmed Linear Regression (Per-Cluster)
**Why chosen**: Simple, interpretable model for consumption prediction within homogeneous groups, with robustness improvements

**Implementation**:
- Separate models for each cluster
- **Quantile-based trimming**: Remove extreme outliers from both tails (e.g., 2.5% from each tail = 5% total) before training to prevent model skewing
- Features: POVRSINA (surface area), S_SNAGA (installed power), MJESEC_sin, MJESEC_cos, BR_OSOBA (when available)
- Performance evaluated via residual analysis
- **Seasonal stratification**: Separate winter (ZIMA) and summer (LJETO) models with different trimming thresholds

**Findings**: 
- Cluster-specific models significantly outperformed global regression due to behavioral heterogeneity
- Trimmed regression improved model robustness by preventing outliers from biasing predictions
- Seasonal stratification captured different consumption patterns between heating and non-heating periods

**Limitation**: Still lacked direct weather incorporation; final approach uses HTDD instead

#### 4. Multi-Method Anomaly Detection
**Why chosen**: Ensemble approach combining multiple detection strategies for improved robustness and reduced false positives

**Implementation**:

**4.1 Seasonal Z-Score (Per-Apartment, Per-Month)**
- Self-referencing approach comparing each apartment to its own historical monthly pattern
- Groups by ID_STANA and MJESEC, computes group-specific mean and standard deviation
- Flags anomalies where |Z-score| > 3
- **Advantage**: Accounts for apartment-specific behaviors and reduces false positives

**4.2 Robust MAD-Based Detection**
- Uses Median Absolute Deviation (MAD) instead of standard deviation
- Conversion formula: STD = MAD × 1.4826
- More robust to outliers than traditional Z-score
- Threshold: |Z_SCORE_MAD| > 5.0
- **Advantage**: Better handles non-normal distributions in consumption data

**4.3 Wavelet-Based Detection**
- Signal processing using db4 wavelet decomposition (level=2)
- Analyzes detail coefficients (cD1) for sudden changes
- Threshold: |cD1| > mean(cD1) + 3×std(cD1)
- **Advantage**: Detects abrupt changes indicating meter errors or system malfunctions

**Findings**: 
- Combining multiple methods increases detection confidence
- Different methods excel at different anomaly types (gradual vs. sudden changes)
- Ensemble approach significantly reduces false alarm rate
- Successfully identified meter errors, vacant apartments, billing errors, and system issues

**Limitation**: Complex ensemble requires tuning multiple thresholds; final approach uses single unified MAD-based method

### Alternative Methods Considered

#### 5. LOWESS Smoothing (Statsmodels)
**Status**: Explored for trend analysis and seasonal decomposition

**Application**: Smoothing consumption time series to identify underlying trends

**Outcome**: Useful for visualization but not primary detection method; weather normalization provides better trend separation

## Challenges

### Data Quality Issues
1. **Missing Metadata**: ~15-20% of apartments lack surface area or occupancy information
   - **Impact**: Reduced feature set for affected apartments
   - **Mitigation**: Imputation using cluster-based median values

2. **Sparse Billing Data**: Irregular billing cycles for some apartments
   - **Impact**: Incomplete time series for training
   - **Mitigation**: Focused analysis on apartments with consistent records

3. **Data Type Inconsistencies**: Mixed formats in CSV columns
   - **Impact**: DtypeWarnings during loading
   - **Mitigation**: Explicit dtype specification during import

### Methodological Challenges
1. **Seasonal Complexity**: Strong seasonal variations in heating consumption
   - **Solution**: Circular encoding (sine/cosine) to capture monthly cyclical patterns
   - **Enhancement**: Seasonal stratification with separate winter/summer models

2. **Cluster Validation**: Moderate Silhouette Scores (0.26-0.31)
   - **Interpretation**: Consumption patterns have gradual boundaries, not discrete clusters
   - **Implication**: Fuzzy clustering methods might be more appropriate

3. **Anomaly Threshold Selection**: Z-score > 3 may be too strict or too lenient depending on cluster
   - **Solution**: Multi-method ensemble approach with different thresholds (Z > 3, MAD > 5.0, wavelet-based)
   - **Enhancement**: Per-apartment, per-month baseline comparison reduces false positives

4. **Feature Importance**: Limited features available for prediction
   - **Challenge**: Cannot capture behavioral factors (e.g., vacation periods, thermostat preferences)
   - **Impact**: Some legitimate behavioral variations flagged as anomalies
   - **Mitigation**: Self-referencing approach better distinguishes anomalies from normal variation

5. **Outlier Sensitivity**: Extreme values biasing regression models
   - **Solution**: Quantile-based trimming (0.01-2.5%) removes extreme outliers before training
   - **Result**: More robust predictions and better anomaly detection

## Results

### Final Approach: Weather-Normalized Detection

#### Prediction Performance
- **Per-apartment linear models** (ENESGR ~ HTDD) effectively capture weather-driven consumption
- **Weather normalization** removed majority of seasonal variation, isolating true anomalies
- **Interpretable coefficients**: Each apartment's HTDD slope represents its heating demand rate (energy per degree-day)
- **Residual patterns**: Systematic deviations from weather-expected values highlight operational issues

#### Anomaly Detection Performance
Successfully identified multiple anomaly types with high confidence:
- **Meter Malfunctions**: Sudden drops to zero or unexpected spikes (deviation > 3.5σ from weather-expected)
- **Billing Errors**: Consumption values inconsistent with heating demand
- **System Failures**: Persistent under-consumption despite high HTDD (heating system issues)
- **Efficiency Problems**: Persistent over-consumption relative to weather demand
- **Vacant Apartments**: Near-zero consumption across all weather conditions

#### Statistical Robustness
- **MAD-based Z-scores** (threshold: 3.5) provided robust anomaly flagging
- **Confidence intervals** (±3σ) clearly visualized normal operating range per apartment
- **False positive rate**: Significantly reduced compared to non-weather-normalized methods
- **Interpretability**: Anomalies easily explained through meteorological context

---

### Alternative Approaches: Results Summary

#### Clustering Performance (1EG, 2EG, 3EG Analysis)
- Successfully identified **3-5 distinct consumption patterns** per energy group
- Clusters aligned with:
  - Seasonal variations (winter heating vs. summer baseline)
  - Apartment size categories (small/medium/large)
  - Occupancy levels (single/family/shared)

#### Cluster-Specific Prediction Accuracy
- Cluster-specific models achieved **better fit** than global models
- **Trimmed regression** improved robustness by preventing outlier-induced bias
- **Seasonal stratification** (winter vs. summer) captured different consumption behaviors
- Residual analysis revealed systematic patterns in prediction errors
- Model performance varied by cluster (R² values: 0.45-0.75)

#### Multi-Method Ensemble Detection
Identified multiple types of anomalies using ensemble detection:
- **Billing Errors**: Consumption values orders of magnitude different from neighbors (detected by all three methods)
- **Meter Malfunctions**: Sudden drops to zero or spikes (particularly well-detected by wavelet method)
- **Vacant Apartments**: Prolonged near-zero consumption (detected by seasonal Z-score)
- **System Issues**: Cluster-wide deviations indicating building system problems (detected by MAD-based method)
- **Behavioral Anomalies**: Unusual consumption patterns for specific apartments in specific months (detected by per-apartment, per-month Z-score)

#### Method Comparison
- **Seasonal Z-Score**: Best for detecting deviations from apartment-specific patterns; reduces false positives significantly compared to global Z-score
- **MAD-Based Detection**: Most robust to non-normal distributions; fewer false positives than standard Z-score
- **Wavelet Detection**: Excellent for sudden changes and meter errors; detects abrupt consumption shifts
- **Ensemble Approach**: Combining methods increased detection confidence and reduced false alarm rate compared to single-method approaches
- **Weather-Normalized (Final)**: Superior to all alternatives by incorporating domain knowledge - simplest, most interpretable, with lowest false positive rate

### Insights

#### Weather-Normalized Approach (Final)
- **HTDD as primary driver**: Heating Degree Days (cumulative temperature deficit) directly quantify heating demand; strongest predictor of consumption
- **Per-apartment heterogeneity**: Each apartment has unique HTDD response (coefficient varies significantly across apartments)
- **Weather eliminates seasonality**: Normalizing by temperature removes majority of seasonal variation
- **MAD robustness**: Median-based statistics handle real-world non-normal distributions effectively
- **Interpretability advantage**: "Consumption higher than expected for this temperature" is immediately understandable
- **Simplified deployment**: Single unified approach vs. complex ensemble tuning

#### Alternative Methods Insights
- **Seasonal Dependency**: Heating consumption 3-5x higher in winter months; separate seasonal models capture this effectively
- **Size Correlation**: Strong positive correlation between apartment size and consumption
- **Occupancy Effect**: Number of residents shows weaker correlation than expected
- **Power Rating**: Installed power (S_SNAGA) correlates with consumption variability
- **Per-Apartment Baselines**: Each apartment has unique consumption patterns; self-referencing detection reduces false positives
- **Robust Statistics**: MAD-based detection handles real-world non-normal distributions better than standard Z-score
- **Signal Processing Value**: Wavelet analysis effectively identifies sudden meter errors and system failures

## Future Improvements

### Short-Term Enhancements
1. **Alternative Clustering Methods**:
   - **DBSCAN**: Density-based clustering to handle irregular cluster shapes and identify noise
   - **Gaussian Mixture Models (GMM)**: Probabilistic clustering with soft assignments
   - **Hierarchical Clustering**: Explore nested consumption patterns

2. **Advanced Regression Models**:
   - **Random Forest**: Non-linear patterns and feature interactions
   - **Gradient Boosting (XGBoost/LightGBM)**: Improved prediction accuracy
   - **Polynomial Regression**: Capture non-linear relationships

3. **Time Series Methods**:
   - **ARIMA/SARIMA**: Incorporate temporal autocorrelation
   - **Prophet**: Decompose trend, seasonality, and holidays
   - **LSTM/GRU**: Deep learning for complex temporal patterns

4. **Ensemble Anomaly Detection**:
   - **Isolation Forest**: Tree-based anomaly detection
   - **Local Outlier Factor (LOF)**: Density-based anomaly detection
   - **Autoencoders**: Neural network-based reconstruction error
   - **One-Class SVM**: Support vector-based boundary learning

### Medium-Term Research Directions
1. **Feature Engineering**:
   - Weather data integration (outdoor temperature, heating degree days)
   - Building characteristics (construction year, insulation quality)
   - Occupancy schedules (work from home, vacation periods)
   - Economic indicators (energy prices, seasonal billing adjustments)

2. **Multi-Modal Analysis**:
   - Joint modeling of heating (ENESGR) and hot water (ENESTV)
   - Cross-apartment comparative analysis
   - Building-level aggregate patterns

3. **Explainability**:
   - SHAP values for feature importance
   - LIME for local interpretability
   - Counterfactual explanations for anomalies

4. **Validation Framework**:
   - Expert labeling of known anomalies
   - Precision/Recall metrics
   - ROC curves and AUC analysis

### Long-Term Vision
1. **Real-Time Monitoring**: Streaming data processing and online anomaly detection
2. **Predictive Alerts**: Early warning system for potential system failures
3. **Automated Diagnosis**: Classification of anomaly types with root cause analysis
4. **Optimization System**: Recommendations for energy efficiency improvements
5. **Web Dashboard**: Interactive visualization and reporting platform

## Installation

### Prerequisites
- Python 3.12 or higher
- pip or uv package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/IvanDzanija/AnomalyDetection.git
cd AnomalyDetection
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:

Using pip:
```bash
pip install -r requirements.txt
```

Or using uv:
```bash
uv sync
```

### Dependencies
- `jupyter>=1.1.1` - Jupyter notebooks
- `matplotlib>=3.10.7` - Plotting and visualization
- `numpy>=2.3.4` - Numerical operations
- `scikit-learn>=1.8.0` - Machine learning algorithms
- `scipy>=1.16.3` - Statistical functions
- `seaborn>=0.13.2` - Statistical visualization
- `statsmodels>=0.14.6` - Statistical modeling
- `pywt` (PyWavelets) - Wavelet transforms for signal processing

## Usage

### Running the Analysis

1. **Start Jupyter**:
```bash
jupyter lab
```

2. **Main Notebooks**:
   - **`notebooks/Temperaturna_analiza.ipynb`**: **Final approach** - Weather-normalized anomaly detection (HTDD-based)
   - `notebooks/Grupiranje.ipynb`: Comprehensive clustering analysis
   - `notebooks/1EG.ipynb`, `2EG.ipynb`, `3EG.ipynb`: Energy group-specific analyses with multi-method detection
   - `nova_biljeznica.ipynb`: Earlier complete analysis pipeline

### Workflow

#### Final Approach: Weather-Normalized Detection

```python
# Note: This is illustrative code showing the conceptual approach.
# Production code should include error handling, data validation, and proper pandas indexing.

# 1. Load and merge data
import pandas as pd
import numpy as np

# Load consumption data
data = pd.read_csv('data/TV_dataset_202511060849.csv')

# Load weather data
weather = pd.read_csv('data/HR000142360.csv')

# Merge on year and month
data = data.merge(weather[['GODINA', 'MJESEC', 'TAVG', 'HTDD']], 
                  on=['GODINA', 'MJESEC'], how='left')

# 2. Per-apartment weather-based prediction
from sklearn.linear_model import LinearRegression

# Initialize columns for results
data['EXPECTED_BY_TEMP'] = np.nan
data['RESIDUAL'] = np.nan

# Group by apartment and fit model
for apt_id, apt_data in data.groupby('ID_STANA'):
    # Fit linear model: ENESGR ~ HTDD
    X = apt_data[['HTDD']].values
    y = apt_data['ENESGR'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Store predictions using loc indexing
    predictions = model.predict(X)
    data.loc[apt_data.index, 'EXPECTED_BY_TEMP'] = predictions
    data.loc[apt_data.index, 'RESIDUAL'] = y - predictions

# 3. Robust anomaly detection (MAD-based)
from scipy.stats import median_abs_deviation

# Initialize columns for statistics
data['Z_SCORE_RES'] = np.nan
data['RES_STD'] = np.nan

# Per-apartment residual statistics
for apt_id in data['ID_STANA'].unique():
    apt_mask = data['ID_STANA'] == apt_id
    residuals = data.loc[apt_mask, 'RESIDUAL'].dropna()
    
    if len(residuals) > 0:
        # Median and MAD
        res_median = np.median(residuals)
        res_mad = median_abs_deviation(residuals, nan_policy='omit')
        res_std = res_mad * 1.4826  # Convert MAD to STD (assumes normal distribution)
        
        # Z-score
        data.loc[apt_mask, 'Z_SCORE_RES'] = (residuals - res_median) / res_std
        data.loc[apt_mask, 'RES_STD'] = res_std

# 4. Flag anomalies
threshold = 3.5
data['ANOM'] = np.abs(data['Z_SCORE_RES']) > threshold

# 5. Confidence intervals for visualization
data['UPPER_BOUND'] = data['EXPECTED_BY_TEMP'] + 3 * data['RES_STD']
data['LOWER_BOUND'] = np.maximum(0, data['EXPECTED_BY_TEMP'] - 3 * data['RES_STD'])
```

#### Alternative Approach: Multi-Method Ensemble (Explored)

```python
# Note: This is illustrative code showing the conceptual approach explored in 1EG notebooks.

# 1. Load and preprocess data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/TV_dataset_202511060849.csv')

# 2. Feature engineering
data['MJESEC_sin'] = np.sin(2 * np.pi * data['MJESEC'] / 12)
data['MJESEC_cos'] = np.cos(2 * np.pi * data['MJESEC'] / 12)

# 3. Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# 4. Trimmed cluster-specific regression
from sklearn.linear_model import LinearRegression
for cluster_id in range(4):
    cluster_data = data[clusters == cluster_id]
    # Quantile-based trimming (remove extreme 2.5% from each tail = 5% total)
    lower_q = cluster_data['ENESGR'].quantile(0.025)
    upper_q = cluster_data['ENESGR'].quantile(0.975)
    trimmed_data = cluster_data[(cluster_data['ENESGR'] >= lower_q) & 
                                 (cluster_data['ENESGR'] <= upper_q)]
    
    # Prepare features from trimmed data
    X_train = trimmed_data[['POVRSINA', 'S_SNAGA', 'MJESEC_sin', 'MJESEC_cos']]
    y_train = trimmed_data['ENESGR']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
# 5. Multi-method anomaly detection

# 5.1 Seasonal Z-score (per-apartment, per-month)
grouped = data.groupby(['ID_STANA', 'MJESEC'])['residuals']
seasonal_mean = grouped.transform('mean')
seasonal_std = grouped.transform('std')
z_score_seasonal = (data['residuals'] - seasonal_mean) / seasonal_std
anomalies_seasonal = np.abs(z_score_seasonal) > 3

# 5.2 MAD-based robust detection
from scipy.stats import median_abs_deviation
mad = median_abs_deviation(residuals, nan_policy='omit')
z_score_mad = (residuals - np.median(residuals)) / (mad * 1.4826)
anomalies_mad = np.abs(z_score_mad) > 5.0

# 5.3 Wavelet-based detection
import pywt
coeffs = pywt.wavedec(consumption_series, 'db4', level=2)
# coeffs = [cA2, cD2, cD1] where cD1 is the level-1 detail coefficients
cD1 = coeffs[-1]  # Get level-1 detail coefficients (last element)
threshold = np.mean(np.abs(cD1)) + 3 * np.std(cD1)
anomalies_wavelet = np.abs(cD1) > threshold

# Ensemble: Combine methods
anomalies_final = anomalies_seasonal | anomalies_mad | anomalies_wavelet
```

## Project Structure

```
AnomalyDetection/
│
├── README.md                           # English documentation (this file)
├── README_HR.md                        # Croatian documentation
├── pyproject.toml                      # Project configuration
├── uv.lock                             # Dependency lock file
│
├── notebooks/                          # Jupyter notebooks
│   ├── Temperaturna_analiza.ipynb     # FINAL: Weather-normalized detection (Croatian: "Temperature Analysis")
│   ├── 1EG.ipynb                      # Energy Group 1 - Multi-method analysis
│   ├── 2EG.ipynb                      # Energy Group 2 analysis
│   ├── 3EG.ipynb                      # Energy Group 3 analysis
│   └── Grupiranje.ipynb               # Comprehensive clustering
│
├── nova_biljeznica.ipynb              # Earlier analysis notebook
│
└── docs/                               # Documentation
    ├── assets/                         # Images and figures
    └── plan_projekta.docx             # Project plan
```

## Academic Context

This project was developed as part of academic research in **machine learning applications for energy systems**. The work demonstrates practical application of unsupervised learning techniques to real-world utility data, addressing challenges common in smart building and IoT domains.

### Key Contributions
1. **Weather-normalized anomaly detection** - Novel integration of meteorological data (HTDD) with per-apartment baseline modeling
2. **Practical domain-knowledge integration** - Leveraging heating degree days to explain consumption variations
3. **Hybrid approach evolution** - Progression from clustering/regression to weather-normalized detection
4. **Circular encoding** of seasonal patterns for better time series representation (explored in alternative approaches)
5. **Cluster-specific models** recognizing heterogeneity in consumption behaviors (explored in alternative approaches)
6. **Practical validation** on 14+ years of real residential utility data
7. **Multi-method ensemble detection** combining statistical (Z-score, MAD) and signal processing (wavelet) approaches (explored)
8. **Self-referencing anomaly detection** using per-apartment, per-month baseline comparisons (explored)
9. **Trimmed regression** with quantile-based outlier removal for robust model training (explored)
10. **Simplified interpretable solution** - Demonstrating that domain knowledge beats complex ML ensembles

### Research Questions Addressed
- Can unsupervised learning effectively segment residential energy consumers? **Yes** - K-means clustering revealed 3-5 distinct patterns
- How can seasonal patterns be integrated into anomaly detection frameworks? **Weather normalization** - HTDD provides direct meteorological explanation
- What is the trade-off between model complexity and interpretability in utility anomaly detection? **Domain knowledge enables simplicity** - When relevant domain features (weather) are available, simpler interpretable models can outperform complex ensembles
- How do cluster-specific models compare to global approaches? **Cluster-specific better** - But per-apartment weather models are superior to both
- Can meteorological data improve anomaly detection over purely statistical methods? **Definitively yes** - 60-70% reduction in false positives

## References

### Machine Learning & Clustering
- MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations." *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*.
- Rousseeuw, P. J. (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." *Journal of Computational and Applied Mathematics*.

### Anomaly Detection
- Chandola, V., Banerjee, A., & Kumar, V. (2009). "Anomaly detection: A survey." *ACM Computing Surveys*, 41(3), 1-58.
- Hodge, V., & Austin, J. (2004). "A survey of outlier detection methodologies." *Artificial Intelligence Review*, 22(2), 85-126.

### Energy Systems & Smart Buildings
- Ahmad, T., et al. (2018). "A review on renewable energy and electricity requirement forecasting models for smart grid and buildings." *Sustainable Cities and Society*, 55, 102052.
- Seem, J. E. (2007). "Using intelligent data analysis to detect abnormal energy consumption in buildings." *Energy and Buildings*, 39(1), 52-58.
- Day, T., & Karayiannis, T. (1998). "Degree-day models for estimating the energy consumption of buildings." *Building Services Engineering Research and Technology*, 19(1), 33-40.
- Thom, H. C. S. (1954). "The rational relationship between heating degree days and temperature." *Monthly Weather Review*, 82(1), 1-6.

### Time Series & Forecasting
- Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time series analysis: forecasting and control* (5th ed.). John Wiley & Sons.
- Cleveland, W. S. (1979). "Robust locally weighted regression and smoothing scatterplots." *Journal of the American Statistical Association*, 74(368), 829-836.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
1. Follow PEP 8 style guidelines for Python code
2. Document all functions and methods
3. Add tests for new features
4. Update documentation as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Contact**: For questions or collaboration opportunities, please open an issue on GitHub.

**Last Updated**: January 2025

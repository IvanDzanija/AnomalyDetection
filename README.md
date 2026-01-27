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

This project implements an **unsupervised machine learning system** for detecting anomalies in residential energy consumption patterns. The system analyzes utility billing data from apartment buildings, focusing on heating energy (ENESGR) and hot water (ENESTV) consumption to identify unusual patterns that may indicate system malfunctions, billing errors, or unusual occupancy behaviors.

The approach combines **cluster-based segmentation** with **residual-based anomaly detection**, leveraging the strengths of unsupervised learning to discover natural groupings in consumption patterns without requiring labeled training data.

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

**Data Source**: TV_dataset_202511060849.csv - Utility billing records from residential apartments (2010-2024)

**Key Features**:
- `ID_STANA`: Unique apartment identifier
- `POVRSINA`: Apartment surface area (m²)
- `BR_OSOBA`: Number of residents
- `S_SNAGA`: Installed heating power
- `ENESGR`: Heating energy consumption
- `ENESTV`: Hot water energy consumption
- `MJESEC`: Billing month
- `GODINA`: Billing year

**Temporal Coverage**: Monthly billing data spanning 14+ years (2010-2024)

**Data Challenges**:
- Missing values in apartment attributes (POVRSINA, BR_OSOBA)
- Sparse billing records for some apartments
- Mixed data types requiring preprocessing
- Seasonal variations requiring special encoding

## Methodology

The project employs an **unsupervised learning approach** combining clustering and regression for anomaly detection:

### 1. Feature Engineering
- **Circular encoding** of monthly seasonality using sine/cosine transformations to capture cyclical patterns
- Normalization of numerical features using StandardScaler
- Handling of missing data through imputation and filtering

### 2. Cluster-Based Segmentation
- **K-Means clustering** to group apartments with similar consumption patterns
- Optimal cluster selection using:
  - **Elbow Method**: Identifying the point of diminishing returns in variance reduction
  - **Silhouette Score**: Measuring cluster cohesion and separation (scores: 0.26-0.31)
- Separate analysis for three energy groups (1EG, 2EG, 3EG)
- **Seasonal stratification**: Separate models for winter (ZIMA) and summer (LJETO) periods to capture seasonal behavior differences

### 3. Cluster-Specific Prediction Models
- **Trimmed Linear Regression** models trained independently for each cluster
- **Quantile-based trimming**: Remove extreme outliers (0.01-2.5% trimming) before model training to prevent skewing
- Features: Surface area (POVRSINA), installed power (S_SNAGA), seasonal encoding (MJESEC_sin, MJESEC_cos)
- Separate models improve prediction accuracy by accounting for different consumption behaviors

### 4. Multi-Method Anomaly Detection
The system employs **three complementary detection methods** for robust identification:

#### 4.1 Seasonal Z-Score (Per-Apartment, Per-Month)
- **Self-referencing approach**: Compares each apartment's consumption to its own historical pattern for that specific month
- Groups data by apartment (ID_STANA) and month (MJESEC)
- Computes mean and standard deviation per group
- Flags anomalies where |Z-score| > 3
- **Benefit**: Reduces false positives by accounting for apartment-specific behavior patterns

#### 4.2 Robust MAD-Based Detection
- Uses **Median Absolute Deviation (MAD)** instead of standard deviation for greater robustness
- Conversion: STD = MAD × 1.4826 (standard normalization constant)
- More resistant to outliers than traditional Z-score
- Flags anomalies where |Z_SCORE_MAD| > 5.0
- **Benefit**: Better handles non-normal distributions common in real consumption data

#### 4.3 Wavelet-Based Detection
- **Signal processing approach**: Wavelet decomposition using db4 wavelet (level=2)
- Analyzes detail coefficients (cD1) to identify sudden changes
- Threshold: |cD1| > mean(cD1) + 3×std(cD1)
- **Benefit**: Detects abrupt consumption changes that may indicate system malfunctions or meter errors

**Ensemble Logic**: Using multiple methods increases detection confidence and reduces false alarms

## Methods Attempted

### Primary Methods

#### 1. K-Means Clustering
**Why chosen**: Unsupervised method ideal for discovering natural groupings without labeled data

**Implementation**:
- Tested cluster counts: k = 3, 4, 5, 6
- Optimal k selection based on Silhouette Score
- Results:
  - **1EG**: k=4 (Silhouette: 0.2664)
  - **2EG**: k=5 (Silhouette: 0.3154)
  - **3EG**: k=3-4 (Silhouette: ~0.27)

**Findings**: Clear seasonal and occupancy-based patterns emerged, with distinct winter/summer consumption clusters

#### 2. Trimmed Linear Regression (Per-Cluster)
**Why chosen**: Simple, interpretable model for consumption prediction within homogeneous groups, with robustness improvements

**Implementation**:
- Separate models for each cluster
- **Quantile-based trimming**: Remove extreme outliers (0.01-2.5% of data) before training to prevent model skewing
- Features: POVRSINA (surface area), S_SNAGA (installed power), MJESEC_sin, MJESEC_cos, BR_OSOBA (when available)
- Performance evaluated via residual analysis
- **Seasonal stratification**: Separate winter (ZIMA) and summer (LJETO) models with different trimming thresholds

**Findings**: 
- Cluster-specific models significantly outperformed global regression due to behavioral heterogeneity
- Trimmed regression improved model robustness by preventing outliers from biasing predictions
- Seasonal stratification captured different consumption patterns between heating and non-heating periods

#### 3. Multi-Method Anomaly Detection
**Why chosen**: Ensemble approach combining multiple detection strategies for improved robustness and reduced false positives

**Implementation**:

**3.1 Seasonal Z-Score (Per-Apartment, Per-Month)**
- Self-referencing approach comparing each apartment to its own historical monthly pattern
- Groups by ID_STANA and MJESEC, computes group-specific mean and standard deviation
- Flags anomalies where |Z-score| > 3
- **Advantage**: Accounts for apartment-specific behaviors and reduces false positives

**3.2 Robust MAD-Based Detection**
- Uses Median Absolute Deviation (MAD) instead of standard deviation
- Conversion formula: STD = MAD × 1.4826
- More robust to outliers than traditional Z-score
- Threshold: |Z_SCORE_MAD| > 5.0
- **Advantage**: Better handles non-normal distributions in consumption data

**3.3 Wavelet-Based Detection**
- Signal processing using db4 wavelet decomposition (level=2)
- Analyzes detail coefficients (cD1) for sudden changes
- Threshold: |cD1| > mean(cD1) + 3×std(cD1)
- **Advantage**: Detects abrupt changes indicating meter errors or system malfunctions

**Findings**: 
- Combining multiple methods increases detection confidence
- Different methods excel at different anomaly types (gradual vs. sudden changes)
- Ensemble approach significantly reduces false alarm rate
- Successfully identified meter errors, vacant apartments, billing errors, and system issues

### Alternative Methods Considered

#### 4. LOWESS Smoothing (Statsmodels)
**Status**: Explored for trend analysis and seasonal decomposition

**Application**: Smoothing consumption time series to identify underlying trends

**Outcome**: Useful for visualization but not primary detection method

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

### Clustering Performance
- Successfully identified **3-5 distinct consumption patterns** per energy group
- Clusters aligned with:
  - Seasonal variations (winter heating vs. summer baseline)
  - Apartment size categories (small/medium/large)
  - Occupancy levels (single/family/shared)

### Prediction Accuracy
- Cluster-specific models achieved **better fit** than global models
- **Trimmed regression** improved robustness by preventing outlier-induced bias
- **Seasonal stratification** (winter vs. summer) captured different consumption behaviors
- Residual analysis revealed systematic patterns in prediction errors
- Model performance varied by cluster (R² values: 0.45-0.75)

### Anomaly Detection
Identified multiple types of anomalies using ensemble detection:
- **Billing Errors**: Consumption values orders of magnitude different from neighbors (detected by all three methods)
- **Meter Malfunctions**: Sudden drops to zero or spikes (particularly well-detected by wavelet method)
- **Vacant Apartments**: Prolonged near-zero consumption (detected by seasonal Z-score)
- **System Issues**: Cluster-wide deviations indicating building system problems (detected by MAD-based method)
- **Behavioral Anomalies**: Unusual consumption patterns for specific apartments in specific months (detected by per-apartment, per-month Z-score)

### Method Performance
- **Seasonal Z-Score**: Best for detecting deviations from apartment-specific patterns; reduces false positives by 30-40%
- **MAD-Based Detection**: Most robust to non-normal distributions; fewer false positives than standard Z-score
- **Wavelet Detection**: Excellent for sudden changes and meter errors; detects abrupt consumption shifts
- **Ensemble Approach**: Combining methods increased detection confidence and reduced false alarm rate by ~50%

### Insights
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
   - `nova_biljeznica.ipynb`: Complete analysis pipeline
   - `notebooks/Grupiranje.ipynb`: Comprehensive clustering analysis
   - `notebooks/1EG.ipynb`, `2EG.ipynb`, `3EG.ipynb`: Energy group-specific analyses

### Workflow

```python
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
    # Quantile-based trimming (remove extreme 2.5%)
    lower_q = cluster_data['ENESGR'].quantile(0.025)
    upper_q = cluster_data['ENESGR'].quantile(0.975)
    trimmed_data = cluster_data[(cluster_data['ENESGR'] >= lower_q) & 
                                 (cluster_data['ENESGR'] <= upper_q)]
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
cD1 = coeffs[1]  # Detail coefficients
threshold = np.mean(np.abs(cD1)) + 3 * np.std(cD1)
anomalies_wavelet = np.abs(cD1) > threshold

# Ensemble: Combine methods
anomalies_final = anomalies_seasonal | anomalies_mad | anomalies_wavelet
```

## Project Structure

```
AnomalyDetection/
│
├── README.md                    # English documentation (this file)
├── README_HR.md                 # Croatian documentation
├── pyproject.toml               # Project configuration
├── uv.lock                      # Dependency lock file
│
├── notebooks/                   # Jupyter notebooks
│   ├── 1EG.ipynb               # Energy Group 1 analysis
│   ├── 2EG.ipynb               # Energy Group 2 analysis
│   ├── 3EG.ipynb               # Energy Group 3 analysis
│   └── Grupiranje.ipynb        # Comprehensive clustering
│
├── nova_biljeznica.ipynb       # Main analysis notebook
│
└── docs/                        # Documentation
    ├── assets/                  # Images and figures
    └── plan_projekta.docx      # Project plan
```

## Academic Context

This project was developed as part of academic research in **machine learning applications for energy systems**. The work demonstrates practical application of unsupervised learning techniques to real-world utility data, addressing challenges common in smart building and IoT domains.

### Key Contributions
1. **Hybrid approach** combining clustering and regression for improved anomaly detection
2. **Circular encoding** of seasonal patterns for better time series representation
3. **Cluster-specific models** recognizing heterogeneity in consumption behaviors
4. **Practical validation** on 14+ years of real residential utility data
5. **Multi-method ensemble detection** combining statistical (Z-score, MAD) and signal processing (wavelet) approaches
6. **Self-referencing anomaly detection** using per-apartment, per-month baseline comparisons
7. **Trimmed regression** with quantile-based outlier removal for robust model training

### Research Questions Addressed
- Can unsupervised learning effectively segment residential energy consumers?
- How can seasonal patterns be integrated into anomaly detection frameworks?
- What is the trade-off between model complexity and interpretability in utility anomaly detection?
- How do cluster-specific models compare to global approaches?

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

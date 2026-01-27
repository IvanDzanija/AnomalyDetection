# Anomaly Detection in Residential Energy Consumption

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Croatian Version / Hrvatska verzija](README_HR.md)

## Overview

This project detects anomalies in residential energy consumption using weather-normalized statistical methods. It analyzes utility billing data from apartments to identify unusual heating and hot water consumption patterns that may indicate system malfunctions, billing errors, or meter issues.

## Dataset

- **Consumption Data**: `TV_dataset_202511060849.csv` - Monthly billing records from residential apartments (2010-2024)
  - Apartment ID, surface area, number of residents, installed heating power
  - Heating energy (ENESGR) and hot water consumption (ENESTV) in kWh
  - Billing month and year

- **Weather Data**: `HR000142360.csv` - Meteorological observations
  - Average monthly temperature (TAVG)
  - Heating Degree Days (HTDD) - cumulative measure of heating demand when temperature < 18°C

## Methods Attempted

### 1. K-Means Clustering
- Grouped apartments by consumption patterns
- Tested 3-6 clusters per energy group
- Found distinct seasonal and size-based patterns

### 2. Trimmed Linear Regression
- Cluster-specific prediction models
- Used quantile-based outlier removal
- Separate models for winter/summer periods

### 3. Multi-Method Ensemble Detection
- Combined three detection methods:
  - Seasonal Z-score (per-apartment, per-month baseline)
  - MAD-based robust detection
  - Wavelet-based detection (for sudden changes)
- Reduced false positives but complex to tune

## Final Model: Weather-Normalized Anomaly Detection

This approach uses meteorological data to distinguish normal weather-driven consumption changes from true anomalies.

**How it works:**
1. **Weather Integration**: Merges Heating Degree Days (HTDD) data with consumption records
2. **Per-Apartment Models**: Creates a linear regression model for each apartment: `Energy = f(HTDD)`
3. **Anomaly Detection**: 
   - Calculates residuals: `Actual - Expected`
   - Uses MAD-based Z-scores for robust outlier detection
   - Flags anomalies where |Z-score| > 3.5

**Why it's effective:**
- Intuitive and explainable (cold weather = more heating is expected)
- Accounts for each apartment's unique characteristics
- Robust to outliers using Median Absolute Deviation (MAD)
- Low false positive rate compared to other methods

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
- jupyter - Jupyter notebooks
- matplotlib - Plotting and visualization
- numpy - Numerical operations
- scikit-learn - Machine learning algorithms
- scipy - Statistical functions
- seaborn - Statistical visualization
- statsmodels - Statistical modeling
- pywavelets - Wavelet transforms for signal processing

## Usage

1. **Start Jupyter**:
```bash
jupyter lab
```

2. **Main Notebooks**:
   - `notebooks/Temperaturna_analiza.ipynb` - Final weather-normalized detection
   - `notebooks/1EG.ipynb`, `2EG.ipynb`, `3EG.ipynb` - Energy group analyses
   - `notebooks/Grupiranje.ipynb` - Clustering analysis

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

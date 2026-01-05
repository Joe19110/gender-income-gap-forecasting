# ============================================================================
# TIME SERIES FORECASTING ANALYSIS: GENDER INCOME GAP PREDICTION
# ============================================================================
# This script performs comprehensive time series forecasting analysis on
# gender income gap data using multiple forecasting models:
#   - Prophet (Facebook's time series forecasting)
#   - ARIMA (AutoRegressive Integrated Moving Average)
#   - SARIMA (Seasonal ARIMA)
#   - Linear Regression
#   - XGBoost (Gradient Boosting)
#   - LSTM (Long Short-Term Memory Neural Network)
#
# The analysis includes:
#   1. Demographic analysis and visualization
#   2. Model training and evaluation on test period
#   3. Future forecasting (3-year ahead)
#   4. Comprehensive performance metrics and diagnostics
#   5. Visualizations for all models
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import logging
import scipy.stats as stats

# Set all text to font size 24 and Times New Roman font
plt.rcParams.update({
    'font.family': 'serif',           # Use serif font family
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],  # Times New Roman as primary
    'font.size': 24,           # Default font size - all text
    'axes.titlesize': 24,      # Title font size
    'axes.labelsize': 24,      # Axis label font size
    'xtick.labelsize': 24,     # X-axis tick label size
    'ytick.labelsize': 24,     # Y-axis tick label size
    'legend.fontsize': 24,     # Legend font size
    'figure.titlesize': 24,    # Figure title font size
    'text.usetex': False       # Don't use LaTeX rendering
})

# Set seaborn font sizes (if used) and Times New Roman font
sns.set(font='Times New Roman', font_scale=1.0)  # Times New Roman font, scale 1.0 (base size)
sns.set_style("whitegrid")

# Suppress TensorFlow logging BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow messages (0=all, 1=info, 2=warnings, 3=errors)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations to suppress messages

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress TensorFlow Python logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
n_test = 2               # Number of years to hold out for testing (chronological split)
future_horizon = 3       # Number of years to forecast into the future
prophet_freq = "YS"      # Frequency string for Prophet: Year Start (annual data)
arima_order = (1, 1, 0)  # ARIMA order: (p, d, q) = (autoregressive, differencing, moving average)
sarima_order = (1, 1, 1) # SARIMA order: (p, d, q) via SARIMAX (non-seasonal)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
cleaned_file = r"C:\Users\joell\OneDrive - Bina Nusantara\SEMESTER 4\FODS\cleaned_dataset.csv"
df_all = pd.read_csv(cleaned_file, low_memory=False)

# Ensure necessary columns exist
if 'Year' not in df_all.columns:
    df_all['Year'] = pd.to_numeric(df_all['Year'], errors='coerce')

if 'Sex_Label' not in df_all.columns:
    raise KeyError("Column 'Sex_Label' not found in cleaned dataset. Please ensure the cleaning step produces a 'Sex_Label' column.")

# Keep only relevant columns and positive incomes
df_all = df_all[['Year','Sex_Label','Personal_Income']].dropna()
df_all = df_all[df_all['Personal_Income'] > 0]

# ============================================================================
# DEMOGRAPHIC ANALYSIS AND EXPLORATORY DATA VISUALIZATION
# ============================================================================

print("\n==============================")
print("📊 DEMOGRAPHIC ANALYSIS")
print("==============================")

# ----------------------------------------------------------------------------
# Section 1: Gender Distribution Analysis
#   - Gender count per year (grouped bar chart)
#   - Gender proportion per year (stacked percentage bar chart)
# ----------------------------------------------------------------------------
gender_count = df_all.groupby(['Year', 'Sex_Label']).size().reset_index(name='Count')
gender_pivot = gender_count.pivot(index='Year', columns='Sex_Label', values='Count').fillna(0)
gender_pivot = gender_pivot.sort_index()

print("\n👥 Gender Count Per Year:")
print(gender_pivot)

# Grouped bar chart
plt.figure(figsize=(12,6))
width = 0.35
years = gender_pivot.index.astype(int)
x = np.arange(len(years))
male_counts = gender_pivot['Male'].values
female_counts = gender_pivot['Female'].values

plt.bar(x - width/2, male_counts, width, label='Male Count')
plt.bar(x + width/2, female_counts, width, label='Female Count')
plt.xticks(x, years, rotation=45, fontsize=24)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Number of Individuals", fontsize=24)
plt.title("Gender Count Per Year (Grouped Bar)", fontsize=24, fontweight='bold')
plt.legend(fontsize=24)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# Stacked percent bar chart
gender_pct = gender_pivot.copy()
gender_pct['Total'] = gender_pct.sum(axis=1)
gender_pct['Male_pct'] = gender_pct['Male'] / gender_pct['Total']
gender_pct['Female_pct'] = gender_pct['Female'] / gender_pct['Total']

plt.figure(figsize=(12,6))
plt.bar(years, gender_pct['Male_pct'], label='Male %')
plt.bar(years, gender_pct['Female_pct'], bottom=gender_pct['Male_pct'], label='Female %')
plt.xticks(years, rotation=45, fontsize=24)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Proportion", fontsize=24)
plt.title("Gender Proportion Per Year", fontsize=24, fontweight='bold')
plt.legend(fontsize=24)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# Section 2: Average Income Analysis by Gender
#   - Average income per gender per year (grouped bar chart)
#   - Average income trends over time (line chart)
# ----------------------------------------------------------------------------
income_pivot = df_all.pivot_table(index='Year', columns='Sex_Label', values='Personal_Income', aggfunc='mean').fillna(0)
income_pivot = income_pivot.sort_index()

print("\n📄 Average Income Per Gender Per Year:")
print(income_pivot.round(3))

# Grouped bar chart for average incomes
plt.figure(figsize=(12,6))
bar_width = 0.35
x = np.arange(len(income_pivot.index))
male_avg = income_pivot['Male'].values
female_avg = income_pivot['Female'].values

plt.bar(x - bar_width/2, male_avg, bar_width, label='Male Avg Income')
plt.bar(x + bar_width/2, female_avg, bar_width, label='Female Avg Income')
plt.xticks(x, income_pivot.index.astype(int), rotation=45, fontsize=24)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Average Income", fontsize=24)
plt.title("Average Income by Gender Per Year", fontsize=24, fontweight='bold')
plt.legend(fontsize=24)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# Also show as line plot for trend clarity
plt.figure(figsize=(12,6))
plt.plot(income_pivot.index, male_avg, marker='o', label='Male Avg Income', markersize=8)
plt.plot(income_pivot.index, female_avg, marker='o', label='Female Avg Income', markersize=8)
plt.title("Average Income by Gender Per Year", fontsize=24, fontweight='bold')
plt.xlabel("Year", fontsize=24)
plt.ylabel("Average Income", fontsize=24)
plt.legend(fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# Section 3: Gender Income Gap Calculation
#   - Compute income gap (Male - Female) per year
#   - Visualize gap as bar chart
# ----------------------------------------------------------------------------
df_pivot = df_avg = df_all.groupby(['Year','Sex_Label'])['Personal_Income'].mean().reset_index().pivot(index='Year', columns='Sex_Label', values='Personal_Income').fillna(0)
if 'Male' not in df_pivot.columns or 'Female' not in df_pivot.columns:
    raise KeyError("Pivot missing 'Male' or 'Female' columns. Check Sex_Label values in data.")

gap_series = (df_pivot['Male'] - df_pivot['Female']).sort_index()
gap_years = gap_series.index.astype(int)
gap_vals = gap_series.values

print("\n📄 Gender Income Gap per Year:")
print(pd.DataFrame({'Year': gap_years, 'Gap': gap_vals}).round(3))

# Line chart for gender income gap
plt.figure(figsize=(12,6))
plt.plot(gap_years, gap_vals, marker='o', linewidth=2, markersize=8, color='tab:orange')
plt.xticks(gap_years, rotation=45, fontsize=24)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Male - Female Income", fontsize=24)
plt.title("Gender Income Gap per Year", fontsize=24, fontweight='bold')
plt.yticks(fontsize=24)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# Column/Bar chart for gender income gap
plt.figure(figsize=(12,6))
plt.bar(gap_years, gap_vals, color='tab:orange', alpha=0.7, width=0.6)
plt.xticks(gap_years, rotation=45, fontsize=24)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Male - Female Income", fontsize=24)
plt.title("Gender Income Gap per Year", fontsize=24, fontweight='bold')
plt.yticks(fontsize=24)
plt.grid(alpha=0.2, axis='y')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# Section 4: Summary Statistics
#   - Comprehensive statistics (count, mean, median, min, max) by gender and year
# ----------------------------------------------------------------------------
summary_stats = df_all.groupby(['Year', 'Sex_Label'])['Personal_Income'].agg(
    ['count','mean','median','min','max']
).reset_index()

print("\n📄 Summary Statistics (Income by Gender per Year):")
print(summary_stats.round(3))

# ----------------------------------------------------------------------------
# Section 5: Statistical Test - Independent Samples t-test
#   - Compare mean income between males and females
# ----------------------------------------------------------------------------
# Extract income data by gender
male_income = df_all[df_all['Sex_Label'] == 'Male']['Personal_Income'].values
female_income = df_all[df_all['Sex_Label'] == 'Female']['Personal_Income'].values

# Calculate overall means
mean_male = np.mean(male_income)
mean_female = np.mean(female_income)
mean_difference = mean_male - mean_female

# Calculate sample sizes and standard deviations for diagnostics
n_male = len(male_income)
n_female = len(female_income)
std_male = np.std(male_income, ddof=1)
std_female = np.std(female_income, ddof=1)

# Perform independent samples t-test (Welch's t-test)
t_stat, p_value = stats.ttest_ind(male_income, female_income, equal_var=False)

# Check if p-value is effectively 0 or underflowed
# If p-value is 0 or nan, calculate it manually using t-distribution
p_value_original = p_value
if p_value == 0.0 or np.isnan(p_value) or p_value < np.finfo(float).tiny:
    # Calculate degrees of freedom for Welch's t-test
    s1_sq = std_male**2
    s2_sq = std_female**2
    df_welch = ((s1_sq/n_male + s2_sq/n_female)**2) / (((s1_sq/n_male)**2)/(n_male-1) + ((s2_sq/n_female)**2)/(n_female-1))
    
    # Calculate p-value using t-distribution (two-tailed)
    try:
        p_value_calc = 2 * (1 - stats.t.cdf(abs(t_stat), df_welch))
        p_value = p_value_calc
    except:
        # If calculation fails, use survival function
        p_value = 2 * stats.t.sf(abs(t_stat), df_welch)

# Format p-value for display with maximum precision
if p_value == 0.0 or p_value < np.finfo(float).tiny:
    # For values that are effectively 0, calculate using log scale
    try:
        s1_sq = std_male**2
        s2_sq = std_female**2
        df_welch = ((s1_sq/n_male + s2_sq/n_female)**2) / (((s1_sq/n_male)**2)/(n_male-1) + ((s2_sq/n_female)**2)/(n_female-1))
        # Use log survival function for very small p-values
        log_sf = stats.t.logsf(abs(t_stat), df_welch)
        log_p = np.log10(2) + log_sf / np.log(10)
        if log_p < -100:
            p_value_str = f"< 10^{log_p:.1f}"
        else:
            p_value_str = f"{10**log_p:.50e}"
    except:
        p_value_str = "< 10^-300 (effectively 0)"
elif p_value < 1e-50:
    # For extremely small values, show in scientific notation with many digits
    p_value_str = f"{p_value:.50e}"
elif p_value < 1e-10:
    p_value_str = f"{p_value:.30e}"  # Scientific notation for very small values
elif p_value < 0.0001:
    p_value_str = f"{p_value:.30f}"  # Many decimal places
else:
    p_value_str = f"{p_value:.20f}"  # Many decimal places

stat_test_table = pd.DataFrame({
    'Statistic': ['Mean Income (Male)', 'Mean Income (Female)', 'Mean Difference', 't-statistic', 'p-value'],
    'Value': [
        f"${mean_male:,.2f}",
        f"${mean_female:,.2f}",
        f"${mean_difference:,.2f}",
        f"{t_stat:.4f}",
        p_value_str
    ]
})

print("\n" + "="*80)
print("📊 STATISTICAL TEST: INDEPENDENT SAMPLES T-TEST")
print("="*80)
print("\nDiagnostic Information:")
print(f"  • Sample size (Male): {n_male:,}")
print(f"  • Sample size (Female): {n_female:,}")
print(f"  • Standard deviation (Male): ${std_male:,.2f}")
print(f"  • Standard deviation (Female): ${std_female:,.2f}")
print(f"  • t-statistic: {t_stat:.10f}")
print(f"  • Raw p-value (before formatting): {p_value}")
print(f"  • Original scipy p-value: {p_value_original}")
print("\nComparison of Mean Income Between Males and Females:")
print("-" * 80)
print(stat_test_table.to_string(index=False))
print("\nInterpretation:")
if p_value < 0.001:
    print(f"  • p-value < 0.001: Highly significant difference (p = {p_value:.20e})")
elif p_value < 0.01:
    print(f"  • p-value < 0.01: Very significant difference (p = {p_value:.20f})")
elif p_value < 0.05:
    print(f"  • p-value < 0.05: Significant difference (p = {p_value:.20f})")
else:
    print(f"  • p-value >= 0.05: No significant difference (p = {p_value:.20f})")
print(f"  • Mean difference: ${mean_difference:,.2f}")
print("="*80)

# ============================================================================
# TIME SERIES DATA PREPARATION FOR FORECASTING
# ============================================================================
# Aggregate data and compute gender income gap time series
df_avg = df_all.groupby(['Year','Sex_Label'])['Personal_Income'].mean().reset_index()
df_pivot = df_avg.pivot(index='Year', columns='Sex_Label', values='Personal_Income')

# If columns missing (Male/Female), raise helpful error
if 'Male' not in df_pivot.columns or 'Female' not in df_pivot.columns:
    raise KeyError("Pivot missing 'Male' or 'Female' columns. Check Sex_Label values in data.")

df_pivot['Gap'] = df_pivot['Male'] - df_pivot['Female']
gap_df = df_pivot.reset_index()[['Year','Gap']].rename(columns={'Year':'ds','Gap':'y'})
gap_df['ds'] = pd.to_datetime(gap_df['ds'], format='%Y')
gap_df = gap_df.sort_values('ds').reset_index(drop=True)

# ============================================================================
# TRAIN/TEST SPLIT (CHRONOLOGICAL)
# ============================================================================
# Split data chronologically: last n_test years for testing, rest for training
if len(gap_df) < (n_test + 1):
    raise ValueError(f"Not enough years of data to create train/test split with n_test={n_test}.")

train_df = gap_df.iloc[:-n_test].copy().reset_index(drop=True)
test_df = gap_df.iloc[-n_test:].copy().reset_index(drop=True)

train_years = train_df['ds'].dt.year.values.reshape(-1,1)
train_values = train_df['y'].values
test_years = test_df['ds'].dt.year.values.reshape(-1,1)
test_values = test_df['y'].values

# Create numeric time indices for models that require them (Linear Regression, XGBoost)
# Index represents sequential time points: 0, 1, 2, ..., T-1
time_index_full = np.arange(len(gap_df))
time_index_train = time_index_full[:len(train_df)]
time_index_test = time_index_full[len(train_df):len(train_df)+n_test]

# ============================================================================
# MODEL 1: PROPHET (FACEBOOK TIME SERIES FORECASTING)
# ============================================================================
# Prophet is designed for business time series with strong seasonal patterns
# Fits on training data, then forecasts test period and future horizon
prophet_model = Prophet(yearly_seasonality=False)
prophet_model.fit(train_df.rename(columns={'ds':'ds','y':'y'}))

total_future_steps = n_test + future_horizon
future_full = prophet_model.make_future_dataframe(periods=total_future_steps, freq=prophet_freq)
forecast_full = prophet_model.predict(future_full)

predicted_prophet = forecast_full['yhat'].iloc[len(train_df): len(train_df)+n_test].values
prophet_future_vals = forecast_full['yhat'].iloc[len(train_df)+n_test: len(train_df)+n_test+future_horizon].values

# ============================================================================
# MODEL 2: ARIMA (AUTOREGRESSIVE INTEGRATED MOVING AVERAGE)
# ============================================================================
# ARIMA models capture autocorrelation in time series data
# Order (1,1,0): 1 autoregressive term, 1 differencing, 0 moving average terms
train_index = pd.DatetimeIndex(train_df['ds'], freq='YS')
train_series = pd.Series(train_values, index=train_index)
arima_model = ARIMA(train_series, order=arima_order)
arima_fit = arima_model.fit()

arima_pred = arima_fit.get_forecast(steps=total_future_steps)
arima_pred_mean = arima_pred.predicted_mean
arima_conf_int = arima_pred.conf_int()

arima_forecast = arima_pred_mean.iloc[:n_test].values
arima_future_vals = arima_pred_mean.iloc[n_test:n_test+future_horizon].values

# ============================================================================
# MODEL 3: SARIMA (SEASONAL AUTOREGRESSIVE INTEGRATED MOVING AVERAGE)
# ============================================================================
# SARIMA extends ARIMA to handle seasonal patterns
# Implemented via SARIMAX with order (1,1,1)
sarima_model = SARIMAX(train_series, order=sarima_order, enforce_stationarity=False, enforce_invertibility=False)
sarima_fit = sarima_model.fit(disp=False)

sarima_pred = sarima_fit.get_forecast(steps=total_future_steps)
sarima_pred_mean = sarima_pred.predicted_mean
sarima_conf_int = sarima_pred.conf_int()

sarima_forecast = sarima_pred_mean.iloc[:n_test].values
sarima_future_vals = sarima_pred_mean.iloc[n_test:n_test+future_horizon].values

# ============================================================================
# MODEL 4: LINEAR REGRESSION
# ============================================================================
# Simple linear trend model: predicts gap as a linear function of time
# Uses numeric time index as feature (0, 1, 2, ...)
lr_model = LinearRegression()
lr_model.fit(time_index_train.reshape(-1,1), train_values)
lr_forecast = lr_model.predict(time_index_test.reshape(-1,1))

# Generate future time indices for forecasting
last_index = time_index_full.max()
future_time_index = np.arange(last_index+1, last_index+1+future_horizon)
lr_future_vals = lr_model.predict(future_time_index.reshape(-1,1))

# ============================================================================
# MODEL 5: XGBOOST (GRADIENT BOOSTING REGRESSOR)
# ============================================================================
# Gradient boosting ensemble method that captures non-linear patterns
# PROPER TIME SERIES APPROACH: Use lag features instead of time index
# This captures temporal dependencies, not just curve fitting

# Create lag features for training data
train_df_lags = pd.DataFrame({
    'y': train_values,
    'lag1': np.concatenate([[np.nan], train_values[:-1]]),
    'lag2': np.concatenate([[np.nan, np.nan], train_values[:-2]]),
    'lag3': np.concatenate([[np.nan, np.nan, np.nan], train_values[:-3]])
})
train_df_lags = train_df_lags.dropna().reset_index(drop=True)

# Prepare features and target
X_xgb_train = train_df_lags[['lag1', 'lag2', 'lag3']].values
y_xgb_train = train_df_lags['y'].values

# Train XGBoost with lag features
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, objective='reg:squarederror', verbosity=0)
xgb_model.fit(X_xgb_train, y_xgb_train)

# Forecast test period: use actual values for lags
xgb_forecast = []
last_vals = train_values[-3:].tolist()  # Last 3 values for lags
for i in range(n_test):
    X_test = np.array([[last_vals[-1], last_vals[-2], last_vals[-3]]])
    pred = xgb_model.predict(X_test)[0]
    xgb_forecast.append(pred)
    # Update lag values: shift and add prediction
    last_vals = last_vals[1:] + [pred]

xgb_forecast = np.array(xgb_forecast)

# Forecast future: continue from test period
xgb_future_vals = []
last_vals = list(train_values[-3:]) + list(test_values)  # Use all historical for initial lags
for i in range(future_horizon):
    X_future = np.array([[last_vals[-1], last_vals[-2], last_vals[-3]]])
    pred = xgb_model.predict(X_future)[0]
    xgb_future_vals.append(pred)
    last_vals = last_vals[1:] + [pred]

xgb_future_vals = np.array(xgb_future_vals)

# ============================================================================
# MODEL 6: LSTM (LONG SHORT-TERM MEMORY NEURAL NETWORK)
# ============================================================================
# Deep learning model that captures long-term dependencies in sequences
# PROPER TIME SERIES APPROACH: Use multiple lags (lag-1, lag-2, lag-3)
# This captures richer temporal patterns than single lag
# Data is normalized (MinMax scaling) before training

# Normalize training data to [0, 1] range for neural network
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_values.reshape(-1,1))

# Create lag sequences: Use 3 lags for richer temporal patterns
# X[t] = [value at t-3, value at t-2, value at t-1], y[t] = value at t
X_lstm = []
y_lstm = []
n_lags = 3  # Use 3 lags
for i in range(n_lags, len(train_scaled)):
    X_lstm.append(train_scaled[i-n_lags:i].flatten())
    y_lstm.append(train_scaled[i])
X_lstm = np.array(X_lstm).reshape(-1, n_lags, 1)  # Shape: (samples, timesteps, features)
y_lstm = np.array(y_lstm)

# Build LSTM architecture: Input -> LSTM(32 units) -> Dense(1 output)
# Input shape: (n_lags, 1) - multiple timesteps, single feature
lstm_model = Sequential([
    Input(shape=(n_lags, 1)),
    LSTM(32, activation='tanh'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

# Train model (200 epochs, verbose=0 for silent training)
lstm_model.fit(X_lstm, y_lstm, epochs=200, verbose=0)

# Iterative forecasting for test period: use last n_lags values as input
lstm_forecast_scaled = []
last_scaled_window = train_scaled[-n_lags:].reshape(1, n_lags, 1)
for _ in range(n_test):
    pred_scaled = lstm_model.predict(last_scaled_window, verbose=0)
    lstm_forecast_scaled.append(pred_scaled.flatten()[0])
    # Update window: shift left, add new prediction
    last_scaled_window = np.concatenate([
        last_scaled_window[:, 1:, :],
        pred_scaled.reshape(1, 1, 1)
    ], axis=1)

# Inverse transform to get predictions in original scale
lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast_scaled).reshape(-1,1)).flatten()

# ============================================================================
# BASELINE MODELS (REQUIRED FOR TIME SERIES EVALUATION)
# ============================================================================
# Naive forecast: uses last observed value as prediction for all future periods
# This is the standard baseline - models must beat this to be useful
naive_forecast = np.repeat(train_values[-1], n_test)
naive_future_vals = np.repeat(gap_df['y'].iloc[-1], future_horizon)

# Seasonal naive: for annual data, same as naive (no seasonality within year)
# If monthly/quarterly data, would use value from same period previous year

# ============================================================================
# MODEL EVALUATION: PERFORMANCE METRICS ON TEST PERIOD
# ============================================================================
# Standard time series metrics: MAE, RMSE, MAPE, MASE
# MASE (Mean Absolute Scaled Error) is preferred for time series as it's scale-free
# and compares against naive forecast

def safe_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE) safely.
    Handles division by zero by masking out near-zero values.
    Returns percentage (0-100).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-8  # Avoid division by zero
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def mase(y_true, y_pred, y_train):
    """
    Mean Absolute Scaled Error (MASE) - preferred metric for time series.
    Compares forecast errors to naive forecast errors.
    MASE < 1: better than naive; MASE > 1: worse than naive
    """
    mae_forecast = np.mean(np.abs(y_true - y_pred))
    # Naive forecast error (mean absolute error of one-step naive forecast)
    naive_error = np.mean(np.abs(np.diff(y_train)))
    if naive_error == 0:
        return np.nan
    return mae_forecast / naive_error

# Ensure all predictions and actuals are 1-D arrays for consistent calculations
prophet_pred_test = np.array(predicted_prophet).ravel()
arima_pred_test = np.array(arima_forecast).ravel()
sarima_pred_test = np.array(sarima_forecast).ravel()
lr_pred_test = np.array(lr_forecast).ravel()
xgb_pred_test = np.array(xgb_forecast).ravel()
lstm_pred_test = np.array(lstm_forecast).ravel()
test_values = np.array(test_values).ravel()

models_info = {
    "Naive": naive_forecast,  # Baseline model - must include
    "Prophet": prophet_pred_test,
    "ARIMA": arima_pred_test,
    "SARIMA": sarima_pred_test,
    "Linear Regression": lr_pred_test,
    "XGBoost": xgb_pred_test,
    "LSTM": lstm_pred_test
}

# Extract model diagnostics for statistical models (AIC/BIC)
# AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion)
# are used for model selection: lower values indicate better fit
arima_aic = arima_fit.aic
arima_bic = arima_fit.bic
sarima_aic = sarima_fit.aic
sarima_bic = sarima_fit.bic

# Build comprehensive metrics table with standard time series metrics
metrics_extended = []
for name, pred in models_info.items():
    mae = mean_absolute_error(test_values, pred)
    rmse = np.sqrt(mean_squared_error(test_values, pred))
    mape_v = safe_mape(test_values, pred)
    mase_v = mase(test_values, pred, train_values)
    
    row = {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape_v,
        "MASE": mase_v
    }
    
    # Add AIC/BIC diagnostics for statistical time series models (ARIMA/SARIMA)
    if name == "ARIMA":
        row["AIC"] = arima_aic
        row["BIC"] = arima_bic
    elif name == "SARIMA":
        row["AIC"] = sarima_aic
        row["BIC"] = sarima_bic
    else:
        row["AIC"] = np.nan
        row["BIC"] = np.nan
    
    metrics_extended.append(row)

metrics_df_extended = pd.DataFrame(metrics_extended).set_index("Model")

print("\n" + "="*80)
print("📊 TIME SERIES MODEL PERFORMANCE EVALUATION (TEST PERIOD)")
print("="*80)
print(f"\nTest Period: {test_df['ds'].dt.year.min()} - {test_df['ds'].dt.year.max()} ({n_test} years)")
print(f"Training Period: {train_df['ds'].dt.year.min()} - {train_df['ds'].dt.year.max()} ({len(train_df)} years)")
print("\nForecast Accuracy Metrics:")
print("-" * 80)
print(metrics_df_extended[["MAE", "RMSE", "MAPE (%)", "MASE"]].round(3))
print("\nMASE Interpretation: < 1.0 = better than naive baseline, > 1.0 = worse than naive")
print("-" * 80)
print("\nModel Diagnostics (AIC/BIC - Lower is Better):")
print("-" * 80)
print(metrics_df_extended[["AIC", "BIC"]].round(2))

# Rank models by MASE (lower is better) - standard for time series
metrics_df_extended_sorted = metrics_df_extended.sort_values("MASE", ascending=True)
print("\nModel Ranking (by MASE - lower is better):")
print("-" * 80)
for idx, (model, row) in enumerate(metrics_df_extended_sorted.iterrows(), 1):
    mase_val = row['MASE']
    better_than_naive = "✓" if mase_val < 1.0 else "✗"
    print(f"{idx}. {model:20s} - MASE: {mase_val:.3f} {better_than_naive} | MAE: {row['MAE']:.2f} | RMSE: {row['RMSE']:.2f}")

# Store MASE for later use
results_mase = {model: metrics_df_extended.loc[model, "MASE"] for model in metrics_df_extended.index}

# ============================================================================
# RESIDUAL DIAGNOSTICS (REQUIRED FOR STATISTICAL MODELS)
# ============================================================================
# Check if residuals are white noise - core requirement for ARIMA/SARIMA models
# White noise residuals indicate the model has captured all predictable patterns

print("\n" + "="*80)
print("🔍 RESIDUAL DIAGNOSTICS")
print("="*80)

# ARIMA Residual Diagnostics
resid_arima = arima_fit.resid
print("\nARIMA Residual Diagnostics:")
print("-" * 80)
print(f"Mean: {resid_arima.mean():.4f} (should be ~0)")
print(f"Std Dev: {resid_arima.std():.4f}")

# Ljung-Box test for residual autocorrelation
# Null hypothesis: residuals are independently distributed (white noise)
# Use appropriate lags based on sample size (max lag should be < n/4)
max_lag = min(10, len(resid_arima) // 4) if len(resid_arima) > 4 else len(resid_arima) - 1
test_lags = [min(5, max_lag), max_lag] if max_lag >= 5 else [max_lag]
test_lags = list(set(test_lags))  # Remove duplicates
if len(test_lags) > 0 and max(test_lags) > 0:
    ljung_box_arima = acorr_ljungbox(resid_arima, lags=test_lags, return_df=True)
    print("\nLjung-Box Test (H0: residuals are white noise):")
    print(ljung_box_arima.round(4))
    print("p-value > 0.05 indicates white noise residuals ✓")
else:
    print("\nLjung-Box Test: Insufficient data for statistical test")

# Plot ARIMA residuals
plt.figure(figsize=(12, 6))
plt.plot(resid_arima.index, resid_arima.values)
plt.axhline(0, color='red', linestyle='--')
plt.title("ARIMA Residuals Over Time", fontsize=24, fontweight='bold')
plt.xlabel("Time", fontsize=24)
plt.ylabel("Residuals", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Plot ARIMA Residual ACF
plt.figure(figsize=(12, 6))
plot_acf(resid_arima, lags=min(10, len(resid_arima)//2), ax=plt.gca())
plt.title("ARIMA Residual ACF", fontsize=24, fontweight='bold')
plt.xlabel("Lag", fontsize=24)
plt.ylabel("ACF", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()
plt.show()

# SARIMA Residual Diagnostics
resid_sarima = sarima_fit.resid
print("\nSARIMA Residual Diagnostics:")
print("-" * 80)
print(f"Mean: {resid_sarima.mean():.4f} (should be ~0)")
print(f"Std Dev: {resid_sarima.std():.4f}")

# Use appropriate lags based on sample size
max_lag = min(10, len(resid_sarima) // 4) if len(resid_sarima) > 4 else len(resid_sarima) - 1
test_lags = [min(5, max_lag), max_lag] if max_lag >= 5 else [max_lag]
test_lags = list(set(test_lags))  # Remove duplicates
if len(test_lags) > 0 and max(test_lags) > 0:
    ljung_box_sarima = acorr_ljungbox(resid_sarima, lags=test_lags, return_df=True)
    print("\nLjung-Box Test (H0: residuals are white noise):")
    print(ljung_box_sarima.round(4))
    print("p-value > 0.05 indicates white noise residuals ✓")
else:
    print("\nLjung-Box Test: Insufficient data for statistical test")

# Plot SARIMA residuals
plt.figure(figsize=(12, 6))
plt.plot(resid_sarima.index, resid_sarima.values)
plt.axhline(0, color='red', linestyle='--')
plt.title("SARIMA Residuals Over Time", fontsize=24, fontweight='bold')
plt.xlabel("Time", fontsize=24)
plt.ylabel("Residuals", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Plot SARIMA Residual ACF
plt.figure(figsize=(12, 6))
plot_acf(resid_sarima, lags=min(10, len(resid_sarima)//2), ax=plt.gca())
plt.title("SARIMA Residual ACF", fontsize=24, fontweight='bold')
plt.xlabel("Lag", fontsize=24)
plt.ylabel("ACF", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()
plt.show()

# Prophet Residual Diagnostics
# Prophet residuals = actual - predicted for training period
prophet_train_pred = forecast_full['yhat'].iloc[:len(train_df)].values
prophet_resid = train_values - prophet_train_pred
prophet_resid_series = pd.Series(prophet_resid, index=train_df['ds'])

print("\nProphet Residual Diagnostics:")
print("-" * 80)
print(f"Mean: {prophet_resid_series.mean():.4f} (should be ~0)")
print(f"Std Dev: {prophet_resid_series.std():.4f}")

# Use appropriate lags based on sample size
max_lag = min(10, len(prophet_resid_series) // 4) if len(prophet_resid_series) > 4 else len(prophet_resid_series) - 1
test_lags = [min(5, max_lag), max_lag] if max_lag >= 5 else [max_lag]
test_lags = list(set(test_lags))  # Remove duplicates
if len(test_lags) > 0 and max(test_lags) > 0:
    ljung_box_prophet = acorr_ljungbox(prophet_resid_series, lags=test_lags, return_df=True)
    print("\nLjung-Box Test (H0: residuals are white noise):")
    print(ljung_box_prophet.round(4))
    print("p-value > 0.05 indicates white noise residuals ✓")
else:
    print("\nLjung-Box Test: Insufficient data for statistical test")

# Plot Prophet residuals
plt.figure(figsize=(12, 6))
plt.plot(prophet_resid_series.index, prophet_resid_series.values)
plt.axhline(0, color='red', linestyle='--')
plt.title("Prophet Residuals Over Time", fontsize=24, fontweight='bold')
plt.xlabel("Time", fontsize=24)
plt.ylabel("Residuals", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Plot Prophet Residual ACF
plt.figure(figsize=(12, 6))
plot_acf(prophet_resid_series, lags=min(10, len(prophet_resid_series)//2), ax=plt.gca())
plt.title("Prophet Residual ACF", fontsize=24, fontweight='bold')
plt.xlabel("Lag", fontsize=24)
plt.ylabel("ACF", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()
plt.show()

# Linear Regression Residual Diagnostics
# Linear Regression residuals = actual - fitted for training period
lr_train_pred = lr_model.predict(time_index_train.reshape(-1,1))
lr_resid = train_values - lr_train_pred
lr_resid_series = pd.Series(lr_resid, index=train_df['ds'])

print("\nLinear Regression Residual Diagnostics:")
print("-" * 80)
print(f"Mean: {lr_resid_series.mean():.4f} (should be ~0)")
print(f"Std Dev: {lr_resid_series.std():.4f}")

# Use appropriate lags based on sample size
max_lag = min(10, len(lr_resid_series) // 4) if len(lr_resid_series) > 4 else len(lr_resid_series) - 1
test_lags = [min(5, max_lag), max_lag] if max_lag >= 5 else [max_lag]
test_lags = list(set(test_lags))  # Remove duplicates
if len(test_lags) > 0 and max(test_lags) > 0:
    ljung_box_lr = acorr_ljungbox(lr_resid_series, lags=test_lags, return_df=True)
    print("\nLjung-Box Test (H0: residuals are white noise):")
    print(ljung_box_lr.round(4))
    print("p-value > 0.05 indicates white noise residuals ✓")
else:
    print("\nLjung-Box Test: Insufficient data for statistical test")

# Plot Linear Regression residuals
plt.figure(figsize=(12, 6))
plt.plot(lr_resid_series.index, lr_resid_series.values)
plt.axhline(0, color='red', linestyle='--')
plt.title("Linear Regression Residuals Over Time", fontsize=24, fontweight='bold')
plt.xlabel("Time", fontsize=24)
plt.ylabel("Residuals", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Plot Linear Regression Residual ACF
plt.figure(figsize=(12, 6))
plot_acf(lr_resid_series, lags=min(10, len(lr_resid_series)//2), ax=plt.gca())
plt.title("Linear Regression Residual ACF", fontsize=24, fontweight='bold')
plt.xlabel("Lag", fontsize=24)
plt.ylabel("ACF", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()
plt.show()

# ============================================================================
# TEST PERIOD COMPARISON: ACTUAL vs PREDICTED VALUES
# ============================================================================
# Create detailed comparison table showing actual values, predictions, and errors
comparison_df = pd.DataFrame({
    "Year": test_df['ds'].dt.year,
    "Actual_Gap": test_values,
    "Naive": naive_forecast,  # Include baseline
    "Prophet": prophet_pred_test,
    "ARIMA": arima_pred_test,
    "SARIMA": sarima_pred_test,
    "Linear Regression": lr_pred_test,
    "XGBoost": xgb_pred_test,
    "LSTM": lstm_pred_test
})
# Calculate prediction errors (residuals) for each model
# Error = Actual - Predicted (positive error means model underestimates)
for m in ["Naive", "Prophet","ARIMA","SARIMA","Linear Regression","XGBoost","LSTM"]:
    comparison_df[f"{m}_Error"] = comparison_df["Actual_Gap"] - comparison_df[m]

print("\n" + "="*80)
print("📋 TEST PERIOD: ACTUAL vs PREDICTED VALUES")
print("="*80)
print(comparison_df[["Year", "Actual_Gap", "Naive", "Prophet", "ARIMA", "SARIMA", "Linear Regression", "XGBoost", "LSTM"]].round(2))
print("\nPrediction Errors (Actual - Predicted):")
print("-" * 80)
error_cols = [col for col in comparison_df.columns if col.endswith("_Error")]
print(comparison_df[["Year"] + error_cols].round(2))

# ============================================================================
# FUTURE FORECASTING: REFIT MODELS ON FULL DATASET
# ============================================================================
# Retrain all models on the complete historical dataset (train + test)
# to generate 3-year ahead forecasts. This uses all available information
# for more accurate future predictions.

last_actual = gap_df['y'].iloc[-1]  # Last observed value for percentage change calculation

# --- Model 1: Prophet (Full Dataset) ---
prophet_full = Prophet(yearly_seasonality=False)
prophet_full.fit(gap_df.rename(columns={'ds':'ds','y':'y'}))
future_full_final = prophet_full.make_future_dataframe(periods=future_horizon, freq=prophet_freq)
forecast_full_final = prophet_full.predict(future_full_final)
prophet_full_vals = forecast_full_final['yhat'].iloc[-future_horizon:].values

# --- Model 2: ARIMA (Full Dataset) ---
full_index = pd.DatetimeIndex(gap_df['ds'], freq='YS')
full_series = pd.Series(gap_df['y'].values, index=full_index)
arima_full = ARIMA(full_series, order=arima_order).fit()
arima_full_fitted = arima_full.fittedvalues.values  # In-sample fitted values for entire historical period
arima_full_pred = arima_full.get_forecast(steps=future_horizon)
arima_full_vals = arima_full_pred.predicted_mean.values

# --- Model 3: SARIMA (Full Dataset) ---
sarima_full = SARIMAX(full_series, order=sarima_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
sarima_full_fitted = sarima_full.fittedvalues.values  # In-sample fitted values for entire historical period
sarima_full_pred = sarima_full.get_forecast(steps=future_horizon)
sarima_full_vals = sarima_full_pred.predicted_mean.values

# --- Model 4: Linear Regression (Full Dataset) ---
full_time_index = np.arange(len(gap_df))
lr_full = LinearRegression().fit(full_time_index.reshape(-1,1), gap_df['y'].values)
future_time_index = np.arange(full_time_index.max()+1, full_time_index.max()+1+future_horizon)
lr_full_vals = lr_full.predict(future_time_index.reshape(-1,1))

# Calculate formal prediction intervals for Linear Regression
# Using t-distribution for proper statistical inference
X_full = full_time_index.reshape(-1,1)
y_full = gap_df['y'].values
n = len(y_full)
p = 2  # Number of parameters: intercept + slope

# Calculate fitted values and residuals
y_fitted = lr_full.predict(X_full)
residuals = y_full - y_fitted
residual_std = np.std(residuals, ddof=p)  # Standard error with degrees of freedom adjustment

# Calculate 95% prediction intervals using t-distribution
t_critical = stats.t.ppf(0.975, df=n-p)  # 95% confidence, two-tailed
lr_lower = lr_full_vals - t_critical * residual_std
lr_upper = lr_full_vals + t_critical * residual_std

# --- Model 5: XGBoost (Full Dataset) ---
# Create lag features for full dataset
full_df_lags = pd.DataFrame({
    'y': gap_df['y'].values,
    'lag1': np.concatenate([[np.nan], gap_df['y'].values[:-1]]),
    'lag2': np.concatenate([[np.nan, np.nan], gap_df['y'].values[:-2]]),
    'lag3': np.concatenate([[np.nan, np.nan, np.nan], gap_df['y'].values[:-3]])
})
full_df_lags = full_df_lags.dropna().reset_index(drop=True)

X_xgb_full = full_df_lags[['lag1', 'lag2', 'lag3']].values
y_xgb_full = full_df_lags['y'].values

xgb_full = XGBRegressor(n_estimators=200, learning_rate=0.05, objective='reg:squarederror', verbosity=0)
xgb_full.fit(X_xgb_full, y_xgb_full)

# Forecast future: use last 3 values from full dataset
xgb_full_vals = []
last_vals = list(gap_df['y'].values[-3:])
for i in range(future_horizon):
    X_future = np.array([[last_vals[-1], last_vals[-2], last_vals[-3]]])
    pred = xgb_full.predict(X_future)[0]
    xgb_full_vals.append(pred)
    last_vals = last_vals[1:] + [pred]

xgb_full_vals = np.array(xgb_full_vals)

# --- Model 6: LSTM (Full Dataset) ---
# Retrain LSTM on full dataset, then forecast iteratively
scaler_full = MinMaxScaler()
full_scaled = scaler_full.fit_transform(gap_df['y'].values.reshape(-1,1))

# Create lag sequences from full dataset (using n_lags = 3)
X_full = []
y_full = []
for i in range(n_lags, len(full_scaled)):
    X_full.append(full_scaled[i-n_lags:i].flatten())
    y_full.append(full_scaled[i])
X_full = np.array(X_full).reshape(-1, n_lags, 1)
y_full = np.array(y_full)

# Build and train LSTM model
lstm_full = Sequential([
    Input(shape=(n_lags, 1)),
    LSTM(32, activation='tanh'),
    Dense(1)
])
lstm_full.compile(optimizer='adam', loss='mse')
lstm_full.fit(X_full, y_full, epochs=200, verbose=0)

# Iterative forecasting: use last n_lags values as input for next step
lstm_full_preds_scaled = []
last_window = full_scaled[-n_lags:].reshape(1, n_lags, 1)
for _ in range(future_horizon):
    p = lstm_full.predict(last_window, verbose=0)
    lstm_full_preds_scaled.append(p.flatten()[0])
    # Update window: shift left, add new prediction
    last_window = np.concatenate([
        last_window[:, 1:, :],
        p.reshape(1, 1, 1)
    ], axis=1)
lstm_full_vals = scaler_full.inverse_transform(np.array(lstm_full_preds_scaled).reshape(-1,1)).flatten()

# Assemble future forecasts dataframe with prediction intervals
future_years_list = list(np.arange(gap_df['ds'].dt.year.max()+1, gap_df['ds'].dt.year.max()+1+future_horizon))

# Get prediction intervals for models that provide them
# ARIMA confidence intervals
arima_conf_int = arima_full_pred.conf_int()
arima_lower = arima_conf_int.iloc[:, 0].values
arima_upper = arima_conf_int.iloc[:, 1].values

# SARIMA confidence intervals
sarima_conf_int = sarima_full_pred.conf_int()
sarima_lower = sarima_conf_int.iloc[:, 0].values
sarima_upper = sarima_conf_int.iloc[:, 1].values

# Prophet confidence intervals (80% and 95%)
prophet_lower = forecast_full_final['yhat_lower'].iloc[-future_horizon:].values
prophet_upper = forecast_full_final['yhat_upper'].iloc[-future_horizon:].values

future_df = pd.DataFrame({
    "Year": future_years_list,
    "Naive_FC": naive_future_vals,  # Include baseline
    "Prophet_FC": prophet_full_vals,
    "Prophet_Lower": prophet_lower,
    "Prophet_Upper": prophet_upper,
    "ARIMA_FC": arima_full_vals,
    "ARIMA_Lower": arima_lower,
    "ARIMA_Upper": arima_upper,
    "SARIMA_FC": sarima_full_vals,
    "SARIMA_Lower": sarima_lower,
    "SARIMA_Upper": sarima_upper,
    "LR_FC": lr_full_vals,
    "LR_Lower": lr_lower,
    "LR_Upper": lr_upper,
    "XGB_FC": xgb_full_vals,
    "LSTM_FC": lstm_full_vals
})

# Calculate percentage change relative to last observed value
# This shows how much each forecast deviates from the most recent actual value
for col in ["Naive_FC","Prophet_FC","ARIMA_FC","SARIMA_FC","LR_FC","XGB_FC","LSTM_FC"]:
    future_df[col + "_%Diff"] = 100.0 * (future_df[col] - last_actual) / np.where(np.abs(last_actual) > 1e-8, last_actual, np.nan)

print("\n" + "="*80)
print("🔮 FUTURE FORECASTS (3-Year Ahead)")
print("="*80)
print(f"Last Observed Value (Year {gap_df['ds'].dt.year.max()}): {last_actual:.2f}")
print("\nForecast Values:")
print("-" * 80)
forecast_display = future_df[["Year", "Naive_FC", "Prophet_FC", "ARIMA_FC", "SARIMA_FC", "LR_FC", "XGB_FC", "LSTM_FC"]].copy()
forecast_display.columns = ["Year", "Naive", "Prophet", "ARIMA", "SARIMA", "Linear Regression", "XGBoost", "LSTM"]
print(forecast_display.round(2))
print("\nPercentage Change from Last Observed Value:")
print("-" * 80)
pct_cols = [col for col in future_df.columns if col.endswith("_%Diff")]
pct_display = future_df[["Year"] + pct_cols].copy()
pct_display.columns = ["Year"] + [col.replace("_FC_%Diff", "") for col in pct_cols]
print(pct_display.round(2))

# Display prediction intervals for models that provide them
print("\nPrediction Intervals (95% for ARIMA/SARIMA/Linear Regression, 80% for Prophet):")
print("-" * 80)
interval_cols = ["Year", "Prophet_FC", "Prophet_Lower", "Prophet_Upper", 
                 "ARIMA_FC", "ARIMA_Lower", "ARIMA_Upper",
                 "SARIMA_FC", "SARIMA_Lower", "SARIMA_Upper",
                 "LR_FC", "LR_Lower", "LR_Upper"]
print(future_df[interval_cols].round(2))

# ============================================================================
# TABLE 4: LINEAR REGRESSION FORECAST VALUES WITH PREDICTION INTERVALS
# ============================================================================
# Formal statistical prediction intervals using t-distribution
print("\n" + "="*80)
print("📊 TABLE 4: LINEAR REGRESSION FORECAST VALUES WITH 95% PREDICTION INTERVALS")
print("="*80)
print("Formal prediction intervals calculated using t-distribution")
print(f"Residual Standard Error: {residual_std:.2f}")
print(f"Degrees of Freedom: {n-p}")
print(f"t-critical (95%): {t_critical:.4f}")
print("-" * 80)

table4 = pd.DataFrame({
    'Year': future_years_list,
    'Forecasted Income Gap': lr_full_vals,
    'Lower Bound (95% CI)': lr_lower,
    'Upper Bound (95% CI)': lr_upper
})

print(table4.round(2))
print("\nNote: Prediction intervals account for both parameter uncertainty and residual variability")
print("="*80)

# ============================================================================
# VISUALIZATION: GENDER INCOME COMPARISON
# ============================================================================
# Create visualizations comparing male and female average incomes over time
df_gender_plot = df_pivot.reset_index()[['Year','Male','Female']].sort_values('Year')
# Note: gender counts were computed earlier in demographic analysis section
counts_df = gender_pivot.reset_index().rename(columns={'index':'Year'})

# Grouped bar chart: Average income by gender per year
plt.figure(figsize=(12,6))
width = 0.35
years = df_gender_plot['Year'].astype(int).values
x = np.arange(len(years))
plt.bar(x - width/2, df_gender_plot['Male'].values, width, label='Male Avg Income')
plt.bar(x + width/2, df_gender_plot['Female'].values, width, label='Female Avg Income')
plt.xticks(x, years, rotation=45, fontsize=24)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Average Personal Income", fontsize=24)
plt.title("Average Income by Gender per Year (Grouped Bar)", fontsize=24, fontweight='bold')
plt.legend(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# Combined visualization: Average income (bars) and sample counts (lines) on dual axes
plt.figure(figsize=(12,6))
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.bar(x - width/2, df_gender_plot['Male'].values, width, label='Male Avg Income')
ax1.bar(x + width/2, df_gender_plot['Female'].values, width, label='Female Avg Income')
# plot counts as lines
male_counts_aligned = gender_pivot.loc[df_gender_plot['Year'], 'Male'].values
female_counts_aligned = gender_pivot.loc[df_gender_plot['Year'], 'Female'].values
ax2.plot(x, male_counts_aligned, color='black', marker='o', linestyle='--', label='Male Count')
ax2.plot(x, female_counts_aligned, color='gray', marker='o', linestyle='--', label='Female Count')
ax1.set_xticks(x); ax1.set_xticklabels(years, rotation=45, fontsize=24)
ax1.set_xlabel("Year", fontsize=24)
ax1.set_ylabel("Average Income", fontsize=24)
ax2.set_ylabel("Counts", fontsize=24)
ax1.set_title("Avg Income (bars) & Counts (lines) by Gender per Year", fontsize=24, fontweight='bold')
ax1.tick_params(axis='y', labelsize=24)
ax2.tick_params(axis='y', labelsize=24)
ax1.legend(loc='upper left', fontsize=24); ax2.legend(loc='upper right', fontsize=24)
plt.tight_layout()
plt.show()

print("\n📊 Table: Male vs Female Income Per Year")
print(df_gender_plot.round(3))

# ============================================================================
# INDIVIDUAL MODEL VISUALIZATIONS
# ============================================================================
# Create separate plots for each model showing:
#   - Training period: actual values (black)
#   - Test period: actual values (green) and predictions (dashed)
#   - Future period: forecasts (dotted) with shaded forecast horizon

def plot_model(train_df, test_df, train_vals, test_vals, test_pred, future_x, future_pred, title, color):
    """
    Plot time series with train/test/future periods clearly distinguished.
    
    Parameters:
    - train_df, test_df: DataFrames with 'ds' column (datetime)
    - train_vals, test_vals: Actual values for train and test periods
    - test_pred: Model predictions for test period
    - future_x: Future years array
    - future_pred: Model forecasts for future period
    - title: Plot title
    - color: Color for model predictions/forecasts
    """
    plt.figure(figsize=(12,5))
    plt.plot(train_df['ds'].dt.year, train_vals, marker='o', color='black', label='Train (Actual)', markersize=8)
    plt.plot(test_df['ds'].dt.year, test_vals, marker='o', color='green', label='Test (Actual)', markersize=8)
    plt.plot(test_df['ds'].dt.year, test_pred, linestyle='--', marker='x', color=color, label='Test Pred', markersize=8)
    plt.plot(future_x, future_pred, linestyle=':', marker='x', color=color, label='Future Forecast', markersize=8)
    
    # Shade the forecast horizon area for visual clarity
    future_start = test_df['ds'].dt.year.max() + 0.2
    future_end = future_x.max() + 0.8
    plt.axvspan(future_start, future_end, color='orange', alpha=0.12)
    
    plt.xlabel("Year", fontsize=24)
    plt.ylabel("Gender Income Gap (Male - Female)", fontsize=24)
    plt.title(title, fontsize=24, fontweight='bold')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.show()

future_years_arr = np.array(future_years_list)

# Prophet plot
plot_model(train_df, test_df, train_values, test_values, prophet_pred_test, future_years_arr, prophet_full_vals,
           "Prophet – Actual vs Test Predictions vs 3-Year Forecast", "tab:blue")

# ARIMA plot
plot_model(train_df, test_df, train_values, test_values, arima_pred_test, future_years_arr, arima_full_vals,
           "ARIMA – Actual vs Test Predictions vs 3-Year Forecast", "tab:red")

# SARIMA plot
plot_model(train_df, test_df, train_values, test_values, sarima_pred_test, future_years_arr, sarima_full_vals,
           "SARIMA – Actual vs Test Predictions vs 3-Year Forecast", "tab:orange")

# Linear Regression plot
plot_model(train_df, test_df, train_values, test_values, lr_pred_test, future_years_arr, lr_full_vals,
           "Linear Regression – Actual vs Test Predictions vs 3-Year Forecast", "tab:purple")

# XGBoost plot
plot_model(train_df, test_df, train_values, test_values, xgb_pred_test, future_years_arr, xgb_full_vals,
           "XGBoost – Actual vs Test Predictions vs 3-Year Forecast", "tab:green")

# LSTM plot
plot_model(train_df, test_df, train_values, test_values, lstm_pred_test, future_years_arr, lstm_full_vals,
           "LSTM – Actual vs Test Predictions vs 3-Year Forecast", "tab:brown")

# ============================================================================
# COMBINED MODEL COMPARISON VISUALIZATION
# ============================================================================
# Single plot showing all models together for easy comparison
plt.figure(figsize=(14,7))

# ============================================================================
# SIMPLIFIED COMPARISON: ACTUAL vs BEST MODEL vs NAIVE BASELINE
# ============================================================================
# Find best model (lowest MASE)
best_model_name = metrics_df_extended_sorted.index[0]
best_model_test = models_info[best_model_name]
best_model_future = {
    "Naive": naive_future_vals,
    "Prophet": prophet_full_vals,
    "ARIMA": arima_full_vals,
    "SARIMA": sarima_full_vals,
    "Linear Regression": lr_full_vals,
    "XGBoost": xgb_full_vals,
    "LSTM": lstm_full_vals
}[best_model_name]

plt.figure(figsize=(14, 7))
# Actual values
plt.plot(train_df['ds'].dt.year, train_values, marker='o', color='black', linewidth=2, label='Train (Actual)', markersize=8)
plt.plot(test_df['ds'].dt.year, test_values, marker='o', color='green', linewidth=2, label='Test (Actual)', markersize=8)

# Best model
plt.plot(test_df['ds'].dt.year, best_model_test, linestyle='--', marker='s', color='tab:blue', linewidth=2, label=f'{best_model_name} (Test)', markersize=10)
plt.plot(future_years_arr, best_model_future, linestyle=':', marker='s', color='tab:blue', linewidth=2, label=f'{best_model_name} (Future)', markersize=10)

# Naive baseline
plt.plot(test_df['ds'].dt.year, naive_forecast, linestyle='--', marker='x', color='gray', linewidth=1.5, label='Naive (Test)', markersize=8)
plt.plot(future_years_arr, naive_future_vals, linestyle=':', marker='x', color='gray', linewidth=1.5, label='Naive (Future)', markersize=8)

# Add prediction intervals for best model if available
if best_model_name == "Prophet":
    plt.fill_between(future_years_arr, prophet_lower, prophet_upper, alpha=0.2, color='tab:blue', label='Prophet 80% CI')
elif best_model_name == "ARIMA":
    plt.fill_between(future_years_arr, arima_lower, arima_upper, alpha=0.2, color='tab:blue', label='ARIMA 95% CI')
elif best_model_name == "SARIMA":
    plt.fill_between(future_years_arr, sarima_lower, sarima_upper, alpha=0.2, color='tab:blue', label='SARIMA 95% CI')

# Shade forecast horizon
future_start = test_df['ds'].dt.year.max() + 0.2
future_end = future_years_arr.max() + 0.8
plt.axvspan(future_start, future_end, color='orange', alpha=0.1, label='Forecast Horizon')

plt.xlabel("Year", fontsize=24)
plt.ylabel("Gender Income Gap (Male - Female)", fontsize=24)
plt.title(f"Gender Income Gap: Actual vs Best Model ({best_model_name}) vs Naive Baseline", fontsize=24, fontweight='bold')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(alpha=0.3, linestyle='--')
plt.legend(loc='best', fontsize=24)
plt.tight_layout()
plt.show()

# ============================================================================
# COMPREHENSIVE TIME SERIES TABLE: TRAIN + TEST + FUTURE
# ============================================================================
# Assemble complete time series with actual values and all model predictions
# for the entire period (training, test, and future)

# Combine all years: training + test + future
years_full = list(train_df['ds'].dt.year) + list(test_df['ds'].dt.year) + future_years_list
actual_full = list(train_values) + list(test_values) + [np.nan] * future_horizon

# --- Prophet: Use predictions from train model for train+test, full model for future ---
prophet_full_series = list(forecast_full['yhat'].iloc[:len(train_df) + n_test]) + list(prophet_full_vals)

# --- ARIMA: Use full model fitted values for historical period, forecasts for future ---
# Note: ARIMA with d=1 loses the first observation due to differencing
# fittedvalues is a Series - check if first index matches first data point
arima_fitted_series = arima_full.fittedvalues
if len(arima_fitted_series) < len(gap_df) or arima_fitted_series.index[0] != gap_df['ds'].iloc[0]:
    # First value is missing due to differencing, use actual value
    arima_full_series = [gap_df['y'].iloc[0]] + list(arima_fitted_series.values) + list(arima_full_vals)
else:
    # Check if first fitted value is 0 or NaN, replace with actual if so
    if pd.isna(arima_fitted_series.iloc[0]) or arima_fitted_series.iloc[0] == 0:
        arima_full_series = [gap_df['y'].iloc[0]] + list(arima_fitted_series.values[1:]) + list(arima_full_vals)
    else:
        arima_full_series = list(arima_fitted_series.values) + list(arima_full_vals)

# --- SARIMA: Use full model fitted values for historical period, forecasts for future ---
# Note: SARIMA with d=1 loses the first observation due to differencing
# fittedvalues is a Series - check if first index matches first data point
sarima_fitted_series = sarima_full.fittedvalues
if len(sarima_fitted_series) < len(gap_df) or sarima_fitted_series.index[0] != gap_df['ds'].iloc[0]:
    # First value is missing due to differencing, use actual value
    sarima_full_series = [gap_df['y'].iloc[0]] + list(sarima_fitted_series.values) + list(sarima_full_vals)
else:
    # Check if first fitted value is 0 or NaN, replace with actual if so
    if pd.isna(sarima_fitted_series.iloc[0]) or sarima_fitted_series.iloc[0] == 0:
        sarima_full_series = [gap_df['y'].iloc[0]] + list(sarima_fitted_series.values[1:]) + list(sarima_full_vals)
    else:
        sarima_full_series = list(sarima_fitted_series.values) + list(sarima_full_vals)

# --- Linear Regression: Predict on full historical period, then forecast future ---
lr_full_series = list(lr_full.predict(full_time_index.reshape(-1, 1))) + list(lr_full_vals)

# --- XGBoost: Predict on full historical period using lag features, then forecast future ---
# For historical period, use actual values to create lag features
xgb_full_preds_existing = []
for i in range(n_lags, len(gap_df)):
    X_hist = np.array([[gap_df['y'].iloc[i-1], gap_df['y'].iloc[i-2], gap_df['y'].iloc[i-3]]])
    pred = xgb_full.predict(X_hist)[0]
    xgb_full_preds_existing.append(pred)
# First n_lags values use actual (can't predict without lags)
xgb_full_series = list(gap_df['y'].iloc[:n_lags].values) + xgb_full_preds_existing + list(xgb_full_vals)

# --- LSTM: Compute fitted predictions for historical period (using lag window) ---
# Note: LSTM needs n_lags previous values to predict, so first n_lags values use actual
lstm_preds_existing = []
for i in range(n_lags, len(full_scaled)):
    window = full_scaled[i-n_lags:i].reshape(1, n_lags, 1)
    pred = lstm_full.predict(window, verbose=0)
    lstm_preds_existing.append(pred.flatten()[0])
lstm_fitted_values = list(
    scaler_full.inverse_transform(np.array(lstm_preds_existing).reshape(-1, 1)).flatten()
)
# Use actual values for first n_lags observations since they cannot be predicted
lstm_full_series = list(gap_df['y'].iloc[:n_lags].values) + lstm_fitted_values + list(lstm_full_vals)

# Ensure all series have the same length (pad shorter series with NaN, trim longer ones)
n_full = len(years_full)

def _pad_or_trim(seq, target_len):
    """
    Helper function to ensure all series have the same length.
    Pads shorter series with NaN, trims longer series.
    """
    seq = list(seq)
    if len(seq) < target_len:
        return seq + [np.nan] * (target_len - len(seq))
    return seq[:target_len]

years_full_aligned = _pad_or_trim(years_full, n_full)
actual_full_aligned = _pad_or_trim(actual_full, n_full)
prophet_full_aligned = _pad_or_trim(prophet_full_series, n_full)
arima_full_aligned = _pad_or_trim(arima_full_series, n_full)
sarima_full_aligned = _pad_or_trim(sarima_full_series, n_full)
lr_full_aligned = _pad_or_trim(lr_full_series, n_full)
xgb_full_aligned = _pad_or_trim(xgb_full_series, n_full)
lstm_full_aligned = _pad_or_trim(lstm_full_series, n_full)

full_table = pd.DataFrame({
    "Year": years_full_aligned,
    "Actual_Gap": actual_full_aligned,
    "Prophet": prophet_full_aligned,
    "ARIMA": arima_full_aligned,
    "SARIMA": sarima_full_aligned,
    "Linear Regression": lr_full_aligned,
    "XGBoost": xgb_full_aligned,
    "LSTM": lstm_full_aligned,
})

print("\n" + "="*80)
print("📊 COMPLETE TIME SERIES: TRAIN + TEST + FUTURE FORECASTS")
print("="*80)
print("\nLegend:")
print("  [Train] = Training period (fitted values)")
print("  [Test]  = Test period (out-of-sample predictions)")
print("  [Future] = Forecast period (future predictions)")
print("\n" + "-" * 80)

# Add period labels
full_table_display = full_table.copy()
full_table_display.insert(1, "Period", "")
full_table_display.loc[:len(train_df)-1, "Period"] = "[Train]"
full_table_display.loc[len(train_df):len(train_df)+n_test-1, "Period"] = "[Test]"
full_table_display.loc[len(train_df)+n_test:, "Period"] = "[Future]"

print(full_table_display.round(2))

# ============================================================================
# MODEL DIAGNOSTICS AND ERROR ANALYSIS
# ============================================================================
# Visualize model performance using MASE
plt.figure(figsize=(10, 5))
models_list = list(results_mase.keys())
mase_vals = [results_mase[m] for m in models_list]
colors = ['green' if v < 1.0 else 'red' for v in mase_vals]  # Green if better than naive
bars = plt.bar(models_list, mase_vals, color=colors, alpha=0.7)
plt.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Naive Baseline (MASE=1.0)')
plt.ylabel("MASE", fontsize=24)
plt.title("Model Performance: MASE on Test Period\n(MASE < 1.0 = better than naive)", fontsize=24, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=24)
plt.yticks(fontsize=24)
for i, (bar, v) in enumerate(zip(bars, mase_vals)):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}", ha='center', fontsize=24)
plt.legend(fontsize=24)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Prediction errors (residuals) plot - focus on best models
plt.figure(figsize=(10, 5))
plt.plot(comparison_df['Year'], comparison_df['Naive_Error'], marker='x', color='gray', label='Naive Error', linewidth=2, markersize=8)
plt.plot(comparison_df['Year'], comparison_df['Prophet_Error'], marker='o', label='Prophet Error', markersize=8)
plt.plot(comparison_df['Year'], comparison_df['ARIMA_Error'], marker='s', label='ARIMA Error', markersize=8)
plt.plot(comparison_df['Year'], comparison_df['SARIMA_Error'], marker='^', label='SARIMA Error', markersize=8)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Year", fontsize=24)
plt.ylabel("Error (Actual - Predicted)", fontsize=24)
plt.title("Prediction Errors (Test Period)", fontsize=24, fontweight='bold')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# ROLLING/EXPANDING WINDOW VALIDATION (ROBUSTNESS CHECK)
# ============================================================================
# This is a more rigorous evaluation method that tests model performance
# across multiple time windows, showing robustness and reducing overfitting concerns
# Expanding window: training set grows over time, test on next observation

print("\n" + "="*80)
print("🔄 ROLLING/EXPANDING WINDOW VALIDATION")
print("="*80)
print("This evaluates model robustness across multiple time windows.")
print("Expanding window: training set grows, test on next observation(s)")
print("-" * 80)

# Parameters for rolling validation
min_train_size = max(5, len(gap_df) // 3)  # Minimum training size (at least 5 or 1/3 of data)
test_size = 1  # Test on 1 observation at a time (most rigorous)

# Store results across all windows
rolling_results = {
    "Prophet": {"mae": [], "rmse": [], "mase": []},
    "ARIMA": {"mae": [], "rmse": [], "mase": []},
    "SARIMA": {"mae": [], "rmse": [], "mase": []},
    "Linear Regression": {"mae": [], "rmse": [], "mase": []},
    "XGBoost": {"mae": [], "rmse": [], "mase": []},
    "LSTM": {"mae": [], "rmse": [], "mase": []},
    "Naive": {"mae": [], "rmse": [], "mase": []}
}

print(f"\nRunning expanding window validation...")
print(f"Minimum training size: {min_train_size}")
print(f"Number of validation windows: {len(gap_df) - min_train_size - test_size + 1}")

# Expanding window: for each time point from min_train_size onwards
for t in range(min_train_size, len(gap_df) - test_size + 1):
    # Training: all data up to t
    train_window = gap_df.iloc[:t].copy()
    # Test: next test_size observations
    test_window = gap_df.iloc[t:t+test_size].copy()
    
    train_vals_window = train_window['y'].values
    test_vals_window = test_window['y'].values
    
    if len(train_vals_window) < 3:  # Need at least 3 for lag features
        continue
    
    try:
        # Naive baseline
        naive_pred = np.repeat(train_vals_window[-1], len(test_vals_window))
        naive_mae = mean_absolute_error(test_vals_window, naive_pred)
        naive_rmse = np.sqrt(mean_squared_error(test_vals_window, naive_pred))
        naive_mase_val = mase(test_vals_window, naive_pred, train_vals_window)
        rolling_results["Naive"]["mae"].append(naive_mae)
        rolling_results["Naive"]["rmse"].append(naive_rmse)
        rolling_results["Naive"]["mase"].append(naive_mase_val)
        
        # Prophet
        try:
            prophet_window = Prophet(yearly_seasonality=False)
            prophet_window.fit(train_window.rename(columns={'ds':'ds','y':'y'}))
            future_window = prophet_window.make_future_dataframe(periods=test_size, freq=prophet_freq)
            forecast_window = prophet_window.predict(future_window)
            prophet_pred = forecast_window['yhat'].iloc[-test_size:].values
            rolling_results["Prophet"]["mae"].append(mean_absolute_error(test_vals_window, prophet_pred))
            rolling_results["Prophet"]["rmse"].append(np.sqrt(mean_squared_error(test_vals_window, prophet_pred)))
            rolling_results["Prophet"]["mase"].append(mase(test_vals_window, prophet_pred, train_vals_window))
        except:
            pass  # Skip if Prophet fails for this window
        
        # ARIMA
        try:
            train_series_window = pd.Series(train_vals_window, index=train_window['ds'])
            train_series_window.index = pd.DatetimeIndex(train_series_window.index, freq='YS')
            arima_window = ARIMA(train_series_window, order=arima_order).fit()
            arima_pred_obj = arima_window.get_forecast(steps=test_size)
            arima_pred = arima_pred_obj.predicted_mean.values
            rolling_results["ARIMA"]["mae"].append(mean_absolute_error(test_vals_window, arima_pred))
            rolling_results["ARIMA"]["rmse"].append(np.sqrt(mean_squared_error(test_vals_window, arima_pred)))
            rolling_results["ARIMA"]["mase"].append(mase(test_vals_window, arima_pred, train_vals_window))
        except:
            pass
        
        # SARIMA
        try:
            sarima_window = SARIMAX(train_series_window, order=sarima_order, 
                                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            sarima_pred_obj = sarima_window.get_forecast(steps=test_size)
            sarima_pred = sarima_pred_obj.predicted_mean.values
            rolling_results["SARIMA"]["mae"].append(mean_absolute_error(test_vals_window, sarima_pred))
            rolling_results["SARIMA"]["rmse"].append(np.sqrt(mean_squared_error(test_vals_window, sarima_pred)))
            rolling_results["SARIMA"]["mase"].append(mase(test_vals_window, sarima_pred, train_vals_window))
        except:
            pass
        
        # Linear Regression
        try:
            time_idx_train = np.arange(len(train_window))
            time_idx_test = np.arange(len(train_window), len(train_window) + test_size)
            lr_window = LinearRegression().fit(time_idx_train.reshape(-1,1), train_vals_window)
            lr_pred = lr_window.predict(time_idx_test.reshape(-1,1))
            rolling_results["Linear Regression"]["mae"].append(mean_absolute_error(test_vals_window, lr_pred))
            rolling_results["Linear Regression"]["rmse"].append(np.sqrt(mean_squared_error(test_vals_window, lr_pred)))
            rolling_results["Linear Regression"]["mase"].append(mase(test_vals_window, lr_pred, train_vals_window))
        except:
            pass
        
        # XGBoost (with lag features)
        try:
            if len(train_vals_window) >= 3:
                train_lags = pd.DataFrame({
                    'y': train_vals_window,
                    'lag1': np.concatenate([[np.nan], train_vals_window[:-1]]),
                    'lag2': np.concatenate([[np.nan, np.nan], train_vals_window[:-2]]),
                    'lag3': np.concatenate([[np.nan, np.nan, np.nan], train_vals_window[:-3]])
                }).dropna()
                if len(train_lags) > 0:
                    X_xgb_w = train_lags[['lag1', 'lag2', 'lag3']].values
                    y_xgb_w = train_lags['y'].values
                    xgb_window = XGBRegressor(n_estimators=100, learning_rate=0.05, 
                                            objective='reg:squarederror', verbosity=0)
                    xgb_window.fit(X_xgb_w, y_xgb_w)
                    # Forecast
                    xgb_pred = []
                    last_vals = train_vals_window[-3:].tolist()
                    for _ in range(test_size):
                        X_test = np.array([[last_vals[-1], last_vals[-2], last_vals[-3]]])
                        pred = xgb_window.predict(X_test)[0]
                        xgb_pred.append(pred)
                        last_vals = last_vals[1:] + [pred]
                    xgb_pred = np.array(xgb_pred)
                    rolling_results["XGBoost"]["mae"].append(mean_absolute_error(test_vals_window, xgb_pred))
                    rolling_results["XGBoost"]["rmse"].append(np.sqrt(mean_squared_error(test_vals_window, xgb_pred)))
                    rolling_results["XGBoost"]["mase"].append(mase(test_vals_window, xgb_pred, train_vals_window))
        except:
            pass
        
        # LSTM (simplified for speed - fewer epochs)
        try:
            if len(train_vals_window) >= n_lags + 1:
                scaler_w = MinMaxScaler()
                train_scaled_w = scaler_w.fit_transform(train_vals_window.reshape(-1,1))
                X_lstm_w = []
                y_lstm_w = []
                for i in range(n_lags, len(train_scaled_w)):
                    X_lstm_w.append(train_scaled_w[i-n_lags:i].flatten())
                    y_lstm_w.append(train_scaled_w[i])
                if len(X_lstm_w) > 0:
                    X_lstm_w = np.array(X_lstm_w).reshape(-1, n_lags, 1)
                    y_lstm_w = np.array(y_lstm_w)
                    lstm_window = Sequential([
                        Input(shape=(n_lags, 1)),
                        LSTM(16, activation='tanh'),  # Smaller for speed
                        Dense(1)
                    ])
                    lstm_window.compile(optimizer='adam', loss='mse')
                    lstm_window.fit(X_lstm_w, y_lstm_w, epochs=50, verbose=0)  # Fewer epochs for speed
                    # Forecast
                    lstm_pred_scaled = []
                    last_window = train_scaled_w[-n_lags:].reshape(1, n_lags, 1)
                    for _ in range(test_size):
                        pred = lstm_window.predict(last_window, verbose=0)
                        lstm_pred_scaled.append(pred.flatten()[0])
                        last_window = np.concatenate([
                            last_window[:, 1:, :],
                            pred.reshape(1, 1, 1)
                        ], axis=1)
                    lstm_pred = scaler_w.inverse_transform(np.array(lstm_pred_scaled).reshape(-1,1)).flatten()
                    rolling_results["LSTM"]["mae"].append(mean_absolute_error(test_vals_window, lstm_pred))
                    rolling_results["LSTM"]["rmse"].append(np.sqrt(mean_squared_error(test_vals_window, lstm_pred)))
                    rolling_results["LSTM"]["mase"].append(mase(test_vals_window, lstm_pred, train_vals_window))
        except:
            pass
    
    except Exception as e:
        continue  # Skip this window if any error occurs

# Aggregate results across all windows
print("\n" + "="*80)
print("📊 ROLLING WINDOW VALIDATION RESULTS (Aggregated)")
print("="*80)
print("Metrics averaged across all validation windows")
print("-" * 80)

rolling_summary = []
for model_name in rolling_results.keys():
    if len(rolling_results[model_name]["mae"]) > 0:
        avg_mae = np.mean(rolling_results[model_name]["mae"])
        avg_rmse = np.mean(rolling_results[model_name]["rmse"])
        avg_mase = np.mean(rolling_results[model_name]["mase"])
        std_mase = np.std(rolling_results[model_name]["mase"])
        n_windows = len(rolling_results[model_name]["mae"])
        rolling_summary.append({
            "Model": model_name,
            "Windows": n_windows,
            "Avg MAE": avg_mae,
            "Avg RMSE": avg_rmse,
            "Avg MASE": avg_mase,
            "Std MASE": std_mase
        })

rolling_summary_df = pd.DataFrame(rolling_summary).set_index("Model")
rolling_summary_df = rolling_summary_df.sort_values("Avg MASE")
print(rolling_summary_df.round(3))

print("\n" + "-" * 80)
print("Key Insights:")
print(f"  • Models tested across {rolling_summary_df['Windows'].max():.0f} validation windows")
print(f"  • Lower MASE = better performance")
print(f"  • Std MASE shows consistency (lower = more stable)")
print(f"  • Best model: {rolling_summary_df.index[0]} (MASE: {rolling_summary_df.iloc[0]['Avg MASE']:.3f})")

# Visualize rolling validation results
plt.figure(figsize=(12, 6))
models_to_plot = [m for m in rolling_summary_df.index if len(rolling_results[m]["mase"]) > 0]
mase_means = [rolling_summary_df.loc[m, "Avg MASE"] for m in models_to_plot]
mase_stds = [rolling_summary_df.loc[m, "Std MASE"] for m in models_to_plot]
colors = ['green' if m < 1.0 else 'red' for m in mase_means]

x_pos = np.arange(len(models_to_plot))
bars = plt.bar(x_pos, mase_means, yerr=mase_stds, capsize=5, color=colors, alpha=0.7)
plt.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Naive Baseline (MASE=1.0)')
plt.xlabel("Model", fontsize=24)
plt.ylabel("MASE (Mean ± Std)", fontsize=24)
plt.title("Rolling Window Validation: Average MASE Across All Windows\n(Lower is Better, Error Bars = Std Dev)", 
          fontsize=24, fontweight='bold')
plt.xticks(x_pos, models_to_plot, rotation=45, ha='right', fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)
plt.grid(alpha=0.3, axis='y')
for i, (bar, mean, std) in enumerate(zip(bars, mase_means, mase_stds)):
    plt.text(bar.get_x() + bar.get_width()/2, mean + std + 0.05, 
             f"{mean:.3f}\n(±{std:.3f})", ha='center', fontsize=24)
plt.tight_layout()
plt.show()

# ============================================================================
# PROPHET MODEL DECOMPOSITION
# ============================================================================
# Display Prophet's time series components (trend, seasonality, etc.)
# This helps understand what patterns Prophet identified in the data
fig = prophet_model.plot_components(forecast_full)
# Update font sizes for all subplots in Prophet components
for ax in fig.axes:
    ax.set_xlabel(ax.get_xlabel(), fontsize=24)
    ax.set_ylabel(ax.get_ylabel(), fontsize=24)
    ax.set_title(ax.get_title(), fontsize=24, fontweight='bold')
    ax.tick_params(labelsize=24)
plt.suptitle("Prophet Components", fontsize=24, fontweight='bold')
plt.tight_layout()
plt.show()

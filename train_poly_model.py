import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import holidays
import argparse
import os
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_features(df):
    logger.info("Creating time-based features...")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['julianday'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    
    us_holidays = holidays.US()
    df['isholiday'] = [1 if d in us_holidays else 0 for d in df.index.date]
    return df

def add_lags(df, target_col):
    logger.info("Adding lag features (1, 2, and 3 years)...")
    df = df.copy()
    target_map = df[target_col].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df

def load_data(data_path):
    logger.info(f"Loading data from {data_path}...")
    ext = os.path.splitext(data_path)[1].lower()
    
    if ext == '.csv':
        df = pd.read_csv(data_path)
    elif ext == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    if 'Datetime' in df.columns:
        df = df.set_index('Datetime')
    
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def train_poly_model(data_path):
    df = load_data(data_path)
    target_col = df.columns[0]
    
    # Feature engineering
    df = create_features(df)
    df = add_lags(df, target_col)
    df = df.dropna()

    # We use a subset of features for Poly to avoid memory explosion
    # The sin/cos already handle the "crazy shapes" of daily cycles!
    FEATURES = ['hour_sin', 'hour_cos', 'dayofweek', 'month', 'isholiday', 'lag1']
    X = df[FEATURES]
    y = df[target_col]

    # The Robust Pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('reg', Ridge())
    ])

    # Intensive GridSearch
    param_grid = {
        'poly__degree': [1, 2, 3],
        'reg__alpha': np.logspace(-3, 3, 10),
        'poly__interaction_only': [True, False]
    }

    tss = TimeSeriesSplit(n_splits=5, test_size=24*365, gap=24)
    
    logger.info(f"Starting Polynomial GridSearch on {len(FEATURES)} features...")
    grid = GridSearchCV(pipe, param_grid, cv=tss, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)
    grid.fit(X, y)

    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Best CV RMSE: {-grid.best_score_:0.4f}")

    # Save
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    model_name = f"poly_{base_name}_model.joblib"
    joblib.dump(grid.best_estimator_, model_name)
    logger.info(f"Model saved as {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_args()
    
    try:
        train_poly_model(args.data)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

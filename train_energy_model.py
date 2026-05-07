import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
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
    logger.info("Adding lag features (24h, 7d, 1y, 2y, 3y)...")
    df = df.copy()
    target_map = df[target_col].to_dict()
    # High-impact short-term lags
    df['lag_24h'] = (df.index - pd.Timedelta('1 days')).map(target_map)
    df['lag_7d'] = (df.index - pd.Timedelta('7 days')).map(target_map)
    # Long-term seasonal lags
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

def train_model(data_path):
    df = load_data(data_path)

    target_col = df.columns[0]
    logger.info(f"Target column detected: {target_col}")

    # Outlier removal (based on notebook)
    initial_count = len(df)
    df = df.query(f'{target_col} > 19000').copy()
    logger.info(f"Removed {initial_count - len(df)} outliers (where {target_col} <= 19000)")

    # Feature engineering
    df = create_features(df)
    df = add_lags(df, target_col)

    # Drop rows with NaN from lags
    df = df.dropna()
    logger.info(f"Final dataset shape after dropping NaNs from lags: {df.shape}")

    FEATURES = ['hour_sin', 'hour_cos', 'dayofweek', 'month', 'year',
                'julianday', 'weekofyear', 'isholiday', 
                'lag_24h', 'lag_7d', 'lag1', 'lag2', 'lag3']
    X = df[FEATURES]
    y = df[target_col]

    # TimeSeriesSplit
    tss = TimeSeriesSplit(n_splits=5, test_size=24*365, gap=24)

    # Pipeline
    pipe = Pipeline([
        ('reg', xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree'))
    ])

    # Param grid for GridSearchCV
    param_grid = {
        'reg__learning_rate': [0.005, 0.01, 0.02], 
        'reg__n_estimators': [400, 500, 700], 
        'reg__max_depth': [5, 7], 
        'reg__subsample': [0.8],
        'reg__colsample_bytree': [0.8]
    }
    #first run
    # param_grid = {
    #     'reg__n_estimators': [500, 1000],
    #     'reg__max_depth': [3, 5],
    #     'reg__learning_rate': [0.01, 0.05],
    #     'reg__subsample': [0.8],
    #     'reg__colsample_bytree': [0.8]
    # }

    logger.info("Starting GridSearchCV...")
    grid = GridSearchCV(pipe, param_grid, cv=tss, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)
    grid.fit(X, y)

    logger.info(f"Best parameters: {grid.best_params_}")
    logger.info(f"Best CV RMSE: {-grid.best_score_:0.4f}")

    # Final evaluation on the last split
    best_model = grid.best_estimator_
    
    # Save the model
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    model_name = f"{base_name}_model.joblib"
    joblib.dump(best_model, model_name)
    logger.info(f"Model saved as {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/PJME_hourly.csv', help='Path to the dataset (CSV or Parquet)')
    args = parser.parse_args()
    
    try:
        train_model(args.data)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

import nbformat

notebook_path = 'time_series_energy_forecast/explore_energy_consumption.ipynb'

with open(notebook_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

# The new correct cells
cell_30 = nbformat.v4.new_code_cell("""# 1. Setup the Exact Features the Model Expects
FEATURES = ['hour_sin', 'hour_cos', 'dayofweek', 'month', 'year',
            'julianday', 'weekofyear', 'isholiday', 
            'lag_24h', 'lag_7d', 'lag1', 'lag2', 'lag3']
TARGET = 'PJME_MW'

# 2. Load the trained model from the joblibs folder
import joblib
daily_xgb_model = joblib.load('joblibs/PJME_hourly_model.joblib')
print("Model loaded successfully!")
""")

cell_31 = nbformat.v4.new_code_cell("""import numpy as np
import pandas as pd
import holidays

def run_recursive_forecast(model, history_df, horizon_hours, target_col, features_list):
    forecast_df = history_df.copy()
    future_predictions = []
    
    us_holidays = holidays.US()

    for _ in range(horizon_hours):
        # 1. Determine the next timestamp
        next_time = forecast_df.index[-1] + pd.Timedelta(hours=1)
        
        # 2. Build the feature row for this specific timestamp
        next_row = pd.DataFrame(index=[next_time])
        
        # 3. Calculate Time-Based Features
        next_row['hour_sin'] = np.sin(2 * np.pi * next_time.hour / 24)
        next_row['hour_cos'] = np.cos(2 * np.pi * next_time.hour / 24)
        next_row['dayofweek'] = next_time.dayofweek
        next_row['month'] = next_time.month
        next_row['year'] = next_time.year
        next_row['julianday'] = next_time.dayofyear
        next_row['weekofyear'] = next_time.isocalendar().week.astype(int)
        
        # 4. Calculate Lag Features (Sourced from history_df + previous predictions)
        next_row['lag_24h'] = forecast_df.iloc[-24][target_col]
        next_row['lag_7d']  = forecast_df.iloc[-168][target_col]
        next_row['lag1']    = forecast_df.iloc[-8736][target_col]
        next_row['lag2']    = forecast_df.iloc[-17472][target_col]
        next_row['lag3']    = forecast_df.iloc[-26208][target_col]
        
        # 5. Handle Holiday 
        next_row['isholiday'] = 1 if next_time.date() in us_holidays else 0 

        # 6. Prediction
        X_input = next_row[features_list]
        prediction = model.predict(X_input)[0]
        
        # 7. Update the rolling history
        next_row[target_col] = prediction
        forecast_df = pd.concat([forecast_df, next_row])
        
        # 8. Store results for final output
        future_predictions.append({
            'Datetime': next_time,
            'Prediction': prediction
        })

    return pd.DataFrame(future_predictions).set_index('Datetime')
""")

cell_32 = nbformat.v4.new_code_cell("""# 3. Run the Forecast for the next 48 hours!
# Note: history_df just needs to be the raw dataframe ending at the "current" time.
# The function will calculate all the features and lags on the fly.

final_forecast = run_recursive_forecast(
    model=daily_xgb_model,
    history_df=df,  # Assuming 'df' is your raw loaded data from earlier in the notebook
    horizon_hours=48, 
    target_col=TARGET, 
    features_list=FEATURES
)

# Display the predictions
final_forecast.head(10)
""")

cell_33 = nbformat.v4.new_code_cell("""import matplotlib.pyplot as plt

# Plot the last 48 hours of REAL data against the 48 hours of PREDICTED data
fig, ax = plt.subplots(figsize=(15, 5))

# Plot historical (last 48 hours)
df[TARGET].iloc[-48:].plot(ax=ax, label='Historical Truth', color='blue')

# Plot prediction
final_forecast['Prediction'].plot(ax=ax, label='Recursive Forecast', color='red', linestyle='--')

ax.set_title("48-Hour Energy Forecast (Recursive XGBoost)")
ax.legend()
plt.show()
""")

# Replace the last 5 broken cells with our 4 clean cells
nb.cells = nb.cells[:-5] + [cell_30, cell_31, cell_32, cell_33]

with open(notebook_path, 'w') as f:
    nbformat.write(nb, f)

print("Notebook successfully updated.")

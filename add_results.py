import nbformat

notebook_path = 'time_series_energy_forecast/explore_energy_consumption.ipynb'

with open(notebook_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

results_md = """# Final Results & Model Comparison

| Model | RMSE | Improvement from Baseline | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline (Mean)** | 6,463.99 | 0% | Naive horizontal predictor |
| **Linear Regression** | 5,342.17 | 17% | Base trend detection |
| **XGBoost (Tutorial)** | 3,727.56 | 42% | Standard feature set (Rob Mulla) |
| **Polynomial Regression** | 3,702.46 | 43% | Unstable (Degree 6, $\\\\alpha=1000$) |
| **XGBoost (Optimized)** | **2,397.94** | **63%** | **Best Performance** (24h/7d Lags) |

## Discussion of Trade-offs

### The \"Wall\" of Forecasting
While the **Lag-Optimized XGBoost** achieved the highest accuracy, it is bound by the **24-hour horizon**. Because the model relies on a 24-hour lag, predicting beyond that window requires a **Recursive Forecasting** strategy (feeding predictions back as inputs). This is powerful but sensitive to \"drift\" over long horizons.

### Stability vs. Complexity
The **Polynomial Regression** (Degree 6) was theoretically competitive but mathematically unstable, as evidenced by the `LinAlgWarnings` during training. To stabilize it, a massive L2 penalty ($\\\\alpha=1000$) was required. For real-world infrastructure like the PJM grid, the tree-based **XGBoost** model remains the superior choice due to its inherent stability and robustness against outliers.
"""

# Create the new cell
results_cell = nbformat.v4.new_markdown_cell(results_md)

# Find where to insert it (before Cell 30 which sets up the features)
# Based on the previous check, Cell 30 was the start of the final section
nb.cells.insert(30, results_cell)

with open(notebook_path, 'w') as f:
    nbformat.write(nb, f)

print("Results section successfully added to the notebook.")

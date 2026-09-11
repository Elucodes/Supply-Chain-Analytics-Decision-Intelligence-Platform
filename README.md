# Supply Chain Analytics Decision Intelligence Platform

This project contains a supply-chain analytics workflow with:

- data cleaning and feature engineering
- demand forecasting
- disruption prediction
- inventory and logistics optimization
- statistical validation
- a Streamlit dashboard

## Project structure

- `src/` – Python analytics modules
- `data/` – source CSV files
- `visualisations/` – generated charts and plots
- `database/` – SQLite outputs created at runtime
- `john/` – project virtual environment

## Quick start

From the project root:

1. Create a virtual environment (if needed):
   ```powershell
   py -m venv john
   ```
2. Install dependencies:
   ```powershell
   .\john\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. Run the analytics pipeline:
   ```powershell
   .\john\Scripts\python.exe .\src\data_processing.py
   ```
4. Launch the dashboard:
   ```powershell
   .\john\Scripts\streamlit.exe run .\src\"analytics dashboard.py"
   ```

## Notes

- The pipeline writes generated tables to the `database` folder.
- The dashboard reads those outputs and displays the analytics results.
- Some forecasting models (such as Prophet and XGBoost) are required by the predictive workflow.

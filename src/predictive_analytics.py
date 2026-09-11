"""
Predictive Analytics Module

Purpose:
- Demand forecasting using ARIMA, Prophet, Random Forest and XGBoost
- Monte Carlo demand simulation
- Supply chain disruption prediction

This module supports data-driven decision making through predictive modelling.
"""

# Predictive Analytics Module
import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")


class PredictiveAnalytics:

    def __init__(self, db_path=None, visual_dir=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        database_dir = os.path.join(base_dir, "database")
        visuals_dir = os.path.join(base_dir, "visuals")

        os.makedirs(database_dir, exist_ok=True)
        os.makedirs(visuals_dir, exist_ok=True)

        self.db_path = db_path or os.path.join(database_dir, "supply_chain.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.visual_dir = visual_dir or visuals_dir
        os.makedirs(self.visual_dir, exist_ok=True)

   
    # Demand Forecasting
   
    def demand_forecasting(self, horizon=30):
        print("\n=== DEMAND FORECASTING ===")
        inv = pd.read_sql("SELECT Date, Demand FROM inventory_engineered", self.conn)
        inv['Date'] = pd.to_datetime(inv['Date'])
        inv = inv.groupby('Date')['Demand'].sum().reset_index().sort_values('Date')

        # Train-test split
        train = inv.iloc[:-horizon]
        test = inv.iloc[-horizon:]

        results = []

        def save_preds(model_name, dates, actual, pred):
            df_pred = pd.DataFrame({"Date": dates, "Actual": actual, "Pred": pred})
            out_path = os.path.join(self.visual_dir, f"forecast_{model_name.lower()}_pred.csv")
            df_pred.to_csv(out_path, index=False)

        # ---- ARIMA
        try:
            arima = ARIMA(train['Demand'], order=(5,1,0)).fit()
            pred = arima.forecast(steps=horizon)
            rmse = np.sqrt(mean_squared_error(test['Demand'], pred))
            mape = mean_absolute_percentage_error(test['Demand'], pred)
            results.append(("ARIMA", rmse, mape))
            save_preds("arima", test['Date'], test['Demand'], pred)

            plt.plot(inv['Date'], inv['Demand'], label="Actual")
            plt.plot(test['Date'], pred, label="ARIMA Forecast")
            plt.legend(); plt.title("ARIMA Forecast vs Actual")
            plt.tight_layout()
            plt.savefig(os.path.join(self.visual_dir, "forecast_arima.png"), dpi=300)
            plt.close()
        except Exception as e:
            print("⚠ ARIMA failed:", e)

        # ---- Prophet
        try:
            prophet_df = train.rename(columns={"Date":"ds","Demand":"y"})
            m = Prophet(daily_seasonality=True)
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=horizon)
            forecast = m.predict(future)
            pred = forecast.iloc[-horizon:]['yhat']
            rmse = np.sqrt(mean_squared_error(test['Demand'], pred))
            mape = mean_absolute_percentage_error(test['Demand'], pred)
            results.append(("Prophet", rmse, mape))
            save_preds("prophet", test['Date'], test['Demand'], pred)

            m.plot(forecast); plt.title("Prophet Forecast")
            plt.tight_layout()
            plt.savefig(os.path.join(self.visual_dir, "forecast_prophet.png"), dpi=300)
            plt.close()
        except Exception as e:
            print("⚠ Prophet failed:", e)

        # ---- Random Forest
        try:
            df = inv.copy()
            df['day'] = df['Date'].dt.dayofyear
            df['year'] = df['Date'].dt.year
            X = df[['day','year']]
            y = df['Demand']
            X_train, X_test = X.iloc[:-horizon], X.iloc[-horizon:]
            y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            pred = rf.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mape = mean_absolute_percentage_error(y_test, pred)
            results.append(("RandomForest", rmse, mape))
            save_preds("rf", test['Date'], y_test, pred)

            plt.plot(inv['Date'], inv['Demand'], label="Actual")
            plt.plot(test['Date'], pred, label="RF Forecast")
            plt.legend(); plt.title("Random Forest Forecast")
            plt.tight_layout()
            plt.savefig(os.path.join(self.visual_dir, "forecast_rf.png"), dpi=300)
            plt.close()
        except Exception as e:
            print("⚠ Random Forest failed:", e)

        # ---- XGBoost
        try:
            df = inv.copy()
            df['day'] = df['Date'].dt.dayofyear
            df['year'] = df['Date'].dt.year
            X = df[['day','year']]
            y = df['Demand']
            X_train, X_test = X.iloc[:-horizon], X.iloc[-horizon:]
            y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]
            XGB = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
            XGB.fit(X_train, y_train)
            pred = XGB.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mape = mean_absolute_percentage_error(y_test, pred)
            results.append(("XGBoost", rmse, mape))
            save_preds("xgb", test['Date'], y_test, pred)

            plt.plot(inv['Date'], inv['Demand'], label="Actual")
            plt.plot(test['Date'], pred, label="XGB Forecast")
            plt.legend(); plt.title("XGBoost Forecast")
            plt.tight_layout()
            plt.savefig(os.path.join(self.visual_dir, "forecast_xgb.png"), dpi=300)
            plt.close()
        except Exception as e:
            print("⚠ XGBoost failed:", e)

        df_results = pd.DataFrame(results, columns=["Model","RMSE","MAPE"])
        df_results.to_sql("forecast_results", self.conn, if_exists="replace", index=False)
        print(df_results)

   
    # Monte Carlo Demand Simulation
  
    def monte_carlo_demand(self, periods=30, sims=1000):
        print("\n=== MONTE CARLO DEMAND SIMULATION ===")
        inv = pd.read_sql("SELECT Date, Demand FROM inventory_engineered", self.conn)
        demand = inv['Demand'].dropna().values
        mu, sigma = np.mean(demand), np.std(demand)

        simulations = []
        for s in range(sims):
            path = np.random.normal(mu, sigma, periods)
            simulations.append(path)

        sim_df = pd.DataFrame(simulations).T
        sim_df.to_sql("monte_carlo_demand", self.conn, if_exists="replace", index=False)

        plt.figure(figsize=(10,6))
        plt.plot(sim_df.iloc[:,:50], alpha=0.3)
        plt.title("Monte Carlo Demand Simulation (50 paths)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.visual_dir, "monte_carlo_demand.png"), dpi=300)
        plt.close()
        print("✔ Monte Carlo simulation saved")


    # Predict Disruptions

    def predict_disruptions(self):
        print("\n=== PREDICTING DISRUPTIONS ===")
        logi = pd.read_sql("SELECT * FROM logistics_engineered", self.conn)
        logi = logi.dropna(subset=['Delay_Flag','Asset_Utilization','Waiting_Time'])

        X = logi[['Asset_Utilization','Waiting_Time','Demand_Forecast','Temperature','Humidity']]
        y = logi['Delay_Flag']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        results = []

        # Logistic Regression
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        logreg_prob = lr.predict_proba(X_test)[:,1]
        logreg_pred = lr.predict(X_test)
        results.append(("LogReg", roc_auc_score(y_test, logreg_prob),
                        precision_score(y_test, logreg_pred),
                        recall_score(y_test, logreg_pred),
                        f1_score(y_test, logreg_pred)))

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_prob = rf.predict_proba(X_test)[:,1]
        rf_pred = rf.predict(X_test)
        results.append(("RandomForest", roc_auc_score(y_test, rf_prob),
                        precision_score(y_test, rf_pred),
                        recall_score(y_test, rf_pred),
                        f1_score(y_test, rf_pred)))

        # XGBoost
        xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
        xgb.fit(X_train, y_train)
        xgb_prob = xgb.predict_proba(X_test)[:,1]
        xgb_pred = xgb.predict(X_test)
        results.append(("XGBoost", roc_auc_score(y_test, xgb_prob),
                        precision_score(y_test, xgb_pred),
                        recall_score(y_test, xgb_pred),
                        f1_score(y_test, xgb_pred)))

        # Save model performance results
        df_results = pd.DataFrame(results, columns=["Model","AUC","Precision","Recall","F1"])
        df_results.to_sql("disruption_results", self.conn, if_exists="replace", index=False)
        print(df_results)

    
        # NEW: Save unified predictions
     
        df_preds = pd.DataFrame({
            "Instance": range(len(y_test)),
            "True_Label": y_test.values,
            "LogReg_Prob": logreg_prob,
            "RF_Prob": rf_prob,
            "XGB_Prob": xgb_prob
        })
        df_preds.to_sql("disruption_predictions", self.conn, if_exists="replace", index=False)
        print("✔ Disruption predictions saved")


   
    # Close
 
    def close(self):
        self.conn.close()


if __name__ == "__main__":
    analytics = PredictiveAnalytics(db_path="supply_chain.db", visual_dir="visuals")
    analytics.demand_forecasting(horizon=30)
    analytics.monte_carlo_demand(periods=30, sims=500)
    analytics.predict_disruptions()
    analytics.close()
    print("\nPREDICTIVE ANALYTICS WORKFLOW COMPLETE: Forecasting, Monte Carlo, and Risk Prediction completed.")

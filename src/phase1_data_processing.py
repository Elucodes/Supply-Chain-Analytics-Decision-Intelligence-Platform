import os
import time
import random
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
pd.options.mode.chained_assignment = None  # silence SettingWithCopyWarning for readability

class Phase1Processor:

    def __init__(self, db_path='supply_chain.db', visual_dir='visuals'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.datasets = {}
        self.visual_dir = visual_dir
        os.makedirs(self.visual_dir, exist_ok=True)

        
        self.expected = {
            'dataco': [
                'Type','Days for shipping (real)','Days for shipment (scheduled)','Benefit per order',
                'Sales per customer','Delivery Status','Late_delivery_risk','Category Id','Category Name',
                'Customer City','Customer Country','Customer Email',
                'Customer Fname','Customer Id','Customer Lname','Customer Password','Customer Segment',
                'Customer State','Customer Street','Customer Zipcode','Department Id','Department Name',
                'Latitude','Longitude','Market','Order City','Order Country','Order Customer Id',
                'order date (DateOrders)','Order Id','Order Item Cardprod Id','Order Item Discount',
                'Order Item Discount Rate','Order Item Id','Order Item Product Price',
                'Order Item Profit Ratio','Order Item Quantity','Sales','Order Item Total',
                'Order Profit Per Order','Order Region','Order State','Order Status','Order Zipcode',
                'Product Card Id','Product Category Id','Product Description','Product Image',
                'Product Name','Product Price','Product Status','shipping date (DateOrders)','Shipping Mode'
            ],
            'inventory': [
                'Date','Store ID','Product ID','Category','Region','Inventory Level','Units Sold',
                'Units Ordered','Demand Forecast','Price','Discount','Weather Condition',
                'Holiday/Promotion','Competitor Pricing','Seasonality'
            ],
            'logistics': [
                'Timestamp','Asset_ID','Latitude','Longitude','Inventory_Level','Shipment_Status',
                'Temperature','Humidity','Traffic_Status','Waiting_Time','User_Transaction_Amount',
                'User_Purchase_Frequency','Logistics_Delay_Reason','Asset_Utilization','Demand_Forecast',
                'Logistics_Delay'
            ]
        }


    # Load CSVs and validate schema
   
    def load_and_validate(self,
                          dataco_path='DataCoSupplyChain.csv',
                          inventory_path='retail_inventory.csv',
                          logistics_path='smart_logistics.csv'):
        print("=== LOADING DATASETS ===")
        # load
        if not os.path.exists(dataco_path):
            raise FileNotFoundError(f"Missing file: {dataco_path}")
        if not os.path.exists(inventory_path):
            raise FileNotFoundError(f"Missing file: {inventory_path}")
        if not os.path.exists(logistics_path):
            raise FileNotFoundError(f"Missing file: {logistics_path}")

        dataco = pd.read_csv(dataco_path, encoding='utf-8-sig', low_memory=False)
        inventory = pd.read_csv(inventory_path, encoding='utf-8-sig', low_memory=False)
        logistics = pd.read_csv(logistics_path, encoding='utf-8-sig', low_memory=False)

        # ✅ FIX: Strip quotes and any remaining BOM characters from column names
        dataco.columns = dataco.columns.str.strip().str.strip("'\"").str.replace('ï»¿', '', regex=False)
        inventory.columns = inventory.columns.str.strip().str.strip("'\"").str.replace('ï»¿', '', regex=False)
        logistics.columns = logistics.columns.str.strip().str.strip("'\"").str.replace('ï»¿', '', regex=False)

        # Validate expected columns (we'll warn for extras, error for missing)
        missing_dataco = [c for c in self.expected['dataco'] if c not in dataco.columns]
        missing_inv = [c for c in self.expected['inventory'] if c not in inventory.columns]
        missing_log = [c for c in self.expected['logistics'] if c not in logistics.columns]

        if missing_dataco:
            raise ValueError(f"dataco missing required columns: {missing_dataco}")
        if missing_inv:
            raise ValueError(f"inventory missing required columns: {missing_inv}")
        if missing_log:
            raise ValueError(f"logistics missing required columns: {missing_log}")

        # store raw
        self.datasets['dataco'] = dataco
        self.datasets['inventory'] = inventory
        self.datasets['logistics'] = logistics
        print("✔ All required columns present for dataco, inventory, logistics")

        # standardize a small set of columns that we'll use
        self._standardize_column_names()

    def _standardize_column_names(self):
        # dataco renames
        d = self.datasets['dataco']
        rename_map = {
            'order date (DateOrders)': 'Order_Date',
            'Order Item Quantity': 'Quantity',
            'Order Profit Per Order': 'Profit',
            'Delivery Status': 'Delivery_Status',
            'Order Id': 'Order_Id',
            'Order Item Product Price': 'Order_Item_Product_Price',
            'Order Item Total': 'Order_Item_Total',
            'Product Card Id': 'Product_Id',
            'Product Name': 'Product_Name',
            'Product Price': 'Product_Price',
            'Customer Id': 'Customer_Id',
            'Latitude': 'Latitude',
            'Longitude': 'Longitude',
            'Days for shipping (real)': 'Days_Shipping_Real',
            'Days for shipment (scheduled)': 'Days_Shipping_Scheduled',
            'Shipping Mode': 'Shipping_Mode',
            'Sales': 'Sales',
            'Category Name': 'Category_Name',
            'Product Category Id': 'Product_Category_Id'
        }
        d.rename(columns={k: v for k, v in rename_map.items() if k in d.columns}, inplace=True)
        self.datasets['dataco'] = d

        # inventory renames
        i = self.datasets['inventory']
        inv_rename = {
            'Store ID': 'Store_ID',
            'Product ID': 'Product_ID',
            'Inventory Level': 'Stock_Level',
            'Units Sold': 'Demand',
            'Units Ordered': 'Units_Ordered',
            'Demand Forecast': 'Demand_Forecast',
            'Holiday/Promotion': 'Holiday_Promotion',
            'Competitor Pricing': 'Competitor_Pricing'
        }
        i.rename(columns={k: v for k, v in inv_rename.items() if k in i.columns}, inplace=True)
        self.datasets['inventory'] = i

        # logistics renames
        l = self.datasets['logistics']
        log_rename = {
            'Asset_ID': 'Asset_ID',
            'Inventory_Level': 'Inventory_Level',
            'Shipment_Status': 'Shipment_Status',
            'Waiting_Time': 'Waiting_Time',
            'Asset_Utilization': 'Asset_Utilization',
            'Demand_Forecast': 'Demand_Forecast',
            'Logistics_Delay': 'Logistics_Delay',
            'Logistics_Delay_Reason': 'Logistics_Delay_Reason'
        }
        l.rename(columns={k: v for k, v in log_rename.items() if k in l.columns}, inplace=True)
        self.datasets['logistics'] = l

    # Cleaning & coercion
    
    def clean_and_coerce(self):
        print("\n=== CLEANING & COERCION ===")
        # dataco
        d = self.datasets['dataco'].copy()
        before = len(d)
        d = d.drop_duplicates().reset_index(drop=True)
        if len(d) != before:
            print(f"dataco: removed {before - len(d)} duplicate rows")
        # parse date
        d['Order_Date'] = pd.to_datetime(d['Order_Date'], errors='coerce')
        if d['Order_Date'].isnull().any():
            print(f"⚠ Warning: {d['Order_Date'].isnull().sum()} rows have invalid Order_Date")
            d = d.dropna(subset=['Order_Date'])
        # numeric coercion for key fields
        for col in ['Quantity', 'Profit', 'Order_Item_Product_Price', 'Order_Item_Total', 'Sales']:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors='coerce')
        self.datasets['dataco'] = d
        print("✔ dataco cleaned")

        # inventory
        i = self.datasets['inventory'].copy()
        before = len(i)
        i = i.drop_duplicates().reset_index(drop=True)
        if len(i) != before:
            print(f"inventory: removed {before - len(i)} duplicate rows")
        i['Date'] = pd.to_datetime(i['Date'], errors='coerce')
        if i['Date'].isnull().any():
            print(f"⚠ Warning: {i['Date'].isnull().sum()} rows have invalid Date")
            i = i.dropna(subset=['Date'])
        for col in ['Stock_Level', 'Demand', 'Units_Ordered', 'Price', 'Discount']:
            if col in i.columns:
                i[col] = pd.to_numeric(i[col], errors='coerce')
        self.datasets['inventory'] = i
        print("✔ inventory cleaned")

        # logistics
        l = self.datasets['logistics'].copy()
        before = len(l)
        l = l.drop_duplicates().reset_index(drop=True)
        if len(l) != before:
            print(f"logistics: removed {before - len(l)} duplicate rows")
        l['Timestamp'] = pd.to_datetime(l['Timestamp'], errors='coerce')
        if l['Timestamp'].isnull().any():
            print(f"⚠ Warning: {l['Timestamp'].isnull().sum()} rows have invalid Timestamp")
            l = l.dropna(subset=['Timestamp'])
        for col in ['Temperature', 'Humidity', 'Waiting_Time', 'Asset_Utilization', 'Demand_Forecast', 'Logistics_Delay']:
            if col in l.columns:
                l[col] = pd.to_numeric(l[col], errors='coerce')
        self.datasets['logistics'] = l
        print("✔ logistics cleaned")


    # Feature engineering
   
    def feature_engineering(self):
        print("\n=== FEATURE ENGINEERING ===")
        # dataco features
        d = self.datasets['dataco'].copy()
        d['Year'] = d['Order_Date'].dt.year
        d['Month'] = d['Order_Date'].dt.month
        # Unit_Price: try Order_Item_Product_Price otherwise Sales/Quantity
        if 'Order_Item_Product_Price' in d.columns:
            d['Unit_Price'] = pd.to_numeric(d['Order_Item_Product_Price'], errors='coerce')
        else:
            d['Unit_Price'] = d['Sales'] / d['Quantity'].replace(0, np.nan)
        # Profit margin from Profit and Sales
        if 'Profit' in d.columns and 'Sales' in d.columns:
            d['Profit_Margin'] = d['Profit'] / d['Sales'].replace(0, np.nan)
        else:
            d['Profit_Margin'] = np.nan
        self.datasets['dataco'] = d
        print("✔ dataco features: Year, Month, Unit_Price, Profit_Margin")

        # inventory features: rolling & lags per Store+Product
        i = self.datasets['inventory'].copy()
        # ensure sort
        if 'Date' not in i.columns or 'Store_ID' not in i.columns or 'Product_ID' not in i.columns:
            raise ValueError("inventory missing Date/Store_ID/Product_ID required for feature engineering")
        i = i.sort_values(['Store_ID', 'Product_ID', 'Date']).reset_index(drop=True)
        # rolling 7-day & 30-day
        i['Demand_7day_avg'] = i.groupby(['Store_ID', 'Product_ID'])['Demand'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        i['Demand_30day_avg'] = i.groupby(['Store_ID', 'Product_ID'])['Demand'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        i['Demand_lag1'] = i.groupby(['Store_ID', 'Product_ID'])['Demand'].shift(1)
        i['Demand_lag2'] = i.groupby(['Store_ID', 'Product_ID'])['Demand'].shift(2)
        # stock indicators
        i['Stock_Out'] = (i['Stock_Level'] == 0).astype(int)
        i['Days_of_Supply'] = i['Stock_Level'] / i['Demand'].replace(0, np.nan)
        self.datasets['inventory'] = i
        print("✔ inventory features: rolling, lags, Stock_Out, Days_of_Supply")

        # logistics features
        l = self.datasets['logistics'].copy()
        l['Year'] = l['Timestamp'].dt.year
        l['Month'] = l['Timestamp'].dt.month
        l['Day_of_Week'] = l['Timestamp'].dt.dayofweek
        l['Hour'] = l['Timestamp'].dt.hour
        # Delay flag
        if 'Logistics_Delay' in l.columns:
            l['Delay_Flag'] = (l['Logistics_Delay'] > 0).astype(int)
        else:
            l['Delay_Flag'] = 0
        self.datasets['logistics'] = l
        print("✔ logistics features: Delay_Flag, time features")

 
    # Persist to DB
    
    def save_engineered_tables(self):
        print("\n=== SAVING ENGINEERED TABLES TO DB ===")
        for name, df in self.datasets.items():
            # clean column names for SQL
            df_sql = df.copy()
            df_sql.columns = [c.strip().replace(' ', '_').replace('/', '_').replace('%','pct') for c in df_sql.columns]
            table_name = f"{name}_engineered"
            df_sql.to_sql(table_name, self.conn, if_exists='replace', index=False)
            print(f"✔ Saved {table_name} ({len(df_sql)} rows)")

        # create some indexes (safe if columns exist)
        cur = self.conn.cursor()
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_dataco_order_date ON dataco_engineered(Order_Date)")
        except Exception:
            pass
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_inventory_date ON inventory_engineered(Date)")
        except Exception:
            pass
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_logistics_time ON logistics_engineered(Timestamp)")
        except Exception:
            pass
        self.conn.commit()
        print("✔ Indexes created where applicable")

    
    # IoT simulation
    
    def simulate_iot(self, num_events=200, lat_center=6.45, lon_center=3.39,
                    commit_every=50, continuous=False, sleep_time=2.0):
       
        print("\n=== SIMULATING IoT EVENTS ===")
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS iot_events (
                Event_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Event_Timestamp TEXT,
                Asset_ID TEXT,
                Location_Lat REAL,
                Location_Lon REAL,
                Temperature REAL,
                Humidity REAL,
                Delay_Flag INTEGER,
                Traffic_Status TEXT
            )
        """)

        inserted = 0

        if continuous:
            print(f"🌍 Continuous IoT simulation started... (new events every {sleep_time}s)")
            try:
                while True:
                    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    asset = f"TRUCK_{random.randint(1, 12)}"
                    lat = round(lat_center + random.uniform(-0.15, 0.15), 6)
                    lon = round(lon_center + random.uniform(-0.15, 0.15), 6)
                    temp = round(random.uniform(15, 35), 2)
                    hum = round(random.uniform(30, 90), 2)
                    delay_flag = int(random.random() < 0.12)
                    traffic = random.choice(['Clear', 'Slow', 'Congested', 'Detour'])

                    cur.execute("""
                        INSERT INTO iot_events (Event_Timestamp, Asset_ID, Location_Lat, Location_Lon,
                                                Temperature, Humidity, Delay_Flag, Traffic_Status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (ts, asset, lat, lon, temp, hum, delay_flag, traffic))
                    self.conn.commit()
                    inserted += 1

                    print(f"Inserted IoT event #{inserted} at {ts}")
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("⏹ Continuous IoT simulation stopped by user.")

        else:
            for i in range(num_events):
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                asset = f"TRUCK_{random.randint(1, 12)}"
                lat = round(lat_center + random.uniform(-0.15, 0.15), 6)
                lon = round(lon_center + random.uniform(-0.15, 0.15), 6)
                temp = round(random.uniform(15, 35), 2)
                hum = round(random.uniform(30, 90), 2)
                delay_flag = int(random.random() < 0.12)
                traffic = random.choice(['Clear', 'Slow', 'Congested', 'Detour'])

                cur.execute("""
                    INSERT INTO iot_events (Event_Timestamp, Asset_ID, Location_Lat, Location_Lon,
                                            Temperature, Humidity, Delay_Flag, Traffic_Status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (ts, asset, lat, lon, temp, hum, delay_flag, traffic))
                
                inserted += 1
                if inserted % commit_every == 0:
                    self.conn.commit()
                    print(f"Inserted {inserted}/{num_events} IoT events...")

                time.sleep(0.01)

            self.conn.commit()
            print(f"✔ IoT simulation completed ({inserted} events).")




    # Baseline KPI computation
    
  
    def compute_baseline_kpis(self):
        print("\n=== COMPUTING BASELINE KPIS ===")
        results = []

        # Inventory KPIs
        inv = pd.read_sql("SELECT * FROM inventory_engineered", self.conn)
        stockout_rate = float(inv['Stock_Out'].mean())
        results.append(('Inventory','Stockout_Rate', stockout_rate))
        results.append(('Inventory','Avg_Demand', float(inv['Demand'].mean())))
        results.append(('Inventory','Median_Days_of_Supply', float(inv['Days_of_Supply'].median())))

        # Logistics KPIs
        log = pd.read_sql("SELECT * FROM logistics_engineered", self.conn)
        delay_rate = float(log['Delay_Flag'].mean())
        results.append(('Logistics','Delay_Rate', delay_rate))
        results.append(('Logistics','Avg_Utilization', float(log['Asset_Utilization'].mean())))
        results.append(('Logistics','Avg_Waiting_Time', float(log['Waiting_Time'].mean())))

        # DataCo KPIs
        dat = pd.read_sql("SELECT * FROM dataco_engineered", self.conn)
        profit_margin = float(dat['Profit_Margin'].mean())
        results.append(('DataCo','Avg_Profit_Margin', profit_margin))

        # ✅ Supplier KPIs
        try:
            if {"Days_Shipping_Real","Days_Shipping_Scheduled"}.issubset(dat.columns):
                dat["On_Time"] = (dat["Days_Shipping_Real"] <= dat["Days_Shipping_Scheduled"]).astype(int)
            else:
                dat["On_Time"] = np.nan

            # Choose supplier/category column
            if "Category_Name" in dat.columns:
                group_col = "Category_Name"
            elif "Product_Category_Id" in dat.columns:
                group_col = "Product_Category_Id"
            else:
                group_col = "Customer_Country"  # fallback

            supplier_perf = (
                dat.groupby(group_col)
                .agg(
                    Orders=("Order_Id", "nunique"),
                    On_Time_Delivery_Rate=("On_Time", "mean"),
                    Avg_Profit_Margin=("Profit_Margin", "mean"),
                    Avg_Sales=("Sales", "mean")
                )
                .reset_index()
            )
            supplier_perf["Perf_Index"] = (
                0.5 * supplier_perf["On_Time_Delivery_Rate"].fillna(0) +
                0.5 * supplier_perf["Avg_Profit_Margin"].fillna(0)
            )
            supplier_perf.rename(columns={group_col: "Supplier"}, inplace=True)
            supplier_perf.to_sql("supplier_kpis", self.conn, if_exists="replace", index=False)
            print("✔ Supplier KPIs computed and saved to 'supplier_kpis'")
        except Exception as e:
            print("⚠ Supplier KPI computation failed:", e)

        # IoT optional if exists
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='iot_events'")
        if cur.fetchone():
            iot = pd.read_sql("SELECT * FROM iot_events", self.conn)
            iot_delay_rate = float(iot['Delay_Flag'].mean())
            results.append(('IoT','IoT_Delay_Rate', iot_delay_rate))
            results.append(('IoT','IoT_Avg_Temperature', float(iot['Temperature'].mean())))
            results.append(('IoT','IoT_Avg_Humidity', float(iot['Humidity'].mean())))

        # Strategic KPI: Resilience Index
        try:
            resilience_index = ((1 - delay_rate) + (1 - stockout_rate) + profit_margin) / 3
            results.append(('Strategic','Resilience_Index', resilience_index))
        except Exception as e:
            print("⚠ Could not compute Resilience Index:", e)

        baseline_df = pd.DataFrame(results, columns=['Domain','KPI','Value'])
        baseline_df.to_sql('baseline_kpis', self.conn, if_exists='replace', index=False)
        print("✔ Baseline KPIs computed and saved to baseline_kpis")
        print(baseline_df)



  
    # Full EDA summaries
    
    def eda_summaries(self, print_sample=True):
        print("\n=== FULL EDA SUMMARIES ===")
        for name in ['dataco', 'inventory', 'logistics']:
            print(f"\n--- {name.upper()} ---")
            df = pd.read_sql(f"SELECT * FROM {name}_engineered LIMIT 10000", self.conn)
            print("Shape:", df.shape)
            # missing values
            mis = df.isnull().sum()
            mis = mis[mis > 0].sort_values(ascending=False)
            print("\nMissing values (top):")
            print(mis.head(20))
            # dtypes
            print("\nDtypes:")
            print(df.dtypes.value_counts())
            # numeric summary
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                print("\nNumeric summary (first 10):")
                print(df[num_cols].describe().T.head(10))
                # correlation matrix for numeric cols (sensible size)
                if len(num_cols) > 1:
                    corr = df[num_cols].corr()
                    print("\nCorrelation matrix (numeric, sample):")
                    print(corr.iloc[:min(8, len(corr)), :min(8, len(corr))].round(3))
            # categorical summary
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                print("\nCategorical columns sample counts:")
                for c in cat_cols[:8]:
                    print(f" - {c}: {df[c].nunique()} unique")
            if print_sample:
                print("\nSample rows:")
                print(df.head(3).T)

  
    # Full visuals 
 
    def generate_visuals(self):
        print("\n=== GENERATING FULL VISUALS ===")

        # Dataco visuals
        try:
            df = pd.read_sql(
                "SELECT Order_Date, Sales, Profit_Margin, Profit FROM dataco_engineered",
                self.conn
            )
            df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
            df = df.dropna(subset=['Order_Date'])
            df['Order_Month'] = df['Order_Date'].dt.to_period('M').astype(str)

            monthly = df.groupby("Order_Month")[["Sales","Profit"]].sum().reset_index()
            plt.figure(figsize=(10,5))
            sns.lineplot(data=monthly, x="Order_Month", y="Sales", label="Sales")
            sns.lineplot(data=monthly, x="Order_Month", y="Profit", label="Profit")
            plt.xticks(rotation=45)
            plt.title("Monthly Sales and Profit")
            plt.tight_layout()
            plt.savefig(os.path.join(self.visual_dir, "dataco_monthly_sales_profit.png"), dpi=300)
            plt.close()
            print("✔ Dataco visuals saved")
        except Exception as e:
            print("⚠ Dataco visuals failed:", e)

        # Inventory visuals
        try:
            df = pd.read_sql(
                "SELECT Date, Demand, Demand_7day_avg, Stock_Level, Stock_Out FROM inventory_engineered",
                self.conn
            )
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            plt.figure(figsize=(10,5))
            sns.lineplot(data=df, x="Date", y="Demand", label="Demand", alpha=0.7)
            sns.lineplot(data=df, x="Date", y="Demand_7day_avg", label="7-day Avg", color="red")
            plt.title("Demand and Rolling Average")
            plt.tight_layout()
            plt.savefig(os.path.join(self.visual_dir, "inventory_demand_trends.png"), dpi=300)
            plt.close()
            print("✔ Inventory visuals saved")
        except Exception as e:
            print("⚠ Inventory visuals failed:", e)

        # Logistics visuals
        try:
            df = pd.read_sql(
                "SELECT Timestamp, Delay_Flag, Asset_Utilization, Waiting_Time FROM logistics_engineered",
                self.conn
            )
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            plt.figure(figsize=(8,5))
            sns.histplot(df["Waiting_Time"].dropna(), bins=30, kde=True)
            plt.title("Distribution of Waiting Times")
            plt.tight_layout()
            plt.savefig(os.path.join(self.visual_dir, "logistics_waiting_times.png"), dpi=300)
            plt.close()
            print("✔ Logistics visuals saved")
        except Exception as e:
            print("⚠ Logistics visuals failed:", e)

        # IoT visuals
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='iot_events'")
            if cur.fetchone():
                df = pd.read_sql(
                    "SELECT Event_Timestamp, Temperature, Humidity, Delay_Flag FROM iot_events",
                    self.conn
                )
                df['Event_Timestamp'] = pd.to_datetime(df['Event_Timestamp'], errors='coerce')
                plt.figure(figsize=(10,5))
                sns.lineplot(data=df, x="Event_Timestamp", y="Temperature", label="Temp")
                sns.lineplot(data=df, x="Event_Timestamp", y="Humidity", label="Humidity")
                plt.title("IoT Sensor Trends")
                plt.tight_layout()
                plt.savefig(os.path.join(self.visual_dir, "iot_trends.png"), dpi=300)
                plt.close()
                print("✔ IoT visuals saved")
        except Exception as e:
            print("⚠ IoT visuals failed:", e)

        # ✅ KPI Visuals (Baseline)
        try:
            df = pd.read_sql("SELECT * FROM baseline_kpis", self.conn)
            plt.figure(figsize=(10,5))
            sns.barplot(data=df, x="KPI", y="Value", hue="Domain", dodge=False)
            plt.title("Baseline KPIs (including Resilience Index)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(self.visual_dir, "baseline_kpis.png"), dpi=300)
            plt.close()
            print("✔ Baseline KPI visuals saved")
        except Exception as e:
            print("⚠ Baseline KPI visuals failed:", e)


    
        print(f"All visuals saved to '{self.visual_dir}'")



    # Clean close
 
    def close(self):
        self.conn.close()


if __name__ == "__main__":
    p = Phase1Processor(db_path='supply_chain.db', visual_dir='visuals')

    # Load & validate datasets
    p.load_and_validate(
        dataco_path='DataCoSupplyChain.csv',
        inventory_path='retail_inventory.csv',
        logistics_path='smart_logistics.csv'
    )

    # Clean and coerce datatypes
    p.clean_and_coerce()

    # Feature engineering
    p.feature_engineering()

    # Save engineered tables to DB
    p.save_engineered_tables()

    # Compute baseline KPIs
    p.compute_baseline_kpis()

    # Full EDA summaries
    p.eda_summaries(print_sample=True)

    # Generate visuals
    p.generate_visuals()

    # Continuous IoT streaming (keeps inserting events until you press Ctrl+C)
    print("\n🇬🇧 Starting continuous IoT streaming... Press Ctrl+C to stop.")
    p.simulate_iot(
        continuous=True, 
        sleep_time=5.0,
        lat_center=51.5074,   # London latitude
        lon_center=-0.1278    # London longitude
    )

    # Close DB connection (runs only after IoT loop is stopped manually)
    p.close()
    print("\nPHASE 1 COMPLETE (pipeline + continuous IoT mode).")
import pandas as pd
import sqlite3
import os

def ingest_raw_data():
    print("--> Starting data ingestion step...")
    
    # 1. Define paths to your files
    csv_path = "data/raw_data.xlsx"  # Even though it ends in .xlsx, your file content is CSV text!
    db_path = "data/gas_monitoring.db"
    
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    
    # 2. Read the raw text data
    print(f"    Reading raw data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"    Successfully loaded {len(df)} rows into memory.")
    
    # Standardize column names to lowercase to prevent SQLite crashes
    df.columns = df.columns.str.strip().str.lower()
    
    # 3. Save it into your SQLite database table
    print(f"    Writing data to SQLite database at {db_path}...")
    conn = sqlite3.connect(db_path)
    df.to_sql("raw_gas_data", conn, if_exists="replace", index=False)
    conn.close()
    
    print("SUCCESS: Raw data successfully ingested into 'raw_gas_data' table!")

if __name__ == "__main__":
    ingest_raw_data()
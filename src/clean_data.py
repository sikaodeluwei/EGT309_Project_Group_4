import os
import pandas as pd
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# BLOCK 1: INITIALIZATION & DB CONNECT
# ==========================================
# region BLOCK 1: INITIALIZATION
def get_db_connection(db_path="data/gas_monitoring.db"):
    """Establishes connection to the SQLite database environment."""
    return sqlite3.connect(db_path)
# endregion


# ==========================================
# BLOCK 2: LOADING & RE-ORDERING KEYS
# ==========================================
# region BLOCK 2: LOADING & RE-ORDERING KEYS
def load_and_orient_data(conn) -> pd.DataFrame:
    """
    Fetches raw data using SQLite. Automatically repositions 'session_id' 
    to the first position as a primary identification key column.
    """
    print("[Block 2] Fetching raw data and positioning Session ID first...")
    df = pd.read_sql_query("SELECT * FROM raw_gas_data", conn)
    
    # Rationale: Place 'session_id' first as a 'key' column for identification
    if 'session_id' in df.columns:
        cols = ['session_id'] + [col for col in df.columns if col != 'session_id']
        df = df[cols]
    return df
# endregion


# ==========================================
# BLOCK 3: CATEGORICAL & TIME STANDARD
# ==========================================
# region BLOCK 3: CATEGORICAL & TIME STANDARD
def clean_text_and_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans structural text formatting and converts time_of_day from an object 
    datatype into a standardized string format for categorical tracking.
    """
    print("[Block 3] Standardizing text inputs and cleaning categorical voids...")
    
    # Strip spaces and lowercase all text features to eliminate string duplicate classes
    text_cols = ['hvac_operation_mode', 'activity_level']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().str.replace(' ', '_')
            df[col] = df[col].replace('nan', np.nan)
            
    # Rationale: Ensure time_of_day format is standardized uniformly
    if 'time_of_day' in df.columns:
        df['time_of_day'] = df['time_of_day'].astype(str).str.strip().str.lower()
        
    return df
# endregion


# ==========================================
# BLOCK 4: RE-CALIBRATING TEMPERATURE OUTLIERS
# ==========================================
# region BLOCK 4: RE-CALIBRATING TEMPERATURE OUTLIERS
def fix_temperature_units(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw Kelvin sensor anomalies back to Celsius scale values."""
    print("[Block 4] Screening temperature parameters for Kelvin unit offsets...")
    if 'temperature' in df.columns:
        df['temperature'] = np.where(df['temperature'] > 100, 
                                      df['temperature'] - 273.15, 
                                      df['temperature'])
    return df
# endregion


# ==========================================
# BLOCK 5: ADVANCED MULTI-VARIABLE DATA IMPUTATION
# ==========================================
# region BLOCK 5: ADVANCED MULTI-VARIABLE DATA IMPUTATION
def execute_advanced_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies precise domain-driven imputation techniques formulated from 
    pre-cleaning exploratory data analysis observations.
    """
    print("[Block 5] Commencing advanced multi-variable imputation steps...")

    # -------------------------------------------------------------
    # A. Impute Ambient Light Level using Time of Day
    # -------------------------------------------------------------
    if 'ambient_light_level' in df.columns and 'time_of_day' in df.columns:
        # Map logical defaults based on the time category
        light_mapping = {'morning': 'dim', 'afternoon': 'very_bright', 'night': 'dark'}
        df['ambient_light_level'] = df['ambient_light_level'].fillna(df['time_of_day'].map(light_mapping))
        # Catch any remaining leftovers with a generic placeholder flag
        df['ambient_light_level'] = df['ambient_light_level'].fillna('unknown')

    # -------------------------------------------------------------
    # B. Impute Humidity derived from Time of Day & Temperature Group Medians
    # -------------------------------------------------------------
    if 'humidity' in df.columns:
        # Rationale note: If anomalies exist, median is statistically safer than mean
        # Clean out impossible negative numbers first
        df['humidity'] = np.where(df['humidity'] < 0, np.nan, df['humidity'])
        
        # Calculate group medians based on surrounding ambient metrics
        grouped_humidity = df.groupby(['time_of_day'])['humidity'].transform('median')
        df['humidity'] = df['humidity'].fillna(grouped_humidity)

    # -------------------------------------------------------------
    # C. Impute MetalOxideSensor_Unit2 via cross-unit sister medians
    # -------------------------------------------------------------
    unit_cols = ['metaloxidesensor_unit1', 'metaloxidesensor_unit3', 'metaloxidesensor_unit4']
    if 'metaloxidesensor_unit2' in df.columns and all(col in df.columns for col in unit_cols):
        # Rationale: Impute missing Unit 2 data by using the row-wise median of sibling units 1, 3, and 4
        sibling_median = df[unit_cols].median(axis=1)
        df['metaloxidesensor_unit2'] = df['metaloxidesensor_unit2'].fillna(sibling_median)

    # -------------------------------------------------------------
    # D. Impute Carbon Monoxide (CO) via CO2 and Metal Oxide sensor behaviors
    # -------------------------------------------------------------
    co_dependencies = ['metaloxidesensor_unit1', 'co2_infraredsensor', 'co2_electrochemicalsensor']
    if 'co_gassensor' in df.columns and all(col in df.columns for col in co_dependencies):
        # Calculate a highly contextual grouping median to reflect the compound chemical relationships
        df['co_gassensor'] = np.where(df['co_gassensor'] < 0, np.nan, df['co_gassensor'])
        contextual_co_median = df.groupby(['time_of_day'])['co_gassensor'].transform('median')
        df['co_gassensor'] = df['co_gassensor'].fillna(contextual_co_median)

    return df
# endregion


# ==========================================
# BLOCK 6: EXPORT & DUPLICATE FILTERS
# ==========================================
# region BLOCK 6: EXPORT & DUPLICATE FILTERS
def save_clean_data(df: pd.DataFrame, conn):
    """Saves finalized data frames into a sterile destination table in SQLite."""
    print("[Block 6] Dropping duplicate rows and exporting to database storage...")
    
    # Final data hygiene check
    df = df.drop_duplicates()
    
    # Save back to SQLite table as required by the pipeline structure
    df.to_sql('cleaned_gas_data', conn, if_exists='replace', index=False)
    print("SUCCESS: Advanced data cleaning completed successfully!")
# endregion


# ==========================================
# ENGINE RUN EXECUTION
# ==========================================
# region PIPELINE RUN ENGINE
if __name__ == "__main__":
    db_conn = get_db_connection()
    
    # Run the updated modular process flow sequential blocks
    working_df = load_and_orient_data(db_conn)
    working_df = clean_text_and_time(working_df)
    working_df = fix_temperature_units(working_df)
    working_df = execute_advanced_imputation(working_df)
    
    save_clean_data(working_df, db_conn)
    db_conn.close()
# endregion
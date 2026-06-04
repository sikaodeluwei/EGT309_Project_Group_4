import os
import pandas as pd
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity

# class DataLoader: # left for future modularization
# connect to database
conn = sqlite3.connect("gas_monitoring.db")

# read sample data (uses SQL language to pull data from the database)
df = pd.read_sql_query("SELECT * FROM gas_monitoring;", conn)

# class DataUniformer: # left for future modularization
# ensure categorical columns dtype is str for uniform processing (that uses str dtype)
df = df.astype({'Time of Day': str, 'HVAC Operation Mode': str,
            'Ambient Light Level': str, 'Activity Level': str})


# make colums lowercase and replace spaces with underscores
df.columns = df.columns.str.strip().str.lower() # lowercase
df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(' ', '_') # strip spaces to underscores

# make all cell data lowercase and replace spaces with underscores
df = df.map(lambda x: x.lower() if isinstance(x, str) else x) # lowercase
df = df.map(lambda x: x.lower().replace(' ', '_') if isinstance(x, str) else x) # strip spaces to underscores

# add spacing to values for data consistency
df['activity_level'] = df['activity_level'].replace('lowactivity',
                                                    'low_activity')
df['activity_level'] = df['activity_level'].replace('moderateactivity',
                                                'moderate_activity')

# drop session_id as not needed for model training
dfv2 = df.copy().drop('session_id', axis=1)

# reordering data table
# e.g. Columns: ['C', 'A', 'B'] > ['A', 'B', 'C']
dfv3 = dfv2.copy()[['co_gassensor',
                    'co2_infraredsensor',
                    'co2_electrochemicalsensor',

                    'metaloxidesensor_unit1',
                    'metaloxidesensor_unit2',
                    'metaloxidesensor_unit3',
                    'metaloxidesensor_unit4',

                    'temperature',
                    'humidity',
                    'ambient_light_level',
                    'time_of_day',

                    'hvac_operation_mode', 
                    
                    'activity_level']]

# convert temperature from Kelvin to Celsius if values are above 100 (assuming all temps above 100 are in Kelvin)
dfv3['temperature'] = np.where(dfv3['temperature'] > 100, # if temp > 100
                             dfv3['temperature'] - 273.15, # then temp - 273.15
                             dfv3['temperature']) # unchanged if condition unmet

# class DataImputer: # left for future modularization

# impute ambient light level based on time of day
light_mapping = {'morning': 'dim', 'afternoon': 'very_bright', 'night': 'dark'}
dfv3['ambient_light_level'] = dfv3['ambient_light_level'].fillna(dfv3['time_of_day'].map(light_mapping))
dfv3['ambient_light_level'] = dfv3['ambient_light_level'].fillna('np.nan') # optional fallback for any remaining nulls

# impute co_gassensor values based on global median
dfv3['co_gassensor'] = dfv3['co_gassensor'].fillna(dfv3['co_gassensor'].median())

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
    
    # FORCE LOWERCASE AGAIN: Absolute safety net for column headers
    df.columns = df.columns.str.strip().str.lower()
    print(f"    -> Current available columns: {list(df.columns)}")

    # -------------------------------------------------------------
    # A. Impute Ambient Light Level using Time of Day
    # -------------------------------------------------------------
    if 'ambient_light_level' in df.columns and 'time_of_day' in df.columns:
        light_mapping = {'morning': 'dim', 'afternoon': 'very_bright', 'night': 'dark'}
        df['ambient_light_level'] = df['ambient_light_level'].fillna(df['time_of_day'].map(light_mapping))
        df['ambient_light_level'] = df['ambient_light_level'].fillna('unknown')

    # -------------------------------------------------------------
    # B. Impute Humidity derived from Time of Day & Temperature Group Medians
    # -------------------------------------------------------------
    if 'humidity' in df.columns:
        df['humidity'] = np.where(df['humidity'] < 0, np.nan, df['humidity'])
        
        # Safe Check: Ensure time_of_day exists before grouping, otherwise fallback to global median
        if 'time_of_day' in df.columns:
            grouped_humidity = df.groupby(['time_of_day'])['humidity'].transform('median')
            df['humidity'] = df['humidity'].fillna(grouped_humidity)
        else:
            print("    [Warning] 'time_of_day' column missing for group-by! Falling back to global median.")
            df['humidity'] = df['humidity'].fillna(df['humidity'].median())

    # -------------------------------------------------------------
    # C. Impute MetalOxideSensor_Unit2 via cross-unit sister medians
    # -------------------------------------------------------------
    unit_cols = ['metaloxidesensor_unit1', 'metaloxidesensor_unit3', 'metaloxidesensor_unit4']
    if 'metaloxidesensor_unit2' in df.columns and all(col in df.columns for col in unit_cols):
        sibling_median = df[unit_cols].median(axis=1)
        df['metaloxidesensor_unit2'] = df['metaloxidesensor_unit2'].fillna(sibling_median)

    # -------------------------------------------------------------
    # D. Impute Carbon Monoxide (CO) via CO2 behaviors
    # -------------------------------------------------------------
    if 'co_gassensor' in df.columns:
        df['co_gassensor'] = np.where(df['co_gassensor'] < 0, np.nan, df['co_gassensor'])
        if 'time_of_day' in df.columns:
            contextual_co_median = df.groupby(['time_of_day'])['co_gassensor'].transform('median')
            df['co_gassensor'] = df['co_gassensor'].fillna(contextual_co_median)
        else:
            df['co_gassensor'] = df['co_gassensor'].fillna(df['co_gassensor'].median())

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
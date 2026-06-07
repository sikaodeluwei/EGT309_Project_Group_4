import os
import pandas as pd
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity

class DataLoader:
    """Connects to database and loads raw data"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        return self

    def load(self, query: str = "SELECT * FROM gas_monitoring;") -> pd.DataFrame:
        if self.conn is None:
            raise RuntimeError("Call connect() before load().")
        return pd.read_sql_query(query, self.conn)

    def close(self):
        if self.conn:
            self.conn.close()

class DataUniformer:
    """Normalises column names and cell values; corrects known label inconsistencies."""

    CATEGORICAL_COLS = ['Time of Day', 'HVAC Operation Mode',
                        'Ambient Light Level', 'Activity Level']

    ACTIVITY_FIXES = {
        'lowactivity': 'low_activity',
        'moderateactivity': 'moderate_activity',
    }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Ensure categorical columns are str dtype
        existing = [c for c in self.CATEGORICAL_COLS if c in df.columns]
        df = df.astype({c: str for c in existing})

        # Normalise column names
        df.columns = (df.columns.str.strip().str.lower()
                                .str.replace(' ', '_'))

        # Normalise cell values
        df = df.map(lambda x: x.lower().replace(' ', '_')
                    if isinstance(x, str) else x)

        # Fix known label inconsistencies in activity_level
        if 'activity_level' in df.columns:
            df['activity_level'] = df['activity_level'].replace(self.ACTIVITY_FIXES)

        return df
    
class DataImputer:
    """Fills missing values using domain-aware strategies."""

    LIGHT_FROM_TIME = {
        'morning': 'dim',
        'afternoon': 'very_bright',
        'night': 'dark',
    }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Ambient light level: impute from time of day, then fallback
        if 'ambient_light_level' in df.columns and 'time_of_day' in df.columns:
            df['ambient_light_level'] = df['ambient_light_level'].fillna(
                df['time_of_day'].map(self.LIGHT_FROM_TIME)
            )
            df['ambient_light_level'] = df['ambient_light_level'].fillna(np.nan)

        # CO sensor: impute with global median
        if 'co_gassensor' in df.columns:
            df['co_gassensor'] = df['co_gassensor'].fillna(
                df['co_gassensor'].median()
            )

        # -------------------------------------------------------------
        # Future: Impute MetalOxideSensor_Unit2 via cross-unit sibling medians
        # -------------------------------------------------------------
        # unit_cols = ['metaloxidesensor_unit1',
        #              'metaloxidesensor_unit3', 'metaloxidesensor_unit4']
        # if 'metaloxidesensor_unit2' in df.columns and \
        #         all(col in df.columns for col in unit_cols):
        #     sibling_median = df[unit_cols].median(axis=1)
        #     df['metaloxidesensor_unit2'] = df['metaloxidesensor_unit2'].fillna(
        #         sibling_median)

        # -------------------------------------------------------------
        # Future: Impute CO via contextual (time_of_day) median
        # -------------------------------------------------------------
        # if 'co_gassensor' in df.columns:
        #     df['co_gassensor'] = np.where(
        #         df['co_gassensor'] < 0, np.nan, df['co_gassensor'])
        #     if 'time_of_day' in df.columns:
        #         ctx_median = df.groupby('time_of_day')['co_gassensor'] \
        #                        .transform('median')
        #         df['co_gassensor'] = df['co_gassensor'].fillna(ctx_median)
        #     else:
        #         df['co_gassensor'] = df['co_gassensor'].fillna(
        #             df['co_gassensor'].median())

        return df


class DataCleaner:
    """Applies column selection, temperature conversion, and row deduplication."""

    COLUMN_ORDER = [
        'co_gassensor',
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
        'activity_level',
    ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Drop session_id if present
        df = df.drop(columns=['session_id'], errors='ignore')

        # Reorder columns (keep only those that exist)
        ordered = [c for c in self.COLUMN_ORDER if c in df.columns]
        df = df[ordered]

        # Convert temperature from Kelvin to Celsius where > 150 K
        if 'temperature' in df.columns:
            df['temperature'] = np.where(
                df['temperature'] > 150,
                df['temperature'] - 273.15,
                df['temperature'],
            )

        # Replace sentinel 'none' with NaN, then drop remaining nulls
        df['ambient_light_level'] = df['ambient_light_level'].replace(
            'none', np.nan)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        return df

class FeatureEngineer:
    """Encodes categorical features as integers."""

    TIME_OF_DAY_MAP = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}

    AMBIENT_LIGHT_MAP = {
        'very_dim': 0, 'dim': 1, 'moderate': 2, 'bright': 3, 'very_bright': 4
    }

    ACTIVITY_LEVEL_MAP = {
        'low_activity': 0, 'moderate_activity': 1, 'high_activity': 2
    }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'time_of_day' in df.columns:
            df['time_of_day'] = df['time_of_day'].map(self.TIME_OF_DAY_MAP)

        if 'ambient_light_level' in df.columns:
            df['ambient_light_level'] = df['ambient_light_level'].map(
                self.AMBIENT_LIGHT_MAP)

        if 'activity_level' in df.columns:
            df['activity_level'] = df['activity_level'].map(
                self.ACTIVITY_LEVEL_MAP)

        # Integer encoding for HVAC mode (ordinal by discovery order)
        # TODO: replace with One-Hot Encoding
        if 'hvac_operation_mode' in df.columns:
            classes = df['hvac_operation_mode'].unique()
            hvac_map = {mode: i for i, mode in enumerate(classes)}
            df['hvac_operation_mode'] = df['hvac_operation_mode'].map(hvac_map)

        return df

class DataSaver:
    """Persists a DataFrame to a SQLite database."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def save(self, df: pd.DataFrame, table_name: str = 'cleaned_data'):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    loader = DataLoader("gas_monitoring.db")
    loader.connect()
    df = loader.load()
    loader.close()

    df = DataUniformer().transform(df)
    df = DataImputer().transform(df)
    df = DataCleaner().transform(df)
    df = FeatureEngineer().transform(df)

    DataSaver('data/gas_monitoring_cleanedv1.db').save(df)

    print("Finished Running Data Cleaning and Feature Engineering")

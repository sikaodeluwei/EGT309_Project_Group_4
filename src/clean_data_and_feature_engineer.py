import os
import pandas as pd
import numpy as np
import sqlite3
from sklearn.impute import KNNImputer

import config


class DataLoader:
    """Handles database connection and raw data loading."""

    def __init__(self, db_path: str = config.LOADER_DB_PATH):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        return self

    def load(self, query: str = config.LOADER_QUERY) -> pd.DataFrame:
        if self.conn is None:
            raise RuntimeError("Call connect() before load().")
        return pd.read_sql_query(query, self.conn)

    def close(self):
        if self.conn:
            self.conn.close()


class DataUniformer:
    """Normalises column names and cell values; corrects known label inconsistencies."""

    def __init__(
        self,
        categorical_cols: list[str] = config.UNIFORMER_CATEGORICAL_COLS,
        activity_fixes: dict[str, str] = config.UNIFORMER_ACTIVITY_FIXES,
    ):
        self.categorical_cols = categorical_cols
        self.activity_fixes = activity_fixes

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Ensure categorical columns are str dtype
        existing = [c for c in self.categorical_cols if c in df.columns]
        df = df.astype({c: str for c in existing})

        # Normalise column names
        df.columns = (df.columns.str.strip().str.lower()
                                .str.replace(' ', '_'))

        # Normalise cell values
        df = df.map(lambda x: x.lower().replace(' ', '_')
                    if isinstance(x, str) else x)

        # Fix known label inconsistencies in activity_level
        if 'activity_level' in df.columns:
            df['activity_level'] = df['activity_level'].replace(self.activity_fixes)

        return df


class DataImputer:
    """Fills missing values using domain-aware strategies."""

    def __init__(
        self,
        light_from_time: dict[str, str] = config.IMPUTER_LIGHT_FROM_TIME,
        n_neighbors: int = 5
    ):
        self.light_from_time = light_from_time
        # Initialize the KNN imputer here
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Ambient light level: impute from time of day, then fallback
        if 'ambient_light_level' in df.columns and 'time_of_day' in df.columns:
            df['ambient_light_level'] = df['ambient_light_level'].fillna(
                df['time_of_day'].map(self.light_from_time)
            )
            df['ambient_light_level'] = df['ambient_light_level'].fillna(np.nan)

        # Humidity: impute out-of-bounds values using Temperature via KNN
        if 'humidity' in df.columns and 'temperature' in df.columns:
            # Convert invalid humidity values (< 0 or > 100) to NaN so KNNImputer recognizes them
            invalid_humidity_mask = (df['humidity'] < 0) | (df['humidity'] > 100)
            df.loc[invalid_humidity_mask, 'humidity'] = np.nan

            # Extract only the columns needed for this specific imputation
            # (KNN relies on distance, so including unrelated columns might distort results)
            impute_cols = ['temperature', 'humidity']
            
            # Fit and transform the subset, then map it back to the dataframe
            # Note: KNNImputer returns a numpy array, so we grab the second column [:, 1] for humidity
            df['humidity'] = self.knn_imputer.fit_transform(df[impute_cols])[:, 1]

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

    def __init__(
        self,
        column_order: list[str] = config.CLEANER_COLUMN_ORDER,
        kelvin_threshold: float = config.CLEANER_KELVIN_THRESHOLD,
        light_sentinel: str = config.CLEANER_LIGHT_SENTINEL,
    ):
        self.column_order = column_order
        self.kelvin_threshold = kelvin_threshold
        self.light_sentinel = light_sentinel

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Drop session_id if present
        df = df.drop(columns=['session_id'], errors='ignore')

        # Reorder columns (keep only those that exist)
        ordered = [c for c in self.column_order if c in df.columns]
        df = df[ordered]

        # Convert temperature from Kelvin to Celsius where > threshold
        if 'temperature' in df.columns:
            df['temperature'] = np.where(
                df['temperature'] > self.kelvin_threshold,
                df['temperature'] - 273.15,
                df['temperature'],
            )

        # Replace sentinel value with NaN, then drop remaining nulls
        df['ambient_light_level'] = df['ambient_light_level'].replace(
            self.light_sentinel, np.nan)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        return df


class FeatureEngineer:
    """Encodes categorical features as integers."""

    def __init__(
        self,
        time_of_day_map: dict[str, int] = config.FEATURE_TIME_OF_DAY_MAP,
        ambient_light_map: dict[str, int] = config.FEATURE_AMBIENT_LIGHT_MAP,
        activity_level_map: dict[str, int] = config.FEATURE_ACTIVITY_LEVEL_MAP,
    ):
        self.time_of_day_map = time_of_day_map
        self.ambient_light_map = ambient_light_map
        self.activity_level_map = activity_level_map

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'time_of_day' in df.columns:
            df['time_of_day'] = df['time_of_day'].map(self.time_of_day_map)

        if 'ambient_light_level' in df.columns:
            df['ambient_light_level'] = df['ambient_light_level'].map(
                self.ambient_light_map)

        if 'activity_level' in df.columns:
            df['activity_level'] = df['activity_level'].map(self.activity_level_map)

        # Integer encoding for HVAC mode (ordinal by discovery order)
        # TODO: replace with One-Hot Encoding
        if 'hvac_operation_mode' in df.columns:
            classes = df['hvac_operation_mode'].unique()
            hvac_map = {mode: i for i, mode in enumerate(classes)}
            df['hvac_operation_mode'] = df['hvac_operation_mode'].map(hvac_map)

        return df


class DataSaver:
    """Persists a DataFrame to a SQLite database."""

    def __init__(self, db_path: str = config.SAVER_DB_PATH):
        self.db_path = db_path

    def save(self, df: pd.DataFrame, table_name: str = config.SAVER_TABLE_NAME):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    loader = DataLoader()
    loader.connect()
    df = loader.load()
    loader.close()

    df = DataUniformer().transform(df)
    df = DataImputer().transform(df)
    df = DataCleaner().transform(df)
    df = FeatureEngineer().transform(df)

    DataSaver().save(df)

    print("Finished Running Data Cleaning and Feature Engineering")

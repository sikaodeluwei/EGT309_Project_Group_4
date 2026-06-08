import os

import pandas as pd
import numpy as np

import sqlite3


from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier

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
        light_from_time: dict[str, str] = None,
        n_neighbors: int = 5
    ):
        self.light_from_time = light_from_time if light_from_time is not None else {}
        self.n_neighbors = n_neighbors
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors)

        # Initialize the KNN Classifier
        self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Initialize Scalers for each specific pipeline task to avoid leakage
        self.humidity_scaler = RobustScaler()
        self.co_scaler = RobustScaler()

        self.co_features = [
            'CO2_ElectroChemicalSensor', 
            'MetalOxideSensor_Unit1', 
            'MetalOxideSensor_Unit3', 
            'MetalOxideSensor_Unit4'
        ]

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

            # RobustScaler handles NaNs automatically during fit_transform
            scaled_humidity_data = self.humidity_scaler.fit_transform(df[impute_cols])
            
            # Impute on scaled data
            imputed_scaled = self.knn_imputer.fit_transform(scaled_humidity_data)
            
            # Inverse transform to bring back to original scale before saving
            imputed_original = self.humidity_scaler.inverse_transform(imputed_scaled)
            df['humidity'] = imputed_original[:, 1]

        # CO sensor: impute missing categorical labels using KNeighborsClassifier
        # Using 'temperature' and 'humidity' as clean features to predict the missing CO sensor classes
        if 'co_gassensor' in df.columns and all(col in df.columns for col in self.co_features):
            missing_co_mask = df['co_gassensor'].isna()

            if missing_co_mask.any():
                # Verify classifier was successfully trained in fit()
                if hasattr(self.co_scaler, 'mean_'): 
                    X_miss = df.loc[missing_co_mask, self.co_features]
                
                    # Fallback for missing elements in the predictor features themselves
                    X_miss_filled = X_miss.fillna(0) 
                    
                    X_miss_scaled = self.co_scaler.transform(X_miss_filled)
                
                    # Predict the missing values
                    predicted_co = self.knn_classifier.predict(X_miss_scaled)
                    
                    # Assign predictions back to the missing slots
                    df.loc[missing_co_mask, 'co_gassensor'] = predicted_co
                
            # Fallback: If ALL values were missing, fill with a default mode or leave as is
            elif missing_co_mask.all():
                # If there's absolutely no data to train on, fallback to a placeholder
                df['co_gassensor'] = df['co_gassensor'].fillna("Unknown")

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

        # PCA for Metal Oxide Sensors to capture shared patterns while reducing dimensionality
        self.mos_pca = PCA(n_components=1)
        # RobustScaler applied to MOS before PCA — keeps units consistent and suppresses outliers
        self.mos_scaler = RobustScaler()

        # Using RobustScaler for the final pass to protect KNN from outlier distances
        self.final_scaler = RobustScaler()

        # Initialize PCA to extract 1 component representing the shared pattern
        self.is_fitted = False

        # State tracking for One-Hot Encoding, prevents shape mismatches during inference
        self.hvac_categories = None


    def fit(self, df: pd.DataFrame):
        """Fits the PCA and records categorical states from training data."""

        mos_cols = [
            'MetalOxideSensor_Unit1',
            'MetalOxideSensor_Unit2',
            'MetalOxideSensor_Unit3',
            'MetalOxideSensor_Unit4'
        ]

        # --- Metal Oxide Sensor Aggregation ---
        # Fit PCA on the mos_scaler-scaled data
        if all(col in df.columns for col in mos_cols):
            scaled_mos = self.mos_scaler.fit_transform(df[mos_cols])
            self.mos_pca.fit(scaled_mos)

        # Record unique HVAC modes present during training
        if 'hvac_operation_mode' in df.columns:
            self.hvac_categories = df['hvac_operation_mode'].dropna().unique().tolist()

        # Fit the final robust scaler on the completely transformed training set
        temp_df = self._transform_features(df)
        self.final_scaler.fit(temp_df)

        self.is_fitted = True
        return self

    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to apply mappings, OHE, and PCA processing."""
        df = df.copy()

        if 'time_of_day' in df.columns:
            df['time_of_day'] = df['time_of_day'].map(self.time_of_day_map)

        if 'ambient_light_level' in df.columns:
            df['ambient_light_level'] = df['ambient_light_level'].map(
                self.ambient_light_map)

        if 'activity_level' in df.columns:
            df['activity_level'] = df['activity_level'].map(self.activity_level_map)

        # --- One-Hot Encoding for HVAC ---
        if 'hvac_operation_mode' in df.columns:
            # If categories were recorded during fit, align the column to those categories
            # so inference never produces new columns that would break the final scaler
            if self.hvac_categories is not None:
                df['hvac_operation_mode'] = pd.Categorical(
                    df['hvac_operation_mode'], categories=self.hvac_categories
                )

            df = pd.get_dummies(
                df,
                columns=['hvac_operation_mode'],
                prefix='hvac',
                drop_first=True,  # Avoids the dummy variable trap
                dtype=int
            )

        # --- Metal Oxide Sensor Aggregation ---
        mos_cols = [
            'MetalOxideSensor_Unit1',
            'MetalOxideSensor_Unit2',
            'MetalOxideSensor_Unit3',
            'MetalOxideSensor_Unit4'
        ]

        if all(col in df.columns for col in mos_cols):
            # Scale MOS columns before PCA transform, mirroring what was done in fit().
            # prevents fitting of raw values on scaled values that produced
            # meaningless projection at inference time.
            scaled_mos = self.mos_scaler.transform(df[mos_cols])

            # PCA naturally aligns with the direction of maximum variance (the shared shape).
            # Uses fit_transform during the fit() call (is_fitted=False), transform at inference.
            if self.is_fitted:
                df['MetalOxideSensor_Aggregated'] = self.mos_pca.transform(scaled_mos)[:, 0]
            else:
                df['MetalOxideSensor_Aggregated'] = self.mos_pca.fit_transform(scaled_mos)[:, 0]

            # Drop original columns after computing the aggregated feature
            # prevents a KeyError on df[mos_cols].
            df = df.drop(columns=mos_cols)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms features and applies final Robust Scaling for KNN."""
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transforming.")

        transformed_df = self._transform_features(df)

        # Scale final output array using RobustScaler
        scaled_array = self.final_scaler.transform(transformed_df)
        return pd.DataFrame(scaled_array, columns=transformed_df.columns, index=transformed_df.index)

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

    fe = FeatureEngineer()
    fe.fit(df)            
    df = fe.transform(df)

    DataSaver().save(df)

    print("Finished Running Data Cleaning and Feature Engineering")

    DataSaver().save(df)

    print("Finished Running Data Cleaning and Feature Engineering")

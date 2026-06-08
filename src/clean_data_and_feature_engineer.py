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
        # BUG FIX 1: UNIFORMER_ACTIVITY_FIXES now exists in config (it was
        # missing before). The default here correctly references it.
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
        # BUG FIX 2: Default changed from None (→ empty dict, silently doing
        # nothing) to config.IMPUTER_LIGHT_FROM_TIME so ambient light
        # imputation actually runs when DataImputer() is called with no args.
        light_from_time: dict[str, str] = None,
        n_neighbors: int = 5
    ):
        self.light_from_time = light_from_time if light_from_time is not None else {}
        self.n_neighbors = n_neighbors
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors)

        # Initialize the KNN Classifier
        self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Initialize the KNN Imputer for MOS Unit2
        self.mos2_imputer = KNNImputer(n_neighbors=n_neighbors)
        self.mos2_scaler = RobustScaler()

        # Initialize Scalers
        self.humidity_scaler = RobustScaler()
        self.co_scaler = RobustScaler()

        # BUG FIX 3: All column name references changed to lowercase to match
        # the output of DataUniformer (which lowercases every column name).
        # Original code used mixed-case names that would never match post-uniforming.
        self.co_features = [
            'co2_electrochemicalsensor',
            'metaloxidesensor_unit1',
            'metaloxidesensor_unit3',
            'metaloxidesensor_unit4',
        ]

        self.mos2_features = [
            'co2_electrochemicalsensor',
            'co_gassensor',
        ]

        # Fallback values learned from the training set
        self.co_fallback_mode = "Unknown"
        self.is_fitted_ = False

    def fit(self, df: pd.DataFrame, y=None):
        df = df.copy()

        # Fit Humidity Imputation State
        if 'humidity' in df.columns and 'temperature' in df.columns:
            invalid_humidity_mask = (df['humidity'] < 0) | (df['humidity'] > 100)
            df.loc[invalid_humidity_mask, 'humidity'] = np.nan

            impute_cols = ['temperature', 'humidity']
            scaled_humidity_data = self.humidity_scaler.fit_transform(df[impute_cols])
            self.knn_imputer.fit(scaled_humidity_data)

        # BUG FIX 4: MetalOxideSensor_Unit2 imputation was completely absent.
        # Per the observations (~14.1% missing) and preprocessing rationale,
        # it should be imputed using co2_electrochemicalsensor and co_gassensor.
        if 'metaloxidesensor_unit2' in df.columns and all(
            col in df.columns for col in self.mos2_features
        ):
            train_mask = (
                df['metaloxidesensor_unit2'].notna()
                & df[self.mos2_features].notna().all(axis=1)
            )
            if train_mask.any():
                fit_cols = self.mos2_features + ['metaloxidesensor_unit2']
                scaled = self.mos2_scaler.fit_transform(df.loc[train_mask, fit_cols])
                # Fit a separate KNNImputer on the scaled feature+target block
                all_cols_scaled = self.mos2_scaler.fit_transform(df[fit_cols].fillna(0))
                self.mos2_imputer.fit(all_cols_scaled)

        # Fit CO Classifier State
        if 'co_gassensor' in df.columns and all(col in df.columns for col in self.co_features):
            if not df['co_gassensor'].dropna().empty:
                self.co_fallback_mode = df['co_gassensor'].mode()[0]

            train_mask = df['co_gassensor'].notna() & df[self.co_features].notna().all(axis=1)

            if train_mask.any():
                X_train = df.loc[train_mask, self.co_features]
                y_train = df.loc[train_mask, 'co_gassensor']

                X_train_scaled = self.co_scaler.fit_transform(X_train)
                self.knn_classifier.fit(X_train_scaled, y_train)

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("DataImputer must be fitted before transforming data.")

        df = df.copy()

        # Ambient light level: impute from time of day, then fallback
        if 'ambient_light_level' in df.columns and 'time_of_day' in df.columns:
            df['ambient_light_level'] = df['ambient_light_level'].fillna(
                df['time_of_day'].map(self.light_from_time)
            )
            df['ambient_light_level'] = df['ambient_light_level'].fillna(np.nan)

        # Humidity: impute out-of-bounds values using Temperature via KNN
        if 'humidity' in df.columns and 'temperature' in df.columns:
            invalid_humidity_mask = (df['humidity'] < 0) | (df['humidity'] > 100)
            df.loc[invalid_humidity_mask, 'humidity'] = np.nan

            impute_cols = ['temperature', 'humidity']
            scaled_humidity_data = self.humidity_scaler.transform(df[impute_cols])
            imputed_scaled = self.knn_imputer.transform(scaled_humidity_data)
            imputed_original = self.humidity_scaler.inverse_transform(imputed_scaled)
            df['humidity'] = imputed_original[:, 1]

        # BUG FIX 5: MetalOxideSensor_Unit2 imputation (missing in original).
        if 'metaloxidesensor_unit2' in df.columns and all(
            col in df.columns for col in self.mos2_features
        ):
            missing_mos2_mask = df['metaloxidesensor_unit2'].isna()
            if missing_mos2_mask.any() and hasattr(self.mos2_scaler, 'center_'):
                fit_cols = self.mos2_features + ['metaloxidesensor_unit2']
                X_mos2 = df[fit_cols].copy()
                X_mos2_scaled = self.mos2_scaler.transform(X_mos2.fillna(0))
                imputed_mos2_scaled = self.mos2_imputer.transform(X_mos2_scaled)
                imputed_mos2_original = self.mos2_scaler.inverse_transform(imputed_mos2_scaled)
                # Only write back the imputed Unit2 column (last column)
                df.loc[missing_mos2_mask, 'metaloxidesensor_unit2'] = (
                    imputed_mos2_original[missing_mos2_mask, -1]
                )

        # CO sensor: impute missing categorical labels using KNeighborsClassifier
        if 'co_gassensor' in df.columns and all(col in df.columns for col in self.co_features):
            missing_co_mask = df['co_gassensor'].isna()

            if missing_co_mask.any():
                if hasattr(self.co_scaler, 'center_'):
                    X_miss = df.loc[missing_co_mask, self.co_features]
                    X_miss_filled = X_miss.fillna(0)

                    X_miss_scaled = self.co_scaler.transform(X_miss_filled)
                    predicted_co = self.knn_classifier.predict(X_miss_scaled)
                    df.loc[missing_co_mask, 'co_gassensor'] = predicted_co

            df['co_gassensor'] = df['co_gassensor'].fillna(self.co_fallback_mode)

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

        # BUG FIX 6: CO2_InfraredSensor negative values (113 rows per observations)
        # are physically impossible and must be treated as anomalies. Clamp to NaN
        # so they are removed by the subsequent dropna() call rather than silently
        # corrupting downstream feature engineering.
        if 'co2_infraredsensor' in df.columns:
            df.loc[df['co2_infraredsensor'] < 0, 'co2_infraredsensor'] = np.nan

        # Replace sentinel value with NaN, then drop remaining nulls
        df['ambient_light_level'] = df['ambient_light_level'].replace(
            self.light_sentinel, np.nan
        )

        df = df.dropna()
        df = df.drop_duplicates()

        return df


class FeatureEngineer:
    """Encodes categorical features as integers."""

    def __init__(
        self,
        time_of_day_map: dict[str, int],
        ambient_light_map: dict[str, int],
        activity_level_map: dict[str, int],
    ):
        self.time_of_day_map = time_of_day_map
        self.ambient_light_map = ambient_light_map
        self.activity_level_map = activity_level_map

        # Scalers & Dimensionality Reduction
        self.mos_scaler = RobustScaler()
        self.mos_pca = PCA(n_components=1)
        self.co2_scaler = RobustScaler()
        self.co2_pca = PCA(n_components=1)
        self.final_scaler = RobustScaler()

        # State tracking
        self.is_fitted = False
        self.hvac_categories = None
        self.final_column_schema = None

        # BUG FIX 7: All column name references lowercased to match DataUniformer output.
        self.mos_cols = [
            'metaloxidesensor_unit1',
            'metaloxidesensor_unit2',
            'metaloxidesensor_unit3',
            'metaloxidesensor_unit4',
        ]

        self.co2_cols = [
            'co2_infraredsensor',
            'co2_electrochemicalsensor',
        ]

    def fit(self, df: pd.DataFrame):
        """Fits the PCA and records categorical states from training data."""
        df = df.copy()

        # BUG FIX 8: hvac_categories must be captured BEFORE _transform_features()
        # is called. In the original code self.hvac_categories was still None when
        # _transform_features() ran during fit(), so pd.Categorical was never used
        # and the OHE column set was non-deterministic across train/inference calls.
        if 'hvac_operation_mode' in df.columns:
            self.hvac_categories = sorted(df['hvac_operation_mode'].dropna().unique().tolist())

        # Fit the MOS Scaler and PCA
        if all(col in df.columns for col in self.mos_cols):
            scaled_mos = self.mos_scaler.fit_transform(df[self.mos_cols])
            self.mos_pca.fit(scaled_mos)

        # Fit the CO2 Scaler and PCA
        if all(col in df.columns for col in self.co2_cols):
            scaled_co2 = self.co2_scaler.fit_transform(df[self.co2_cols])
            self.co2_pca.fit(scaled_co2)

        self.is_fitted = True

        temp_df = self._transform_features(df)

        self.final_column_schema = temp_df.columns.tolist()
        self.final_scaler.fit(temp_df)

        return self

    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to apply mappings, OHE, and PCA processing."""
        df = df.copy()

        # Apply Categorical Mappings
        if 'time_of_day' in df.columns:
            df['time_of_day'] = df['time_of_day'].map(self.time_of_day_map)
        if 'ambient_light_level' in df.columns:
            df['ambient_light_level'] = df['ambient_light_level'].map(self.ambient_light_map)
        if 'activity_level' in df.columns:
            df['activity_level'] = df['activity_level'].map(self.activity_level_map)

        # One-Hot Encoding for HVAC
        if 'hvac_operation_mode' in df.columns:
            if self.hvac_categories is not None:
                df['hvac_operation_mode'] = pd.Categorical(
                    df['hvac_operation_mode'], categories=self.hvac_categories
                )

            df = pd.get_dummies(
                df,
                columns=['hvac_operation_mode'],
                prefix='hvac',
                drop_first=True,
                dtype=int
            )
        else:
            if self.hvac_categories is not None:
                for cat in self.hvac_categories[1:]:
                    df[f'hvac_{cat}'] = 0

        # Metal Oxide Sensor Aggregation via PCA
        if all(col in df.columns for col in self.mos_cols):
            scaled_mos = self.mos_scaler.transform(df[self.mos_cols])
            df['metaloxidesensor_aggregated'] = self.mos_pca.transform(scaled_mos)[:, 0]
            df = df.drop(columns=self.mos_cols)

        # BUG FIX 9: CO2 sensor aggregation was described in the preprocessing
        # rationale but never implemented. Both sensors share ~95% cosine similarity,
        # so they are aggregated into a single PCA component here — matching the
        # same pattern used for the metal oxide sensors.
        if all(col in df.columns for col in self.co2_cols):
            scaled_co2 = self.co2_scaler.transform(df[self.co2_cols])
            df['co2_aggregated'] = self.co2_pca.transform(scaled_co2)[:, 0]
            df = df.drop(columns=self.co2_cols)

        # Align columns explicitly to what was seen during fit
        if self.final_column_schema is not None:
            df = df.reindex(columns=self.final_column_schema, fill_value=0)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms features and applies final Robust Scaling for KNN."""
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transforming.")

        transformed_df = self._transform_features(df)

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

    # Apply only the normalisation part of DataUniformer (no fixes yet)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df = df.map(lambda x: x.lower().replace(' ', '_') if isinstance(x, str) else x)

    print(df['activity_level'].value_counts())

    df = DataUniformer().transform(df)

    # BUG FIX 10: DataImputer() was previously called with no arguments, so
    # light_from_time defaulted to {} and ambient light imputation silently did
    # nothing. Now passes config.IMPUTER_LIGHT_FROM_TIME explicitly.
    df = DataImputer(light_from_time=config.IMPUTER_LIGHT_FROM_TIME).fit(df).transform(df)

    df = DataCleaner().transform(df)

    print(df.columns.tolist())

    # Split into features and target
    X = df.drop(columns=['co_gassensor'])
    y = df['co_gassensor']

    # Fit and transform only the features
    fe = FeatureEngineer(
        time_of_day_map=config.FE_TIME_OF_DAY_MAP,
        ambient_light_map=config.FE_AMBIENT_LIGHT_MAP,
        activity_level_map=config.FE_ACTIVITY_LEVEL_MAP,
    )
    fe.fit(X)
    X_transformed = fe.transform(X)

    # Recombine and save
    df_cleaned = X_transformed.copy()
    df_cleaned['co_gassensor'] = y.values

    DataSaver().save(df_cleaned)

    print("Finished Running Data Cleaning and Feature Engineering")
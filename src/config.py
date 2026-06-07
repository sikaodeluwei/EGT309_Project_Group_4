"""
config.py

Stores user-adjustable settings for the ML pipeline.
This file is not meant to be run directly.
Other files such as base_model_v2.py import settings from here.
"""

import os

# -----------------------------
# DataLoader
# -----------------------------

LOADER_DB_PATH: str = "gas_monitoring.db"
LOADER_QUERY: str = "SELECT * FROM gas_monitoring;"

# -----------------------------
# DataUniformer
# -----------------------------

# Columns that must be cast to str before normalisation
UNIFORMER_CATEGORICAL_COLS: list[str] = [
    "Time of Day",
    "HVAC Operation Mode",
    "Ambient Light Level",
    "Activity Level",
]

# -----------------------------
# DataImputer
# -----------------------------

# Fallback ambient-light value inferred from time-of-day when light is missing
IMPUTER_LIGHT_FROM_TIME: dict[str, str] = {
    "morning": "dim",
    "afternoon": "very_bright",
    "night": "dark",
}

# Known label inconsistencies in activity_level: {raw_value: corrected_value}
UNIFORMER_ACTIVITY_FIXES: dict[str, str] = {
    "lowactivity": "low_activity",
    "moderateactivity": "moderate_activity",
}

# -----------------------------
# DataCleaner
# -----------------------------

# Desired column order in the cleaned dataset (unlisted columns are dropped)
CLEANER_COLUMN_ORDER: list[str] = [
    "co_gassensor",
    "co2_infraredsensor",
    "co2_electrochemicalsensor",
    "metaloxidesensor_unit1",
    "metaloxidesensor_unit2",
    "metaloxidesensor_unit3",
    "metaloxidesensor_unit4",
    "temperature",
    "humidity",
    "ambient_light_level",
    "time_of_day",
    "hvac_operation_mode",
    "activity_level",
]

# Temperatures above this threshold (K) are converted to Celsius (value − 273.15)
CLEANER_KELVIN_THRESHOLD: float = 150.0

# Sentinel string in ambient_light_level that is treated as NaN
CLEANER_LIGHT_SENTINEL: str = "none"

# -----------------------------
# FeatureEngineer
# -----------------------------

FEATURE_TIME_OF_DAY_MAP: dict[str, int] = {
    "morning": 0,
    "afternoon": 1,
    "evening": 2,
    "night": 3,
}

FEATURE_AMBIENT_LIGHT_MAP: dict[str, int] = {
    "very_dim": 0,
    "dim": 1,
    "moderate": 2,
    "bright": 3,
    "very_bright": 4,
}

FEATURE_ACTIVITY_LEVEL_MAP: dict[str, int] = {
    "low_activity": 0,
    "moderate_activity": 1,
    "high_activity": 2,
}

# ---------------------------------------------------------------------------
# DataSaver
# ---------------------------------------------------------------------------

SAVER_DB_PATH: str = "data/gas_monitoring_cleanedv1.db"
SAVER_TABLE_NAME: str = "cleaned_data"

# -----------------------------
# Data settings
# -----------------------------

DB_PATH = "data/gas_monitoring.db"
TABLE_NAME = "cleaned_gas_data"
TARGET_COLUMN = "activity level"

TEST_SIZE = 0.2
RANDOM_STATE = 42
SCORING_METRIC = "weighted_f1_score"


# -----------------------------
# Output settings
# -----------------------------

RESULTS_DIR = "saved_model"
RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, "basic_model_comparison_results.csv")
BEST_MODEL_NAMES_PATH = os.path.join(RESULTS_DIR, "best_3_model_names.csv")


# -----------------------------
# Candidate models for baseline comparison
# base_model_v2.py will test these models first.
# At this stage, we do not know the top 3 yet.
# -----------------------------

SELECTED_MODELS = [
    "Logistic Regression",
    "K-Nearest Neighbours",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "Extra Trees",
    "Support Vector Machine",
    "Naive Bayes"
]


# -----------------------------
# Base model parameters
# These are used for the first baseline comparison.
# -----------------------------

BASE_MODEL_PARAMS = {
    "Logistic Regression": {
        "max_iter": 1000
    },

    "K-Nearest Neighbours": {},

    "Decision Tree": {
        "random_state": RANDOM_STATE
    },

    "Random Forest": {
        "random_state": RANDOM_STATE
    },

    "Gradient Boosting": {
        "random_state": RANDOM_STATE
    },

    "Extra Trees": {
        "random_state": RANDOM_STATE
    },

    "Support Vector Machine": {},

    "Naive Bayes": {}
}


# -----------------------------
# Hyperparameter grids for next member
# Evaluation/tuning file will read the top 3 model names from:
# saved_model/best_3_model_names.csv
# Then it will use the matching grid below.
# -----------------------------

TUNING_PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },

    "Extra Trees": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },

    "Gradient Boosting": {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5]
    },

    "Decision Tree": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },

    "K-Nearest Neighbours": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"]
    },

    "Support Vector Machine": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    },

    "Logistic Regression": {
        "C": [0.1, 1, 10],
        "max_iter": [1000]
    },

    "Naive Bayes": {}
}
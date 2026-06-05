"""
config.py

Stores user-adjustable settings for the ML pipeline.
This file is not meant to be run directly.
Other files such as base_model_v2.py import settings from here.
"""

import os


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
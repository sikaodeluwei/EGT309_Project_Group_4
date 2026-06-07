"""
basic_models.py

This module trains multiple baseline classification models to predict
the elderly resident's activity level using the cleaned gas monitoring dataset.
It handles text-uniformity processing to collapse target label typos dynamically.
"""

import os
import sys
import sqlite3
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Enforce path awareness
project_root = r"C:\Users\aadar\EGT309_Project_Group_4"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import (
    DB_PATH,
    TABLE_NAME,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    SCORING_METRIC,
    RESULTS_DIR,
    RESULTS_CSV_PATH,
    BEST_MODEL_NAMES_PATH,
    SELECTED_MODELS,
    BASE_MODEL_PARAMS
)


class BasicModelTrainer:
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        self.target_column = TARGET_COLUMN

        self.all_models = self.build_model_library()
        self.models = self.get_selected_models()

    def build_model_library(self):
        return {
            "Logistic Regression": LogisticRegression(**BASE_MODEL_PARAMS["Logistic Regression"]),
            "K-Nearest Neighbours": KNeighborsClassifier(**BASE_MODEL_PARAMS["K-Nearest Neighbours"]),
            "Decision Tree": DecisionTreeClassifier(**BASE_MODEL_PARAMS["Decision Tree"]),
            "Random Forest": RandomForestClassifier(**BASE_MODEL_PARAMS["Random Forest"]),
            "Gradient Boosting": GradientBoostingClassifier(**BASE_MODEL_PARAMS["Gradient Boosting"]),
            "Extra Trees": ExtraTreesClassifier(**BASE_MODEL_PARAMS["Extra Trees"]),
            "Support Vector Machine": SVC(**BASE_MODEL_PARAMS["Support Vector Machine"]),
            "Naive Bayes": GaussianNB(**BASE_MODEL_PARAMS["Naive Bayes"])
        }

    def get_selected_models(self):
        selected_models = {}
        for model_name in SELECTED_MODELS:
            if model_name in self.all_models:
                selected_models[model_name] = self.all_models[model_name]
        return selected_models

    def load_cleaned_data(self):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()
        print(f"Loaded cleaned dataset from table '{TABLE_NAME}' with shape: {df.shape}")
        return df

    def prepare_features(self, df):
        df = df.copy()

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found.")

        # =========================================================================
        # TARGET COLUMN CLEANING: Merge spelling mistakes and formatting typos
        # =========================================================================
        df[self.target_column] = df[self.target_column].astype(str).str.replace("_", " ")
        df[self.target_column] = df[self.target_column].astype(str).str.replace("LowActivity", "Low Activity")
        df[self.target_column] = df[self.target_column].astype(str).str.replace("ModerateActivity", "Moderate Activity")
        df[self.target_column] = df[self.target_column].str.strip()

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        valid_rows = y.notna() & (y != "None") & (y != "nan")
        X = X[valid_rows]
        y = y[valid_rows]

        if "session id" in X.columns:
            X = X.drop(columns=["session id"])

        for col in X.columns:
            converted_col = pd.to_numeric(X[col], errors="coerce")
            if converted_col.notna().sum() > 0:
                X[col] = converted_col

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                median_value = X[col].median()
                X[col] = X[col].fillna(0 if pd.isna(median_value) else median_value)
            else:
                X[col] = X[col].fillna("unknown")

        for col in X.select_dtypes(include=["object", "string"]).columns:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col].astype(str))

        y = self.target_encoder.fit_transform(y.astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=self.random_state, stratify=y
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def train_and_compare_models(self, X_train, X_test, y_train, y_test):
        results = {}
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")
            results[model_name] = {
                "model": model,
                "accuracy": accuracy,
                "weighted_f1_score": weighted_f1
            }
        return dict(sorted(results.items(), key=lambda item: item[1][SCORING_METRIC], reverse=True))

    def select_best_3_models(self, sorted_results):
        return dict(list(sorted_results.items())[:3])

    def save_results_to_csv(self, sorted_results):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        rows = [{"model_name": name, "accuracy": res["accuracy"], "weighted_f1_score": res["weighted_f1_score"]} 
                for name, res in sorted_results.items()]
        pd.DataFrame(rows).to_csv(RESULTS_CSV_PATH, index=False)

    def save_best_3_models(self, best_3_models):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        # Save structural names as a fallback path
        rows = [{"model_name": name, "accuracy": res["accuracy"], "weighted_f1_score": res["weighted_f1_score"]} 
                for name, res in best_3_models.items()]
        pd.DataFrame(rows).to_csv(BEST_MODEL_NAMES_PATH, index=False)

    def run(self):
        df = self.load_cleaned_data()
        X_train, X_test, y_train, y_test = self.prepare_features(df)
        all_results = self.train_and_compare_models(X_train, X_test, y_train, y_test)
        best_3_models = self.select_best_3_models(all_results)
        self.save_results_to_csv(all_results)
        self.save_best_3_models(best_3_models)
        return all_results, best_3_models


if __name__ == "__main__":
    trainer = BasicModelTrainer()
    trainer.run()
"""
basic_models.py

This module trains multiple baseline classification models to predict
the elderly resident's activity level using the cleaned gas monitoring dataset.

It reads user-adjustable settings from config.py, such as:
- database path
- cleaned table name
- target column
- selected candidate models
- base model parameters
- output file paths

Main purpose:
1. Load cleaned data
2. Prepare features and target
3. Train selected candidate models
4. Compare models using accuracy and weighted F1-score
5. Select the best 3 models
6. Save all results and best 3 model names for the evaluation member
"""

import os
import sqlite3
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

from config import (
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
    """
    Trains and compares multiple baseline classification models.

    The class tests selected candidate models from config.py, ranks them by
    weighted F1-score, and saves the best 3 model names for later tuning
    and evaluation.
    """

    def __init__(self, random_state=RANDOM_STATE):
        """
        Initialise the trainer with reproducible settings.
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()

        self.all_models = self.build_model_library()
        self.models = self.get_selected_models()

    def build_model_library(self):
        """
        Build all available baseline models using parameters from config.py.

        Returns:
            dict: Dictionary containing all supported model objects.
        """
        return {
            "Logistic Regression": LogisticRegression(
                **BASE_MODEL_PARAMS["Logistic Regression"]
            ),

            "K-Nearest Neighbours": KNeighborsClassifier(
                **BASE_MODEL_PARAMS["K-Nearest Neighbours"]
            ),

            "Decision Tree": DecisionTreeClassifier(
                **BASE_MODEL_PARAMS["Decision Tree"]
            ),

            "Random Forest": RandomForestClassifier(
                **BASE_MODEL_PARAMS["Random Forest"]
            ),

            "Gradient Boosting": GradientBoostingClassifier(
                **BASE_MODEL_PARAMS["Gradient Boosting"]
            ),

            "Extra Trees": ExtraTreesClassifier(
                **BASE_MODEL_PARAMS["Extra Trees"]
            ),

            "Support Vector Machine": SVC(
                **BASE_MODEL_PARAMS["Support Vector Machine"]
            ),

            "Naive Bayes": GaussianNB(
                **BASE_MODEL_PARAMS["Naive Bayes"]
            )
        }

    def get_selected_models(self):
        """
        Select the candidate models listed in config.py.

        Returns:
            dict: Dictionary of selected model names and model objects.
        """
        selected_models = {}

        for model_name in SELECTED_MODELS:
            if model_name not in self.all_models:
                raise ValueError(
                    f"Model '{model_name}' is not supported. "
                    f"Available models: {list(self.all_models.keys())}"
                )

            selected_models[model_name] = self.all_models[model_name]

        return selected_models

    def load_cleaned_data(self):
        """
        Load the cleaned dataset from SQLite database.

        Returns:
            pd.DataFrame: Cleaned gas monitoring dataset.
        """
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()

        print(f"Loaded cleaned dataset from table '{TABLE_NAME}'")
        print(f"Dataset shape: {df.shape}")

        return df

    def prepare_features(self, df):
        """
        Prepare the cleaned dataset for model training.

        Steps:
        1. Separate features X and target y
        2. Remove rows with missing target values
        3. Drop session ID because it is only an identifier
        4. Convert numeric-looking columns into numeric values
        5. Fill remaining missing values
        6. Encode categorical columns
        7. Encode target labels
        8. Split into training and testing sets
        9. Scale features

        Args:
            df (pd.DataFrame): Cleaned dataset.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        df = df.copy()

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        # Rows with missing target cannot be used for supervised learning
        valid_rows = y.notna()
        X = X[valid_rows]
        y = y[valid_rows]

        # Session ID is an identifier, not a useful prediction feature
        if "session id" in X.columns:
            X = X.drop(columns=["session id"])

        # Convert numeric-looking columns that may have been read as text
        for col in X.columns:
            converted_col = pd.to_numeric(X[col], errors="coerce")

            # Convert only if at least some values are numeric
            if converted_col.notna().sum() > 0:
                X[col] = converted_col

        # Fill missing values before model training
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                median_value = X[col].median()

                if pd.isna(median_value):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_value)
            else:
                X[col] = X[col].fillna("unknown")

        # Encode categorical input features
        for col in X.select_dtypes(include=["object", "string"]).columns:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col].astype(str))

        # Encode target labels
        y = self.target_encoder.fit_transform(y.astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=self.random_state,
            stratify=y
        )

        # Scaling helps models such as KNN, SVM, and Logistic Regression.
        # Fit only on training data to avoid data leakage.
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def train_and_compare_models(self, X_train, X_test, y_train, y_test):
        """
        Train all selected baseline models and compare performance.

        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target

        Returns:
            dict: Model results sorted by selected scoring metric.
        """
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

        sorted_results = dict(
            sorted(
                results.items(),
                key=lambda item: item[1][SCORING_METRIC],
                reverse=True
            )
        )

        return sorted_results

    def select_best_3_models(self, sorted_results):
        """
        Select the best 3 models based on the scoring metric.

        Args:
            sorted_results (dict): Sorted model comparison results.

        Returns:
            dict: Best 3 model results.
        """
        return dict(list(sorted_results.items())[:3])

    def save_results_to_csv(self, sorted_results):
        """
        Save comparison results for all candidate models to CSV.
        """
        os.makedirs(RESULTS_DIR, exist_ok=True)

        rows = []

        for model_name, result in sorted_results.items():
            rows.append({
                "model_name": model_name,
                "accuracy": result["accuracy"],
                "weighted_f1_score": result["weighted_f1_score"]
            })

        results_df = pd.DataFrame(rows)
        results_df.to_csv(RESULTS_CSV_PATH, index=False)

        print(f"\nAll model comparison results saved to: {RESULTS_CSV_PATH}")

    def save_best_3_model_names(self, best_3_models):
        """
        Save only the best 3 model names and scores.

        The trained model objects are not saved here because they can be large.
        The next member can read this CSV and retrain/tune these selected models.
        """
        os.makedirs(RESULTS_DIR, exist_ok=True)

        rows = []

        for model_name, result in best_3_models.items():
            rows.append({
                "model_name": model_name,
                "accuracy": result["accuracy"],
                "weighted_f1_score": result["weighted_f1_score"]
            })

        best_models_df = pd.DataFrame(rows)
        best_models_df.to_csv(BEST_MODEL_NAMES_PATH, index=False)

        print(f"Best 3 model names saved to: {BEST_MODEL_NAMES_PATH}")

    def print_results(self, sorted_results, best_3_models):
        """
        Print all model results and selected best 3 models.
        """
        print("\nModel Comparison Results:")
        print("-" * 70)

        for model_name, result in sorted_results.items():
            print(
                f"{model_name}: "
                f"Accuracy = {result['accuracy']:.4f}, "
                f"Weighted F1 = {result['weighted_f1_score']:.4f}"
            )

        print("\nBest 3 Models Selected:")
        print("-" * 70)

        for model_name, result in best_3_models.items():
            print(
                f"{model_name}: "
                f"Accuracy = {result['accuracy']:.4f}, "
                f"Weighted F1 = {result['weighted_f1_score']:.4f}"
            )

    def run(self):
        """
        Run the full baseline model training and selection workflow.

        Returns:
            tuple: all_results, best_3_models
        """
        df = self.load_cleaned_data()

        X_train, X_test, y_train, y_test = self.prepare_features(df)

        all_results = self.train_and_compare_models(
            X_train,
            X_test,
            y_train,
            y_test
        )

        best_3_models = self.select_best_3_models(all_results)

        self.print_results(all_results, best_3_models)
        self.save_results_to_csv(all_results)
        self.save_best_3_model_names(best_3_models)

        return all_results, best_3_models


if __name__ == "__main__":
    trainer = BasicModelTrainer()
    trainer.run()
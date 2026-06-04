"""
basic_models.py

This module trains multiple baseline classification models to predict
the elderly resident's activity level using the cleaned gas monitoring dataset.

Main responsibilities:
1. Load cleaned data from SQLite database
2. Prepare features and target for machine learning
3. Train multiple basic classification models
4. Compare models using Accuracy and Weighted F1-score
5. Select and save the best 3 models for further evaluation
"""

import os
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


# -----------------------------
# Configuration
# -----------------------------

DB_PATH = "data/gas_monitoring.db"
TABLE_NAME = "cleaned_gas_data"
TARGET_COLUMN = "activity level"

TEST_SIZE = 0.2
RANDOM_STATE = 42
SCORING_METRIC = "weighted_f1_score"

RESULTS_DIR = "saved_model"
RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, "basic_model_comparison_results.csv")
BEST_MODELS_PATH = os.path.join(RESULTS_DIR, "best_3_basic_models.pkl")


class BasicModelTrainer:
    """
    Trains and compares multiple baseline classification models.

    The purpose of this class is to test several suitable machine learning
    algorithms on the same cleaned dataset, then select the best 3 models
    based on weighted F1-score.
    """

    def __init__(self, random_state=RANDOM_STATE):
        """
        Initialise the trainer with a fixed random state for reproducible results.
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()

        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "K-Nearest Neighbours": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Random Forest": RandomForestClassifier(random_state=random_state, class_weight="balanced", criterion="entropy"),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state, subsample=0.8, max_depth=4),
            "Extra Trees": ExtraTreesClassifier(random_state=random_state, class_weight="balanced", criterion="entropy"),
            "Support Vector Machine": SVC(),
            "Naive Bayes": GaussianNB()
        }

    def load_cleaned_data(self):
        """
        Load the cleaned dataset from the SQLite database.

        Returns:
            pd.DataFrame: Cleaned gas monitoring dataset.
        """
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()

        print(f"Loaded cleaned dataset with shape: {df.shape}")
        return df

    def prepare_features(self, df):
        """
        Prepare the cleaned dataset for model training.

        Steps:
        1. Separate features and target
        2. Remove rows with missing target values
        3. Drop session ID because it is only an identifier
        4. Convert numeric-looking columns into numeric values
        5. Fill remaining missing values
        6. Encode categorical columns
        7. Encode target labels
        8. Split into train and test sets
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

        # Session ID is an identifier, not a meaningful predictive feature
        if "session id" in X.columns:
            X = X.drop(columns=["session id"])

        # Convert columns that look numeric but were read as text from SQLite
        for col in X.columns:
            converted_col = pd.to_numeric(X[col], errors="coerce")

            # Convert only if at least some values are valid numbers
            if converted_col.notna().sum() > 0:
                X[col] = converted_col

        # Fill missing values
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                median_value = X[col].median()

                # If a numeric column is completely empty, fill with 0
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

        # Encode target activity labels into numbers
        y = self.target_encoder.fit_transform(y.astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=self.random_state,
            stratify=y
        )

        # Scaling is useful for KNN, SVM, and Logistic Regression.
        # Fit only on training data to avoid data leakage.
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def train_and_compare_models(self, X_train, X_test, y_train, y_test):
        """
        Train all baseline models and compare their performance.

        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target

        Returns:
            dict: Model results sorted by weighted F1-score.
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
        Select the best 3 models based on weighted F1-score.

        Args:
            sorted_results (dict): Sorted model comparison results.

        Returns:
            dict: Best 3 models with their scores.
        """
        return dict(list(sorted_results.items())[:3])

    def save_results_to_csv(self, sorted_results):
        """
        Save model comparison results to a CSV file.

        This allows the evaluation member to use the results later without
        rerunning the full model training process.
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

        print(f"\nModel comparison results saved to: {RESULTS_CSV_PATH}")

    def save_best_3_models(self, best_3_models):
        """
        Save the best 3 trained models as a pickle file.

        This makes the selected models reusable for the evaluation stage.
        """
        os.makedirs(RESULTS_DIR, exist_ok=True)

        with open(BEST_MODELS_PATH, "wb") as file:
            pickle.dump(best_3_models, file)

        print(f"Best 3 models saved to: {BEST_MODELS_PATH}")

    def print_results(self, sorted_results, best_3_models):
        """
        Print all model results and the selected best 3 models.
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
        Run the full basic model training process.

        Returns:
            tuple: all model results and best 3 selected models.
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
        self.save_best_3_models(best_3_models)

        return all_results, best_3_models


if __name__ == "__main__":
    trainer = BasicModelTrainer()
    trainer.run()

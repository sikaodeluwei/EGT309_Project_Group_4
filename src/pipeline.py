"""
pipeline.py

Full consolidated end-to-end ML pipeline for EGT309 Project Group 4.
Integrates advanced data transformation, baseline estimator tracking, 
and automated hyperparameter optimization into a clean OOP architecture.
"""

import os
import sys
import pickle
import logging
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# ─── IMPORT CUSTOM TRANSFORMERS ────────────────────────────────────────────
try:
    from clean_data_and_feature_engineer import (
        DataLoader,
        DataUniformer,
        DataImputer,
        DataCleaner,
        FeatureEngineer,
        DataSaver,
    )
except ImportError:
    print("CRITICAL: Could not find 'clean_data_and_feature_engineer.py'.")
    print("Please ensure it is located in the same working directory.")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from clean_data_and_feature_engineer import (
        DataLoader, DataUniformer, DataImputer, DataCleaner, FeatureEngineer, DataSaver
    )

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _banner(title: str) -> None:
    print(f"\n{'='*64}\n  {title}\n{'='*64}")


# ---------------------------------------------------------------------------
# ─── CENTRALIZED CONFIGURATION ─────────────────────────────────────────────
# ---------------------------------------------------------------------------
class ProjectConfig:
    RAW_DATA_PATH: str = "data/raw_data.xlsx"
    RESULTS_DIR: str = "saved_model"
    RESULTS_CSV_PATH: str = "saved_model/baseline_model_comparison.csv"
    BEST_MODEL_NAMES_PATH: str = "saved_model/best_3_models.csv"
    TUNED_MODEL_PKL_PATH: str = "saved_model/tuned_best_models.pkl"
    TUNED_METRICS_CSV_PATH: str = "saved_model/tuned_metrics_summary.csv"
    
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    SCORING_METRIC: str = "weighted_f1_score"
    TARGET_COLUMN: str = "co_gassensor"
   
    # Target Cleanup Dictionary Mapping (from tuning.py standard)
    UNIFORMER_ACTIVITY_FIXES: dict = {
        "high activity": "High Activity",
        "low activity": "Low Activity",
        "moderate activity": "Moderate Activity"
    }

    IMPUTER_LIGHT_FROM_TIME: dict = {
        "morning": "dim",
        "afternoon": "very_bright",
        "evening": "moderate",
        "night": "dark",
    }
    FE_TIME_OF_DAY_MAP: dict = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}
    FE_AMBIENT_LIGHT_MAP: dict = {
        "very_dim": 0, "dim": 1, "moderate": 2, "bright": 3, "very_bright": 4
    }
    FE_ACTIVITY_LEVEL_MAP: dict = {
        "low_activity": 0, "moderate_activity": 1, "high_activity": 2
    }
   
    SELECTED_MODELS: list = [
        "Logistic Regression",
        "K-Nearest Neighbours",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees",
        "Support Vector Machine",
        "Naive Bayes"
    ]


config = ProjectConfig()

BASE_MODEL_PARAMS: dict = {
    "Logistic Regression":      {"max_iter": 1000},
    "K-Nearest Neighbours":     {},
    "Decision Tree":            {"random_state": config.RANDOM_STATE},
    "Random Forest":            {"random_state": config.RANDOM_STATE},
    "Gradient Boosting":        {"random_state": config.RANDOM_STATE},
    "Extra Trees":              {"random_state": config.RANDOM_STATE},
    "Support Vector Machine":   {},
    "Naive Bayes":              {},
}

TUNING_GRIDS: dict = {
    "Random Forest": {
        "n_estimators": [300, 400],
        "max_depth": [20, 30, None],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"],
    },
    "Extra Trees": {
        "n_estimators": [200, 300],
        "max_depth": [20, 30, None],
        "min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 7],
        "n_estimators": [200, 300],
    },
}

CLASS_MAPPING: list = ["High Activity", "Low Activity", "Moderate Activity"]


# ===========================================================================
# ─── MODELING CLASSES ──────────────────────────────────────────────────────
# ===========================================================================

class BasicModelTrainer:
    """
    Handles robust pipeline data preparation, feature splitting, stratified scaling,
    and trains/ranks baseline estimators using global configurations.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        self.models = self._build_selected_models()
        self.feature_names: List[str] = []

    def _build_selected_models(self) -> Dict[str, Any]:
        library = {
            "Logistic Regression":    LogisticRegression(**BASE_MODEL_PARAMS["Logistic Regression"]),
            "K-Nearest Neighbours":   KNeighborsClassifier(**BASE_MODEL_PARAMS["K-Nearest Neighbours"]),
            "Decision Tree":          DecisionTreeClassifier(**BASE_MODEL_PARAMS["Decision Tree"]),
            "Random Forest":          RandomForestClassifier(**BASE_MODEL_PARAMS["Random Forest"]),
            "Gradient Boosting":      GradientBoostingClassifier(**BASE_MODEL_PARAMS["Gradient Boosting"]),
            "Extra Trees":            ExtraTreesClassifier(**BASE_MODEL_PARAMS["Extra Trees"]),
            "Support Vector Machine": SVC(**BASE_MODEL_PARAMS["Support Vector Machine"]),
            "Naive Bayes":            GaussianNB(**BASE_MODEL_PARAMS["Naive Bayes"]),
        }
       
        selected = {}
        for name in config.SELECTED_MODELS:
            if name in library:
                selected[name] = library[name]
            else:
                raise ValueError(f"Model '{name}' is not recognized in the base model library.")
        return selected

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df = df.copy()

        if config.TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in DataFrame.")

        # Uniform target cleanup logic intercepted from tuning.py rules
        df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(str).str.replace("_", " ")
        df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map(config.UNIFORMER_ACTIVITY_FIXES).fillna(df[config.TARGET_COLUMN])
        df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].str.strip()

        X = df.drop(columns=[config.TARGET_COLUMN])
        y = df[config.TARGET_COLUMN]

        valid_rows = y.notna()
        X = X[valid_rows]
        y = y[valid_rows]

        for drop_col in ["session_id", "session id"]:
            if drop_col in X.columns:
                X = X.drop(columns=[drop_col])

        for col in X.columns:
            converted_col = pd.to_numeric(X[col], errors="coerce")
            if converted_col.notna().sum() > 0:
                X[col] = converted_col

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                median_val = X[col].median()
                X[col] = X[col].fillna(0 if pd.isna(median_val) else median_val)
            else:
                X[col] = X[col].fillna("unknown")

        for col in X.select_dtypes(include=["object", "string"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        y_encoded = self.target_encoder.fit_transform(y.astype(str))
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y_encoded,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y_encoded
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train_and_compare(self, X_train, X_test, y_train, y_test) -> Dict[str, dict]:
        results = {}
        for name, model in self.models.items():
            log.info("Training %s ...", name)
            t0 = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - t0
            y_pred = model.predict(X_test)
           
            acc = accuracy_score(y_test, y_pred)
            wf1 = f1_score(y_test, y_pred, average="weighted")
           
            results[name] = {"model": model, "accuracy": acc, "weighted_f1_score": wf1}
            log.info("  %-26s  acc=%.4f  wF1=%.4f  (%.1fs)", name, acc, wf1, elapsed)

        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1][config.SCORING_METRIC], reverse=True)
        )
        return sorted_results

    def save_results(self, sorted_results: dict) -> None:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
       
        all_rows = [{"model_name": k, "accuracy": v["accuracy"], "weighted_f1_score": v["weighted_f1_score"]}
                    for k, v in sorted_results.items()]
        pd.DataFrame(all_rows).to_csv(config.RESULTS_CSV_PATH, index=False)
        log.info("Saved all baseline comparison charts directly to: %s", config.RESULTS_CSV_PATH)
       
        best_3_names = list(sorted_results.keys())[:3]
        best_rows = [{"model_name": k, "accuracy": sorted_results[k]["accuracy"], "weighted_f1_score": sorted_results[k]["weighted_f1_score"]}
                     for k in best_3_names]
        pd.DataFrame(best_rows).to_csv(config.BEST_MODEL_NAMES_PATH, index=False)
        log.info("Saved top 3 model references directly to: %s", config.BEST_MODEL_NAMES_PATH)


class ModelTuner:
    """
    Handles hyperparameter optimization using GridSearchCV for selected baseline models.
    Dynamically injects class balancing configurations and aggregates true performance metrics.
    """
    def __init__(self, all_models: dict, tuning_grids: dict, output_pkl_path: str, output_csv_path: str):
        self.all_models = all_models
        self.tuning_grids = tuning_grids
        self.output_pkl_path = output_pkl_path
        self.output_csv_path = output_csv_path

    def run_tuning(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, target_model_names: list) -> Dict[str, Any]:
        _banner("STAGE 5 — HYPERPARAMETER TUNING")
        tuned_results = {}
        metric_logs = []
       
        for name in target_model_names:
            if name not in self.all_models or name not in self.tuning_grids:
                log.warning("Skipping tuning for %s (Missing model reference or configuration parameter grid).", name)
                continue
               
            print(f"\nRunning ADVANCED Optimization for: {name}...")
            model = self.all_models[name]
           
            # DYNAMIC INJECTION: Overriding parameters on the fly to deal with imbalance
            if hasattr(model, 'class_weight'):
                model.class_weight = "balanced"
                print("   --> Class weights balanced successfully.")
           
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=self.tuning_grids[name],
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
           
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Extract true metrics to protect pipeline against hardcoded presentation text
            preds = best_model.predict(X_test)
            test_acc = accuracy_score(y_test, preds)
            test_f1 = f1_score(y_test, preds, average='weighted')
           
            print(f" Best Local Params for {name}: {grid_search.best_params_}")
            print(f" CV Weighted F1: {grid_search.best_score_:.4f} | Test Weighted F1: {test_f1:.4f}")
           
            tuned_results[name] = best_model
            metric_logs.append({
                "Model Name": name,
                "Optimized_Accuracy": f"{test_acc:.4f}",
                "Optimized_Weighted_F1": f"{test_f1:.4f}"
            })
           
        # Serialize tuned components to disk
        os.makedirs(os.path.dirname(self.output_pkl_path), exist_ok=True)
        with open(self.output_pkl_path, "wb") as f:
            pickle.dump(tuned_results, f)
        log.info("All optimized tuned model configurations saved to: %s", self.output_pkl_path)

        # Dynamic, production-grade automated tracking output
        if metric_logs:
            df_metrics = pd.DataFrame(metric_logs)
            df_metrics.to_csv(self.output_csv_path, index=False)
            log.info("Success! Updated %s with active pipeline metrics.", self.output_csv_path)

        return tuned_results


class ModelEvaluator:
    """
    Generates final classification diagnostics and confusion matrix visualizations
    for the fully tuned models.
    """
    def __init__(self, results_dir: str, class_mapping: list):
        self.results_dir = results_dir
        self.class_mapping = class_mapping

    def evaluate(self, tuned_models: dict, X_test: np.ndarray, y_test: np.ndarray) -> None:
        _banner("STAGE 6 — PIPELINE EVALUATION")

        for name, model in tuned_models.items():
            preds = model.predict(X_test)

            print(f"\n{'='*60}\n  PERFORMANCE REPORT: {name}\n{'='*60}")
            unique_classes = len(np.unique(y_test))
            target_labels = self.class_mapping[:unique_classes] if unique_classes <= len(self.class_mapping) else None
            print(classification_report(y_test, preds, target_names=target_labels))

            # Render Matrix Visualizations
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix — {name}")
            plt.tight_layout()
           
            fig_path = os.path.join(self.results_dir, f"{name.lower().replace(' ', '_')}_matrix.png")
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            log.info("Saved Confusion Matrix graphic to: %s", fig_path)


# ===========================================================================
# ─── MAIN EXECUTION BLOCK ────────────────----------------------------------
# ===========================================================================

if __name__ == '__main__':
    _banner("EGT309 PROJECT GROUP 4 — DATA UNIFORMER & PIPELINE")

    # 1. Ingestion Phase
    loader = DataLoader()
    loader.connect()
    df = loader.load()
    loader.close()

    # Normalize structure across raw data matrices
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df = df.map(lambda x: x.lower().replace(' ', '_') if isinstance(x, str) else x)

    print("\nInitial Baseline Target Counts:")
    print(df['activity_level'].value_counts())

    # 2. Sequential Cleaning Transforms
    df = DataUniformer().transform(df)
    df = DataImputer(light_from_time=config.IMPUTER_LIGHT_FROM_TIME).fit(df).transform(df)
    df = DataCleaner().transform(df)

    print("\nColumns available before pipeline engineering:")
    print(df.columns.tolist())

    # 3. Feature Engineering Processing Split
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    fe = FeatureEngineer(
        time_of_day_map=config.FE_TIME_OF_DAY_MAP,
        ambient_light_map=config.FE_AMBIENT_LIGHT_MAP,
        activity_level_map=config.FE_ACTIVITY_LEVEL_MAP,
    )
    fe.fit(X)
    X_transformed = fe.transform(X)

    # Recombine engineering variables back into unified framework frame
    df_cleaned = X_transformed.copy()
    df_cleaned[config.TARGET_COLUMN] = y.values

    DataSaver().save(df_cleaned)
    print("\nFinished Running Data Cleaning and Feature Engineering Framework.")

    # 4. Modeling Engine (Baseline Execution)
    trainer = BasicModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_features(df_cleaned)
   
    _banner("STAGE 4 — BASELINE MODEL TRAINING")
    sorted_results = trainer.train_and_compare(X_train, X_test, y_train, y_test)
    trainer.save_results(sorted_results)

    print("\nBaseline Model Comparison Summary:")
    print("-" * 60)
    for name, r in sorted_results.items():
        print(f"  {name:<28}  acc={r['accuracy']:.4f}  wF1={r['weighted_f1_score']:.4f}")

    best_3_names = list(sorted_results.keys())[:3]
    log.info("Best 3 models selected for tuning: %s", best_3_names)

    # 5. Dynamic Hyperparameter Optimization (ModelTuner Execution)
    tuner = ModelTuner(
        all_models=trainer.models,
        tuning_grids=TUNING_GRIDS,
        output_pkl_path=config.TUNED_MODEL_PKL_PATH,
        output_csv_path=config.TUNED_METRICS_CSV_PATH
    )
    tuned_models = tuner.run_tuning(X_train, y_train, X_test, y_test, best_3_names)
   
    # 6. Pipeline Diagnostic Evaluation
    evaluator = ModelEvaluator(
        results_dir=config.RESULTS_DIR,
        class_mapping=CLASS_MAPPING
    )
    evaluator.evaluate(tuned_models, X_test, y_test)
   
    print("\nPipeline execution complete. Model configurations and tracking reports generated successfully.")
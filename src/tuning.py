"""
tuning.py
Handles hyperparameter optimization using GridSearchCV for the top baseline models.
Forces class balancing and expanded grid searching dynamically without touching config or base_model.
"""
import os
import sys
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

# =========================================================================
# PIPELINE STANDARD: Dynamic Path Lookup (Works on ANY computer)
# =========================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Workaround to keep base_model_v2 from crashing on its internal config import
import src.config
sys.modules['config'] = src.config

from src.config import RANDOM_STATE, UNIFORMER_ACTIVITY_FIXES, TARGET_COLUMN
from src.base_model_v2 import BasicModelTrainer

class ModelTuner:
    def __init__(self):
        self.trainer = BasicModelTrainer(random_state=RANDOM_STATE)
        self.tuned_models_dir = "saved_model"
        self.output_path = os.path.join(self.tuned_models_dir, "tuned_best_models.pkl")
        
    def run_tuning(self):
        print("--- Fetching data via SQLite Ingestion ---")
        df = self.trainer.load_cleaned_data()
        
        # Intercept and clean typos using the config file dictionary mapping
        target_col = TARGET_COLUMN
        if target_col in df.columns:
            df[target_col] = df[target_col].astype(str).str.replace("_", " ")
            df[target_col] = df[target_col].map(UNIFORMER_ACTIVITY_FIXES).fillna(df[target_col])
            df[target_col] = df[target_col].str.strip()

        X_train, X_test, y_train, y_test = self.trainer.prepare_features(df)
        
        all_models = self.trainer.models
        target_models = ["Random Forest", "Extra Trees", "Gradient Boosting"]
        
        # =========================================================================
        # PIPELINE ADVANCEMENT: Overriding hyperparameter grids locally
        # =========================================================================
        LOCAL_TUNING_GRIDS = {
            "Random Forest": {
                "n_estimators": [300, 400],
                "max_depth": [20, 30, None],
                "min_samples_split": [2, 5],
                "max_features": ["sqrt", "log2"]
            },
            "Extra Trees": {
                "n_estimators": [200, 300],
                "max_depth": [20, 30, None],
                "min_samples_split": [2, 5]
            },
            "Gradient Boosting": {
                "learning_rate": [0.05, 0.1],
                "max_depth": [5, 7],
                "n_estimators": [200, 300]
            }
        }
        # =========================================================================

        tuned_results = {}
        
        for name in target_models:
            if name not in all_models:
                continue
                
            print(f"\n⚡ Running ADVANCED Optimization for: {name}...")
            model = all_models[name]
            
            # DYNAMIC INJECTION: Inject class balancing on the fly to boost F1-score
            if hasattr(model, 'class_weight'):
                model.class_weight = "balanced"
                print(f"   --> Class weights balanced successfully.")
            
            param_grid = LOCAL_TUNING_GRIDS[name]
            
            grid_search = GridSearchCV(
                estimator=model, 
                param_grid=param_grid, 
                cv=3, 
                scoring='f1_weighted', 
                n_jobs=-1, 
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            preds = best_model.predict(X_test)
            score = f1_score(y_test, preds, average='weighted')
            
            print(f" ✨ Best Local Params for {name}: {grid_search.best_params_}")
            print(f" 📈 New Tuned Weighted F1 Score: {score:.4f}")
            
            tuned_results[name] = best_model
            
        os.makedirs(self.tuned_models_dir, exist_ok=True)
        with open(self.output_path, "wb") as f:
            pickle.dump(tuned_results, f)
        print(f"\n All optimized tuned model configurations saved to: {self.output_path}")

if __name__ == "__main__":
    tuner = ModelTuner()
    tuner.run_tuning()
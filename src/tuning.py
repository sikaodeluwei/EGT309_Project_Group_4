"""
tuning.py
Handles hyperparameter optimization using GridSearchCV for the top baseline models
by reading parameter matrices directly from src/config.py.
"""
import os
import sys
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

# Fix python paths to easily recognize modules inside the src folder
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# =========================================================================
# FIXED IMPORT: Look inside the src folder where Zhenyu moved config.py
# =========================================================================
from src.config import TUNING_PARAM_GRIDS, RANDOM_STATE
from src.basic_models import BasicModelTrainer

class ModelTuner:
    def __init__(self):
        self.trainer = BasicModelTrainer(random_state=RANDOM_STATE)
        self.tuned_models_dir = "saved_model"
        self.output_path = os.path.join(self.tuned_models_dir, "tuned_best_models.pkl")
        
    def run_tuning(self):
        print("--- Fetching data via SQLite Ingestion ---")
        df = self.trainer.load_cleaned_data()
        X_train, X_test, y_train, y_test = self.trainer.prepare_features(df)
        
        all_models = self.trainer.models
        target_models = ["Random Forest", "Extra Trees", "Gradient Boosting"]
        
        tuned_results = {}
        
        for name in target_models:
            if name not in all_models:
                continue
                
            if name not in TUNING_PARAM_GRIDS:
                print(f"⚠️ Grid array for {name} not found in config.py. Skipping.")
                continue
                
            print(f"\n⚡ Optimizing Hyperparameters for: {name}...")
            model = all_models[name]
            
            param_grid = TUNING_PARAM_GRIDS[name]
            
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
            
            print(f" Best Params for {name}: {grid_search.best_params_}")
            print(f" Tuned Weighted F1 Score: {score:.4f}")
            
            tuned_results[name] = best_model
            
        os.makedirs(self.tuned_models_dir, exist_ok=True)
        with open(self.output_path, "wb") as f:
            pickle.dump(tuned_results, f)
        print(f"\n All tuned model configurations saved to: {self.output_path}")

if __name__ == "__main__":
    tuner = ModelTuner()
    tuner.run_tuning()
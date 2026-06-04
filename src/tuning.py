"""
tuning.py
Handles hyperparameter optimization using GridSearchCV for the top 3 baseline models.
"""
import os
import yaml
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from src.basic_models import BasicModelTrainer

class ModelTuner:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.trainer = BasicModelTrainer(random_state=self.config['data']['random_state'])
        self.tuned_models_dir = "saved_model"
        self.output_path = os.path.join(self.tuned_models_dir, "tuned_best_models.pkl")
        
    def run_tuning(self):
        print("--- Fetching data via SQLite Ingestion ---")
        df = self.trainer.load_cleaned_data()
        X_train, X_test, y_train, y_test = self.trainer.prepare_features(df)
        
        # Pull raw baseline instances dynamically to bypass nested dictionary issues
        all_models = self.trainer.models
        target_models = ["Random Forest", "Extra Trees", "Gradient Boosting"]
        
        tuned_results = {}
        
        for name in target_models:
            if name not in all_models:
                continue
                
            print(f"\n⚡ Optimizing Hyperparameters for: {name}...")
            model = all_models[name]
            param_grid = self.config['tuning'][name]
            
            grid_search = GridSearchCV(
                estimator=model, 
                param_grid=param_grid, 
                cv=3, 
                scoring='f1_weighted', 
                n_jobs=-1, # Uses all your CPU cores to finish within the hour
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_
            
            preds = best_model.predict(X_test)
            score = f1_score(y_test, preds, average='weighted')
            
            print(f" Best Params for {name}: {grid_search.best_params_}")
            print(f" Tuned Weighted F1 Score: {score:.4f}")
            
            # Save the clean, standalone optimized estimator
            tuned_results[name] = best_model
            
        os.makedirs(self.tuned_models_dir, exist_ok=True)
        with open(self.output_path, "wb") as f:
            pickle.dump(tuned_results, f)
        print(f"\n All tuned model configurations saved to: {self.output_path}")

if __name__ == "__main__":
    tuner = ModelTuner()
    tuner.run_tuning()

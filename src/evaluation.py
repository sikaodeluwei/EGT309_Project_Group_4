"""
evaluation.py
Extracts comprehensive metrics, confusion matrices, and feature importance from tuned models.
"""
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# =========================================================================
# PATH FIX: Tell Python to look at your exact directory structure.
# =========================================================================
project_root = r"C:\Users\aadar\EGT309_Project_Group_4"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Fixed import to target config inside the src directory
from src.config import TARGET_COLUMN, RANDOM_STATE
from src.basic_models import BasicModelTrainer

class ModelEvaluator:
    def __init__(self):
        self.trainer = BasicModelTrainer(random_state=RANDOM_STATE)
        self.models_path = os.path.join("saved_model", "tuned_best_models.pkl")
        
    def generate_assessment_reports(self):
        df = self.trainer.load_cleaned_data()
        
        raw_features = df.drop(columns=[TARGET_COLUMN])
        if "session id" in raw_features.columns:
            raw_features = raw_features.drop(columns=["session id"])
        feature_names = raw_features.columns.tolist()
        
        _, X_test, _, y_test = self.trainer.prepare_features(df)
        class_mapping = [str(cls) for cls in self.trainer.target_encoder.classes_]
        
        if not os.path.exists(self.models_path):
            raise FileNotFoundError(f"Missing tuned artifacts at {self.models_path}. Execute src/tuning.py first.")
            
        with open(self.models_path, "rb") as f:
            tuned_models = pickle.load(f)
            
        for name, model in tuned_models.items():
            preds = model.predict(X_test)
            
            print(f"\n========================================================")
            print(f" DETAILED PERFORMANCE REPORT: {name}")
            print(f"========================================================")
            print(classification_report(y_test, preds, target_names=class_mapping))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_mapping, yticklabels=class_mapping)
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            cm_filename = f"saved_model/{name_to_filename(name)}_confusion_matrix.png"
            plt.savefig(cm_filename)
            plt.close()
            print(f" Saved Confusion Matrix plot to: {cm_filename}")
            
            # Feature Importance 
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title(f'Feature Importance Breakdown - {name}')
                sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette="viridis")
                plt.xlabel('Relative Importance Metric')
                plt.tight_layout()
                
                feat_filename = f"saved_model/{name_to_filename(name)}_feature_importance.png"
                plt.savefig(feat_filename)
                plt.close()
                print(f" Saved Feature Importance plot to: {feat_filename}")

def name_to_filename(name):
    return name.lower().replace(" ", "_")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.generate_assessment_reports()
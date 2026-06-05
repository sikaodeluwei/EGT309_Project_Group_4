# EGT309_Project_Group_4
# team name: group 4
# members: zhenyu, aadarsh, junhian

# section 1: who wrote which py file
zhenyu: config.py, base model.py
aadarsh:
junhian:

# section 2: How to run the pipeline

# section 3: docker environment(decide later we doing or not)

# section 4: Summary of key findings of EDA

# section 5: Explain and justify features that are engineered

# section 6: Explanation of choice of models (train at least 3 models) and justify any tuning methods used
## 6.1 Baseline Model Selection

Before selecting the final models, multiple candidate classification models were trained and compared using the same cleaned dataset. The target variable was `activity level`, making this a multi-class classification problem.

The candidate models included Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest, Gradient Boosting, Extra Trees, Support Vector Machine, and Naive Bayes. These models were chosen to provide a mix of simple baseline models, distance-based models, probabilistic models, and tree-based ensemble models.

The purpose of this stage was to avoid manually guessing the best models. Instead, the baseline model file ranked all candidate models based on weighted F1-score and saved the top three model names into `saved_model/best_3_model_names.csv`.

## 6.2(aadarsh do for the top 3)

# section 7: Explain any specific choice of metrics that are important to the problem statement

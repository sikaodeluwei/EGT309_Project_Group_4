# EGT309_Project_Group_4
# Team name: group 4
# Members: zhenyu, aadarsh, junhian

# Section 1: Python file developers
- zhenyu: config.py, base model.py
- aadarsh: tuning.py, evaluation.py
- junhian: config.py, clean_data_and_feature_engineering

# Section 2: How to run the pipeline

Two Methods:
- run pipeline.py file

OR

- run clean_data_and_feature_engineering > 

# Section 3: docker environment(decide later we doing or not)

# Section 4: Summary of key findings of EDA
Observations
- Dataset captures CO, CO₂, metal oxide, temperature, humidity, light, activity, and HVAC data across 10,000 rows

- Missing data exists in:
    - Humidity (approx 19.3%),
    - MetalOxideSensor_Unit2 (approx 14.1%),
    - Ambient Light Level (approx 10.5%),
    - CO_GasSensor (approx 8.3%)
    - Temperature has a unit inconsistency — some readings are in Kelvin- Humidity has out-of-range values (< 0% or > 100%)
    - Categorical columns have inconsistent naming conventions, now standardized in df_clean- CO2_InfraredSensor contains physically impossible negative values (113 rows identified)

- While CO2_InfraredSensor and CO2_ElectroChemicalSensor show different distributions, their underlying patterns are highly similar (cosine similarity ~95%), suggesting they measure the same phenomenon.

- Metal oxide sensor data units are highly correlated (pairwise cosine similarity > 99.6%), indicating they measure similar phenomena, likely from different locations.

Preprocessing Rationale
- Column Ordering:

    - Drop Session ID (no predictive value)- Move Activity Level to the rightmost column (ML target)

    - Place CO_GasSensor first

    - Group Temperature and Humidity together (correlated environmental features)

    - Group categorical columns to the right

- Data Type Conversions:

    - Removing the null values will not automatically convert it back to the desired dtype, requires manual conversion- CO_GasSensor: object → int

    - Time of Day, HVAC Operation Mode, Ambient Light Level, Activity Level: object → stringData Uniformity:

    - Standardize categorical column entries to lowercase with underscores (df_clean already shows this applied).

- Handling Missing Data (Imputation):

    - CO_GasSensor: Impute using metal oxide and CO₂ sensor readings.

    - Ambient Light Level: Impute using Time of Day.

    - Humidity: Treat out-of-range values (<0% or >100%) as missing and impute from Temperature.

    - MetalOxideSensor_Unit2: Impute using CO2_ElectroChemicalSensor and CO_GasSensor

- Handling Anomalies:

    - Temperature: Convert Kelvin readings to Celsius.

    - CO2_InfraredSensor: Treat negative values as anomalies; these will be set to NaN and imputed later (or capped at 0 before imputation). The discrepancy with CO2_ElectroChemicalSensor implies careful handling or aggregation after scaling is needed.


# Section 5: Explaination & Justification of Feature Engineering

- Aggregate Metal Oxide Sensors into a single feature (e.g., MetalOxideSensor_Aggregated or Overall Metal Oxide Levels).

- Aggregate CO2 Sensors into a single feature (e.g., CO2_Aggregated), potentially after scaling to normalize their different magnitudes.

- Time of Day → Integer Encoding (ordered by time ranges)

- HVAC Operation Mode → One-Hot Encoding (distinct, unordered classes)

- Ambient Light Level → Integer Encoding (ordered by intensity levels)

- Activity Level → Integer Encoding (ordered by intensity levels for ML target)

# Section 6: Explanation of choice of models (train at least 3 models) and justify any tuning methods used
## 6.1 Baseline Model Selection

Before selecting the final models, multiple candidate classification models were trained and compared using the same cleaned dataset. The target variable was `activity level`, making this a multi-class classification problem.

The candidate models included Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest, Gradient Boosting, Extra Trees, Support Vector Machine, and Naive Bayes. These models were chosen to provide a mix of simple baseline models, distance-based models, probabilistic models, and tree-based ensemble models.

The purpose of this stage was to avoid manually guessing the best models. Instead, the baseline model file ranked all candidate models based on weighted F1-score and saved the top three model names into `saved_model/best_3_model_names.csv`.

## 6.2 Hyperparameter Tuning
Top 3 Models Optimized: Random Forest, Extra Trees, and Gradient Boosting were tuned using stratified cross-validation (src/tuning.py).

Cost-Sensitive Learning: Injected class_weight='balanced' into Random Forest and Extra Trees to counteract severe class imbalance (High Activity: 929 samples vs. Moderate Activity: 176 samples). This heavily penalized minority class misclassifications.

Grid Search Settings: Fine-tuned n_estimators (up to 300 for voting stability) and max_depth (None, 10, 20 to prevent overfitting).

Champion Model: Extra Trees Classifier won, achieving a peak team score of 0.6757 Weighted $F_1$-Score (a +2.41% absolute improvement over the baseline).

# section 7: Explain any specific choice of metrics that are important to the problem statement
Why Accuracy Was Rejected: Raw accuracy is misleading due to data imbalance; a model could guess "High Activity" every time and look accurate while failing completely to detect an unconscious or fallen senior.

Weighted $F_1$-Score (Primary Metric): Calculates the balance between Precision and Recall while accounting for the row count of each class. Ensures the model is structurally stable across all three activity levels.

Class-Specific Recall (Safety Priority): Measures the system's sensitivity to risk. Minimizes dangerous False Negatives (e.g., missing an emergency). Tuning pushed Low Activity Recall to 70%, successfully catching 357 out of 511 low-activity crisis states.

Precision (Anti-Alarm Fatigue): Measures accuracy of triggered alerts. Maintaining a 55% Precision profile for Low Activity ensures family or emergency services are not spammed with false alarms, preventing them from ignoring real notifications.

# Traffic Demand Prediction using LightGBM

## Overview

This project predicts traffic demand using machine learning techniques and advanced feature engineering. The model is built using LightGBM and employs 5-Fold Cross Validation to achieve robust performance.

### Results

* **OOF R² Score:** 0.959374
* **Competition Score:** 95.9374
* **Model:** LightGBM Regressor
* **Validation Strategy:** 5-Fold Cross Validation

---

## Dataset

The dataset consists of traffic-related information such as:

* Geographical location (Geohash)
* Road Type
* Number of Lanes
* Presence of Large Vehicles
* Nearby Landmarks
* Weather Conditions
* Temperature
* Timestamp

Target Variable:

* `demand` — Traffic demand to be predicted.

---

## Features Engineered

### Time Features

* Hour
* Minute
* Time of Day (minutes since midnight)

### Cyclical Time Features

* Hour Sin
* Hour Cos
* Minute Sin
* Minute Cos

### Target Encodings

Historical demand statistics computed from training data:

* Mean Demand by Geohash
* Mean Demand by Road Type
* Mean Demand by Weather

### Categorical Encoding

Label Encoding applied to:

* Geohash
* RoadType
* LargeVehicles
* Landmarks
* Weather

---

## Model Architecture

### LightGBM Parameters

```python
{
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 8,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42
}
```

### Training Strategy

* 5-Fold Cross Validation
* Early Stopping: 150 rounds
* Maximum Trees: 5000
* Feature Importance Averaging

---

## Top Important Features

| Feature             | Importance |
| ------------------- | ---------- |
| Temperature         | 64666      |
| Time of Day         | 41939      |
| Geohash             | 41809      |
| Geohash Mean Demand | 31826      |
| Hour                | 16545      |
| Hour Sin            | 15781      |
| Hour Cos            | 14449      |
| Number of Lanes     | 10112      |
| Minute              | 9381       |
| Weather Mean Demand | 6579       |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/traffic-demand-prediction.git
cd traffic-demand-prediction
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn lightgbm
```

For Apple Silicon (M1/M2/M3):

```bash
brew install libomp
pip install lightgbm
```

---

## Project Structure

```text
traffic-demand-prediction/
│
├── train.csv
├── test.csv
├── sample_submission.csv
├── solve.py
├── submission_optimized.csv
├── README.md
└── catboost_info/
```

---

## Running the Project

```bash
python3 solve.py
```

The script will:

1. Load datasets
2. Perform feature engineering
3. Train LightGBM using 5-Fold CV
4. Generate predictions
5. Create submission file

Output:

```text
submission_optimized.csv
```

---

## Future Improvements

* Out-of-Fold Target Encoding
* CatBoost Ensemble
* XGBoost + LightGBM Blending
* Geohash-Hour Interaction Features
* RoadType-Hour Demand Statistics
* Hyperparameter Optimization using Optuna

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM

---

## Author

Pranav Thobhani

B.E. Computer Science, BITS Pilani Hyderabad Campus

Interested in Machine Learning, Data Science, Quantitative Finance, and Algorithmic Problem Solving.




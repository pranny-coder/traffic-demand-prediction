# Traffic Demand Prediction using LightGBM

## Overview

This project predicts traffic demand using a LightGBM-based machine learning model. The solution includes feature engineering, cross-validation, and traffic demand forecasting using tabular traffic data.

## Features

- Time-based feature engineering
- Cyclic encoding of temporal features
- Historical demand aggregation
- LightGBM regression model
- K-Fold cross-validation
- Automated prediction generation
## Results

The model was evaluated using 5-Fold Cross Validation.

### Cross-Validation Performance

| Fold | R² Score |
|------|----------|
| 1 | 0.9598 |
| 2 | 0.9598 |
| 3 | 0.9608 |
| 4 | 0.9557 |
| 5 | 0.9606 |

**Overall Out-of-Fold R² Score:** **0.9594**

This indicates that the model explains approximately **95.94% of the variance** in traffic demand, demonstrating strong predictive performance.

### Most Important Features

| Feature | Importance |
|----------|------------|
| Temperature | 64666.0 |
| Time of Day | 41939.8 |
| Geohash | 41809.6 |
| Geohash Mean Demand | 31826.2 |
| Hour | 16545.8 |
| Hour Sin | 15781.8 |
| Hour Cos | 14449.8 |
| Number of Lanes | 10112.8 |
| Minute | 9381.2 |
| Weather Mean Demand | 6579.4 |

### Key Insights

- Achieved **95.94% R² Score** using LightGBM.
- Temperature emerged as the most influential predictor.
- Spatial features (Geohash) significantly improved forecasting accuracy.
- Temporal features such as hour, time of day, and cyclic encodings captured traffic patterns effectively.
- Historical demand aggregation features further enhanced predictive performance.
## Technologies Used

- Python
- Pandas
- NumPy
- LightGBM
- Scikit-Learn

## Project Structure

```text
.
├── solve.py
└── README.md
```

## Running the Project

Install dependencies:

```bash
pip install pandas numpy lightgbm scikit-learn
```

Run the model:

```bash
python solve.py
```

## Model Highlights

- Feature engineering for temporal and geographic information
- Cross-validation for robust evaluation
- Demand forecasting using gradient boosting
- Optimized for tabular traffic datasets

## Future Improvements

- Hyperparameter optimization
- Ensemble models (XGBoost + CatBoost)
- Real-time traffic prediction
- Interactive dashboard for visualization

## Author

**Pranav UmeshBhai Thobhani**  
B.E. Computer Science  
BITS Pilani, Hyderabad Campus

GitHub: https://github.com/pranny-coder

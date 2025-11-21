# Credit Card Fraud Detection

A machine learning model that detects fraudulent credit card transactions. Built with LightGBM, trained on the Kaggle credit card dataset, and includes a Streamlit dashboard.

## What This Does

This model looks at credit card transactions and predicts whether they're fraudulent. It catches about 90% of fraud cases while keeping false alarms reasonably low.

The model uses 30 features (mostly anonymized PCA components from the original dataset) to make predictions. It's been tuned to prioritize catching fraud over avoiding false positives, which is the right trade-off for this problem.

## Project Structure

```
├── main.py                     # Training script
├── sst_ui.py                   # Web dashboard
├── eda.ipynb                   # EDA and baseline model dev
├── main_final.ipynb            # THe very final workflow tested in a notebook
├── fraud_detection_outputs/
│   ├── models/                 # Trained model
│   ├── results/                # Performance metrics
│   └── interpretations/        # SHAP plots
└── creditcard.csv              # Dataset (downloaded automatically)
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
lightgbm
shap
joblib
scikit-learn
imbalanced-learn
streamlit
plotly
```

Install with: `pip install -r requirements.txt`

## Running the Project

### 1. Train the Model

```bash
python main.py
```

This will:
- Download the dataset if needed (284,807 transactions)
- Train a LightGBM classifier with optimized hyperparameters
- Tune the decision threshold to hit 90% recall
- Generate performance metrics and SHAP interpretability plots
- Save everything to `fraud_detection_outputs/`

Training takes a few minutes on a decent laptop.

### 2. Launch the Dashboard

```bash
streamlit run sst_ui.py
```

The dashboard shows:
- Model performance metrics
- Confusion matrix and metric comparisons
- SHAP feature importance analysis
- Live prediction interface
- Cost-benefit calculations

## Model Performance

Based on the confusion matrix (threshold = 0.000):

- **True Negatives:** 56,428 (correctly identified legitimate transactions)
- **False Positives:** 436 (legitimate transactions flagged as fraud)
- **False Negatives:** 9 (frauds that slipped through)
- **True Positives:** 89 (frauds caught)

Key metrics:
- **Recall (Fraud):** ~90.8% - catches 9 out of 10 frauds
- **Precision (Fraud):** ~16.9% - 1 in 6 alerts is real fraud
- **F1-Score (Fraud):** ~28.6%
- **Overall Accuracy:** ~99.2%

The low precision is expected in fraud detection. With a 0.6% fraud rate in the data, even a good model will have many false positives. The important thing is catching most of the actual fraud.

## How It Works

### Pipeline Steps

1. **Preprocessing:** StandardScaler on Time and Amount (V1-V28 are already normalized)
2. **Resampling:** SMOTE to balance the training data
3. **Classification:** LightGBM with 999 trees, tuned hyperparameters

### Decision Threshold Tuning

The model outputs probabilities. We tune the threshold to meet the 90% recall target:
- Lower threshold = catch more fraud but more false alarms
- Higher threshold = fewer false alarms but miss more fraud

For fraud detection, we prefer false alarms over missed fraud.

### Key Features (from SHAP analysis)

Looking at the SHAP plot:

- **V14** is the strongest predictor (high values = fraud)
- **V4, V12, V10** are also important
- **Time and Amount** matter less than expected (fraudsters mimic normal patterns)

The model doesn't rely on a single feature, which makes it more robust.

## Limitations

1. **Class imbalance:** Only 0.6% of transactions are fraud. This makes evaluation tricky.
2. **Anonymized features:** V1-V28 are PCA-transformed, so we can't interpret them directly.
3. **Low precision:** Many false positives are unavoidable given the data distribution.
4. **Static model:** Fraud patterns change over time. This needs regular retraining.
5. **No temporal validation:** We used random split, not time-based. Real deployment should validate on future data.

## Dataset

Uses the Kaggle Credit Card Fraud Detection dataset:
- 284,807 transactions from September 2013
- 492 frauds (0.172% of data)
- Features V1-V28 are PCA components
- Time: seconds elapsed since first transaction
- Amount: transaction value

Dataset downloads automatically on first run.

## Files Generated

After training:
- `fraud_detection_pipeline.joblib` - trained model
- `classification_report.json` - detailed metrics
- `final_confusion_matrix.png` - confusion matrix plot
- `shap_summary_plot.png` - feature importance visualization

## Notes

- The aggressive threshold (0.000) is intentional - we want high recall
- SMOTE helps during training but real deployment wouldn't use it
- The model is production-ready but needs monitoring in the real world
- Cost-benefit analysis in the dashboard is illustrative, not based on real fraud costs

## License

MIT

## Acknowledgments

Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
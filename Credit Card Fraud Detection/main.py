"""
End-to-End Training and Interpretation Script for the Credit Card Fraud Detection Model.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
import joblib 
import json
from pathlib                 import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
                                    classification_report, 
                                    confusion_matrix, 
                                    precision_recall_curve,
                                    recall_score,
                                    precision_score
                                    )
from imblearn.pipeline       import Pipeline as ImbPipeline
from imblearn.over_sampling  import SMOTE
from sklearn.preprocessing   import StandardScaler
from sklearn.compose         import ColumnTransformer

LOCAL_FILE = Path('creditcard.csv')
DATA_URL   =  "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


if LOCAL_FILE.exists():
    df = pd.read_csv(LOCAL_FILE)
    print("Data Loaded locally")
else:
    try:
        print("Local file not found. Downloading from web....")
        df = pd.read_csv(DATA_URL)
        # Saving a copy locally
        df.to_csv(LOCAL_FILE, index=False)
        print(f"Downloaded an saved to {LOCAL_FILE}")
    
    except Exception as e:
        print(e)
print(f"Shape of the dataset : {df.shape}")

# Some configs and Constants
TARGET_COLUMN = 'Class'
TARGET_RECALL = 0.90

output_dir = Path('fraud_detection_outputs')
interpretations_dir = output_dir / 'interpretations'
model_dir = output_dir / 'models'
results_dir = output_dir / 'results'

interpretations_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Best Hyperparameters from our optuna study
BEST_PARAMS = {
  "classifier": 'LightGBM',
  "n_estimators": 999,
  "learning_rate": 0.01648749362485441,
  "num_leaves": 225,
  "max_depth": 6,
  "reg_alpha": 0.05960638225055657,
  "reg_lambda": 0.002496670161891966,
}

def get_X_y(df):
    """Loads the data and seperates features from the target"""

    X, y = df.drop([TARGET_COLUMN], axis=1), df[TARGET_COLUMN]
    print(f"Shape of X : {X.shape}")
    print(f"Shape of y : {y.shape}")

    return X, y

def build_pipeline(params: dict) -> ImbPipeline:
    """
    Build the Final ML pipeline
    """
    print("Started building pipeline")
    # Define columns to be scaled
    # From eda we know that 'Time' and 'Amount' are the only ones that need scaling.
    # BUT again If you want to scale the 'V' features. It is totally safe. It's just a matter of using resources

    numeric_features = ['Time', 'Amount']
    scaler = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numeric_features)
        ],
        remainder='passthrough' # Keep other columns if any
    )

    resampler = SMOTE(random_state=42)

    lgbm_params = {
        'objective': 'binary',
        'metrics': 'binary_logloss',
        'is_unbalance': True,
        'n_jobs': -1,
        'random_state': 42,
        'n_estimators': BEST_PARAMS['n_estimators'],
        'learning_rate': BEST_PARAMS['learning_rate'],
        'num_leaves': BEST_PARAMS['num_leaves'],
        'reg_alpha': BEST_PARAMS['reg_alpha'],
        'reg_lambda': BEST_PARAMS['reg_lambda'],
    }
    classifier = lgb.LGBMClassifier(**lgbm_params)

    # Now add the preprocessor as the first step in the pipeline
    return ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('resampler', resampler),
        ('classifier', classifier)
    ])

def tune_threshold(pipeline: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, np.ndarray]:
    """Finds the optimal probability threshold to meet target recall"""
    print(f"Tuning decision threshold for target recall of {TARGET_RECALL:.0%}...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    print(f"\nShape of precisions : {precisions.shape} | data type of precisins : {type(precisions)}")
    print(f"Shape of recalls : {recalls.shape} | data type of recalls : {type(recalls)}")
    print(f"Shape of thresholds : {thresholds.shape} | data type of thresholds : {type(thresholds)}")
    # Find Indexes where recall is greter than or equal to target recall (90) that we set earlier
    eligible_idx = np.where(recalls >= TARGET_RECALL)[0]
    print(f"\nShape of Eligible_indes : {eligible_idx.shape}")
    print(f"Datatype : {type(eligible_idx)}")
    
    if len(eligible_idx) > 0:
        # We can say that the target is achieved
        # Find the index with highest precision among those meeting recall target
        best_idx = eligible_idx[np.argmax(precisions[eligible_idx])]
        
        # Thresholds array is 1 element shorter, so we need to handle edge cases
        if best_idx >= len(thresholds):
            best_threshold = thresholds[-1]
        else:
            best_threshold = thresholds[best_idx]
        
        actual_recall = recalls[best_idx]
        print(f"\nWe achieved the target recall: {actual_recall:.2%}")
        print(f"Selected threshold: {best_threshold:.6f}")
    else:
        # Target not achieved. womp womp. no worries
        best_threshold = thresholds[0] # minimum threshold
        actual_recall = recalls[0]     # Maximum recall

        print("Target recall not achieved")
        print(f"Maximum possible recall : {actual_recall:.2%}")
        print(f"Using most aggressive threshold : {best_threshold:.4f}")
    
    y_pred_tuned = (y_proba >= best_threshold).astype(int)

    # Showing what we got
    final_recall = recall_score(y_test, y_pred_tuned)
    final_precision = precision_score(y_test, y_pred_tuned)
    print(f"Final metrics → Recall: {final_recall:.2%}, Precision: {final_precision:.2%}")

    return best_threshold, y_pred_tuned

# Create a function to save evaluation atifacts 
def save_evaluation_artifacts(y_test: pd.Series, y_pred: np.ndarray, threshold: float):
    """Generates and saves evaluation reports and plots"""
    report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'], output_dict=True)
    with open(results_dir / "classification_report.json", 'w') as f:
        json.dump(report, f, indent=4)
    print("classification_report saved")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(f'Final Confusion Matrix (Threshold = {threshold:.3f})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(interpretations_dir / "final_confusion_matrix.png")
    plt.close()
    print("Confusion matrix plot saved.")

# Create a function to save interpretation plots
def save_interpretation_plots(pipeline: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Generates and saves SHAP interpretation plots.
    **Explicitly transform data for the explainer**
    """ 
    print("Generating and saving SHAP interpretation plots")
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['classifier']  # Fixed: 'named_Steps' -> 'named_steps'


    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    # Convert to DataFrame with proper feature names
    X_test_transformed_df = pd.DataFrame(
        X_test_transformed,
        index=X_test.index,
        columns=feature_names
    )
    # Create shap explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed_df)

    # Use class 1 (Fraud) SHAP values
    if isinstance(shap_values, list):
        shap_values_fraud = shap_values[1]
    else:
        shap_values_fraud = shap_values

    
    # SHAP summary plot
    shap.summary_plot(
        shap_values_fraud,
        X_test_transformed_df,
        show=False,
        plot_type='dot'
    )
    plt.title("SHAP Summary Plot (Feature Impact on Fraud Prediction)")
    plt.tight_layout()
    plt.savefig(interpretations_dir / "shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved.")

def main():
    """Main function to run the corrected training pipeline"""
    #1. Load the data
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test  = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    #2. Build Pipeline
    pipeline  = build_pipeline(BEST_PARAMS)

    #3. Train model
    print("Training the final model")
    pipeline.fit(X_train, y_train)

    #4. Tune Threshold and evaluate
    tuned_threshold, y_pred_tuned = tune_threshold(pipeline,
                                                   X_test,
                                                   y_test)
    save_evaluation_artifacts(y_test, y_pred_tuned,
                              tuned_threshold)
    
    #5. Interpret - Fixed: added y_test parameter
    save_interpretation_plots(pipeline, X_test, y_test)

    #6. Save the final model
    model_path = model_dir / "fraud_detection_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nFinal trained pipeline saved to : {model_path}")

    print("\nTraining Pipeline Finished Successfully")
    print(f"All outputs saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
import logging
import joblib
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, 
    RobustScaler, 
    OneHotEncoder, 
    OrdinalEncoder, 
    PowerTransformer
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Configure Logging
# Set up logging to INFO level with timestamp format
# In production (AWS/Azure), we need logs to debug since there's no UI to see print statements
# basicConfig creates a root logger that outputs to console
# why not use print() : Logs have severity levels (INFO, Warning, Error) and can be redirected to files
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    
    # Domain-specific NaN handling
    nan_cols = ['Alley', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 
                'BsmtFin Type 2', 'Fireplace Qu', 'Garage Type', 'Garage Finish', 
                'Garage Qual', 'Garage Cond', 'Pool QC', 'Fence', 'Misc Feature']
    for col in nan_cols:
        if col in df.columns: 
            df[col] = df[col].fillna('None')
    
    df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(df['Year Built'])
    df['MS SubClass'] = df['MS SubClass'].astype(str)
    
    # Feature engineering
    df['TotalSF'] = df['Total Bsmt SF'].fillna(0) + df['1st Flr SF'].fillna(0) + df['2nd Flr SF'].fillna(0)
    df['TotalBath'] = (df['Full Bath'] + 0.5 * df['Half Bath'] + 
                       df['Bsmt Full Bath'].fillna(0) + 0.5 * df['Bsmt Half Bath'].fillna(0))
    df['HouseAge'] = df['Yr Sold'] - df['Year Built']
    df['RemodAge'] = df['Yr Sold'] - df['Year Remod/Add']
    
    # DROP REDUNDANT FEATURES 
    df = df.drop(columns=['Total Bsmt SF', '1st Flr SF', '2nd Flr SF',
                          'Full Bath', 'Half Bath', 'Bsmt Full Bath', 'Bsmt Half Bath',
                          'Year Built', 'Year Remod/Add'])
    
    return df

def get_pipeline(X_train: pd.DataFrame, trial:optuna.Trial = None)-> Pipeline:
    """
    Constructs the full ML pipeline (preprocessing + model)
    
    * BUild sklearn Pipeline with ColumnTransformer for preprocessing and model for prediction
    * Pipelines ensure train and test data get identical transformations (prevents data leakage)
    * Define Column groups -> create transformers -> combine -> add model

    Optuna:
    * If trial is provided, optuna suggests hyperparameters, otherwise use defaults
    * Automated tuning finds better configurations than manual grid search
    * Use trial.suggest_* methods to propose values from predefined ranges
    * Don't use when doing quick prototyping, or hyperparameters are already well-tuned from prior experience  
    """

    # Define Column groups
    #1. Ordinal Features (Order matters: Poor < Fair < Good)
    ordinal_cols = ['Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 'Heating QC', 
                'Kitchen Qual', 'Fireplace Qu', 'Garage Qual', 'Garage Cond', 'Pool QC']

    # The ranking logic (None is lowest, Ex is highest)
    quality_order = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
    ordinal_categories = [quality_order]*len(ordinal_cols)

    # 2. Nominal Features (Order Doesn't matter)
    # We exclude the ordinal ones we just identified
    nominal_cols = [c for c in X_train.select_dtypes(include=['object', 'category']).columns if c not in ordinal_cols]

    # 3. Numerical Features
    # We exclude the ones we engineered out or don't want to scale (like MS SubClass if we treated as num)
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()


    ############### Numerical Pipeline
    #
    # Preprocessing Pipeline for numerical columns
    # Numerical features have different scales (1-10 vs 1000-5000) and distributions (skewed)
    # flow : Impute -> Scale -> Transform (Optionally)
    # 
    if trial:
        # Hyperparameter search space for Optuna
        
        #1. Let Optuna choose between RobustScaler and StandardScaler
        #   because different datasets respond better to different scaling methods
        #   #
        #   trial.suggest_categorical presents 2 choices, Optuna picks based on performance

        # ROBUSTSCALER: Uses median and IQRD - robust to outliers
        # STANDARDSCALER : Uses mean and std dev - assumes normal distribution

        scaler_type = trial.suggest_categorical('scaler', ['robust', 'standard'])

        # 2. Let optuna decide whether to apply PowerTransformer
        #    PowerTransformer makes data more guassian (normal distribution)
        #    Linear models (Ridge/Lasso) assume normality - helps them perform better
        #    #
        # Yeo-Johnson method can handle zeros and negative values (Box-Cox) can't
        # Use when Data is heavily skewed (right - tailed distributions in housing data)
        # No need to use when data is already normal and using tree based models
        # Cost : Adds computation time, might overfit on small datasets
        use_power_transform = trial.suggest_categorical('power_transform', [True, False])
        scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        
        if use_power_transform:
            # The 3 steps now : Impute missing -> Scale -> Transform to guassian

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', scaler),
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

        else:
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', scaler),
            ])

    # if no trial at all
    else:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('transformer', PowerTransformer(method='yeo-johnson'))
        ])


    ################ Ordinal Pipeline
    #
    # Encode Ordinal (ranked) categorical variables
    # To preserve inherent order(poor=0, Fair=1, Good=2, Excellent=3)
    # Use OrdinalEncoder with explicit category ordering
    
    # Ordinal Encoder converts categories to integers based on order
    # Preserves ranking (Ex=5 > Gd=4) which linear models can use

    ord_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('encoder', OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown='error'
        ))
    ])    

    ################ Nominal Pipeline
    #
    # Encode NOminal(unranked) categorical variables
    # Nominal categories have no inherent order

    nom_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    ####### COLUMN TRANSFORMER
    # Combine all preprocessing pipelines into one transformer
    # Because Each column type needs different preprocessing
    # It does this by taking a list of (name, transformer, columns) tuples
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numerical_cols),
        ('ord', ord_pipeline, ordinal_cols),
        ('nom', nom_pipeline, nominal_cols)
    ], remainder='drop') # Drop any columns that are left
    
    ######## MODEL CONFIG
    # Stacking ensemble with Ridge + HistGradientBoosting -> Lasso meta-learner 
    # Why? : Combining strengths of different models
    #        - Ridge captures Linear relationships
    #        - HGB captures non-linear patterns
    #        - Lasso learns optimal combination + does feature selection

    # Ridge and HGB make predictions -> Lasso combines them -> Final prediction

    if trial:
        # optuna Hyperparameter tuning
        
        # +++++ Ridge Hyperparameters +++++ 
        
        #  alpha controls L2 regularization stren
        #  Higher alpha = more regularization = simpler model (prevents overfitting)
        #  We will Search logarithmically from 0.1 to 100
                        # 0.1 = minimal regularization, 100 = heavy regularization
        ridge_alpha = trial.suggest_float('ridge_alpha', 0.1, 100.0, log=True)
    
        # ++++++ HistGradientBoosting Hyperparameters ++++

        #   HistGradientBoosting is fast gradient boosting (like XGBoost but sklearn native)
        #   Handles non-linearity, interactions, and missing values well
        
            # MAX_ITER = Number of boosting iterations( trees )
            # WHY : More trees = more complex model, better fit but slower and riskier
            #       50-300 is practical (50=Fast search, 300 = thorough search)
        hgb_max_iter = trial.suggest_int('hgb_max_iter', 50, 300)

            # MAX_DEPTH = Maximum depth of each tree
            # WHY : Deeper trees capture more complex patterns but overfit easily
            # Range 3-15 (3 = shallow/simple, 15 = deep/complex)
        hgb_max_depth = trial.suggest_int('hgb_max_depth', 3, 15)

            # LEARNING_RATE = Step Size for gradient descent
            # WHY : lower rate = more conservative updates = better generalization but slower convergence
            # RANGE : 0.01-0.3
            # LOG SCALE because Effect is multiplicative
            # RULE OF THUMB : learning_rate * max_iter should be roughly constant
            #                 low rate needs more interations to converge
        hgb_learning_rate = trial.suggest_float('hgb_learning_rate', 0.01, 0.3, log=True)

            # L2_REGULARIZATION : Ridge penalty on tree leaf weights
            # Prevents overfitting by penalizing large weights
            # RANGE : 1e-5 to 10 (minimal, strong)
        hgb_l2 = trial.suggest_float('hgb_l2', 1e-5, 10.0, log=True)

        # +++++++ LASSO (META-LEARNER) HYPERPARAMETERS
        # Lasso will combine base model predictions

        # WHY LASSO : L1 regularization  does automatic feature selection
        #             (can however set some weights to exactly zero)

        lasso_alpha = trial.suggest_float('lasso_alpha', 1e-5, 1.0, log=True)

        estimators = [
            ('ridge', Ridge(alpha=ridge_alpha)),
            ('hgb', HistGradientBoostingRegressor(
                random_state=42,
                max_iter=hgb_max_iter,
                max_depth=hgb_max_depth,
                learning_rate=hgb_learning_rate,
                l2_regularization=hgb_l2
            ))
        ]

        # +++++ STACKING REGRESSOR
        # Stackingregressor trains base models, then trains meta-learner on their predictions
        stacking_regressor = StackingRegressor(
            estimators=estimators,
            final_estimator=Lasso(alpha=lasso_alpha, random_state=42),
            n_jobs=-1
        )

    else:
        estimators = [
            ('ridge', Ridge(alpha=10.0)),
            ('hgb', HistGradientBoostingRegressor(
                random_state=42,
                max_iter=100
            ))
        ]
        stacking_regressor = StackingRegressor(
            estimators=estimators,
            final_estimator=Lasso(alpha=0.001, random_state=42),
            n_jobs=-1
        )
    
    # +++++++++++++ TARGET TRANSFORMATION
    # What : TransformedTargetRegressor automatically transforms target (y) during fit and inverse during predict
    final_model = TransformedTargetRegressor(
        regressor=stacking_regressor,
        func=np.log1p,
        inverse_func=np.expm1
    )

    # Main pipeline
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', final_model)
    ])

# Optuna objective
def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """
    Optuna Objective Function
    """
    pipeline = get_pipeline(X_train=X_train, trial=trial)

    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    return -scores.mean()


def train(n_trials: int = 50, use_optuna:bool=True):
    """
    Main function that puts the pieces together
    """       
    logging.info("Starting training pipeline")
    logging.info("Loading dataset")
    df = pd.read_csv('data/AmesHousing.csv')

    # train /test
    X,y = df.drop('SalePrice', axis=1), df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info("Cleaning Train and test data (seperately)")
    X_train_clean = clean_data(X_train)
    X_test_clean  = clean_data(X_test)

    if use_optuna:
        logging.info(f"Starting Optuna optimization with {n_trials} trials....")
        # WHAT: Create Optuna study object
        # WHY: Study manages the optimization process (trials, history, best params)
        # HOW: Specify direction (minimize RMSLE), sampler (how to search), and name
        # 
        # DIRECTION='minimize': We want to minimize RMSLE (lower error = better)
        # ALTERNATIVE: 'maximize' for metrics like R² or accuracy
        # 
        # SAMPLER=TPESampler: Tree-structured Parzen Estimator
        # WHAT: Bayesian optimization algorithm that learns from past trials
        # WHY: More efficient than random/grid search
        #      Focuses search on promising regions of hyperparameter space
        # HOW: Builds probability model of which hyperparameters work well
        # ALTERNATIVES: 
        #   - RandomSampler: Random search (faster but less efficient)
        #   - GridSampler: Grid search (exhaustive but exponentially slow)
        #   - CmaEsSampler: Evolution strategy (good for continuous params)
        # WHY TPE: Best general-purpose choice for mixed discrete/continuous params
        # 
        # SEED=42: Reproducible sampling
        # 
        # STUDY_NAME: Optional identifier for logging/tracking
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            study_name='house_price_optimization'
        )
        study.optimize(
            lambda trial: objective(trial, X_train_clean, y_train),
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=1
        )
        # Optuna results
        logging.info("Optuna Results")
        logging.info(f"BEST RMSLE: {study.best_value:.4f}")
        logging.info(f"Best Hyperparameters: {study.best_params}")
        
        # Get the best trial and build pipeline with it's hyperparameters
        best_trial = study.best_trial
        pipeline = get_pipeline(X_train_clean, best_trial) 

        # Save study object to disk
        joblib.dump(study, 'optuna_study.joblib')
        logging.info("Optuna study saved ")

    else:
        logging.info("No optuna")
        pipeline = get_pipeline(X_train_clean)


    logging.info("Fitting Final model on full training set...")
    pipeline.fit(X_train_clean, y_train)

    # Evaluating and testing the model
    logging.info("Evaluating model on test set")
    y_pred = pipeline.predict(X_test_clean)

    rmse = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred)))
    # MAE
    mae = np.mean(np.abs(y_test - y_pred))
    # r2
    r2 = r2_score(y_test, y_pred)

    logging.info(" FINAL TEST RESULTS ")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE: ${mae:.2f}")
    logging.info(f"R2 Score: {r2:.4f}")

    joblib.dump(pipeline, 'house_price_model.joblib')
    logging.info("Model saved to 'house_price_model.joblib'")


if __name__ == "__main__":
    train(n_trials=50, use_optuna=True)

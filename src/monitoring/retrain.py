# Copyright (c) 2025 Deepika Sharma. Licensed under Apache 2.0.

import pickle
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH        = 'data/creditcard.csv'
MODEL_PATH       = 'models/fraud_model.pkl'
NEW_MODEL_PATH   = 'models/fraud_model_candidate.pkl'
FEATURES_PATH    = 'models/feature_names.pkl'
BASELINE_ROC_AUC = 0.9747   # current model score -- new model must beat this

def load_current_roc_auc():
    """Load the current model's ROC-AUC from mlflow or use baseline."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name('FinStreamAI-Fraud-Detection')
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=['metrics.roc_auc DESC'],
                max_results=1
            )
            if runs:
                return runs[0].data.metrics.get('roc_auc', BASELINE_ROC_AUC)
    except Exception:
        pass
    return BASELINE_ROC_AUC

def retrain():
    """
    Retrain XGBoost on the full dataset.
    Saves new model as candidate only if it beats current ROC-AUC.
    Returns dict with results.
    """
    logger.info("Starting retraining pipeline...")
    start_time = datetime.utcnow()

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df):,} transactions — "
                f"{df['Class'].sum()} fraud cases "
                f"({df['Class'].mean()*100:.3f}%)")

    X = df.drop('Class', axis=1)
    y = df['Class']

    # ── Train/test split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Class imbalance ───────────────────────────────────────────────────────
    fraud_count = y_train.sum()
    legit_count = len(y_train) - fraud_count
    scale_pos_weight = legit_count / fraud_count
    logger.info(f"scale_pos_weight: {scale_pos_weight:.1f}")

    # ── Train new model ───────────────────────────────────────────────────────
    logger.info("Training new XGBoost model...")
    new_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
    new_model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_proba = new_model.predict_proba(X_test)[:, 1]
    y_pred  = new_model.predict(X_test)
    new_roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    fraud_metrics = report.get('1', {})

    logger.info(f"New model ROC-AUC:  {new_roc_auc:.4f}")
    logger.info(f"New model Precision: {fraud_metrics.get('precision', 0):.4f}")
    logger.info(f"New model Recall:    {fraud_metrics.get('recall', 0):.4f}")
    logger.info(f"New model F1:        {fraud_metrics.get('f1-score', 0):.4f}")

    # ── Compare against current model ─────────────────────────────────────────
    current_roc_auc = load_current_roc_auc()
    logger.info(f"Current model ROC-AUC: {current_roc_auc:.4f}")
    improved = new_roc_auc > current_roc_auc + 0.001

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    mlflow.set_experiment('FinStreamAI-Fraud-Detection')
    run_name = f"Retrain-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param('trigger', 'drift_detected')
        mlflow.log_param('scale_pos_weight', round(scale_pos_weight, 2))
        mlflow.log_param('n_estimators', 100)
        mlflow.log_param('max_depth', 6)
        mlflow.log_metric('roc_auc', new_roc_auc)
        mlflow.log_metric('fraud_precision', fraud_metrics.get('precision', 0))
        mlflow.log_metric('fraud_recall', fraud_metrics.get('recall', 0))
        mlflow.log_metric('fraud_f1', fraud_metrics.get('f1-score', 0))
        mlflow.log_metric('improved', int(improved))
        mlflow.sklearn.log_model(new_model, 'model')

    # ── Save candidate model if improved ──────────────────────────────────────
    duration = (datetime.utcnow() - start_time).total_seconds()

    if improved:
        with open(NEW_MODEL_PATH, 'wb') as f:
            pickle.dump(new_model, f)
        logger.info(
            f"New model saved as candidate — "
            f"ROC-AUC improved {current_roc_auc:.4f} -> {new_roc_auc:.4f}"
        )
    else:
        logger.info(
            f"New model did NOT improve — "
            f"keeping current model "
            f"({current_roc_auc:.4f} vs {new_roc_auc:.4f})"
        )

    return {
        'status': 'improved' if improved else 'no_improvement',
        'current_roc_auc': round(current_roc_auc, 4),
        'new_roc_auc': round(new_roc_auc, 4),
        'improved': improved,
        'candidate_saved': improved,
        'duration_seconds': round(duration, 1),
        'run_name': run_name
    }

if __name__ == '__main__':
    result = retrain()
    print(f"\nRetraining result: {result}")
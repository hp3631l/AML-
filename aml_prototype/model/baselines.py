"""
Baseline Models for AML Prototype.

Compares the Temporal GraphSAGE performance against simple tabular models
(Logistic Regression and Random Forest) using only the 102d node features.
This proves whether the graph structure (edges) adds predictive value.
"""

import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.data_prep import build_pyg_graph

def run_baselines():
    print("=== Training Baseline Models ===")
    
    data = build_pyg_graph()
    
    # We only use node features for baselines
    X = data.x.numpy()
    y = data.y.numpy()
    
    print(f"Dataset: {X.shape[0]} nodes, {X.shape[1]} features")
    
    # Stratified split (70/30) for simplicity in baselines
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    except ValueError:
        print("Warning: Stratified split failed. Falling back to random split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predict probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.decision_function(X_test)
            
        preds = model.predict(X_test)
        
        # Metrics
        try:
            auc_roc = roc_auc_score(y_test, probs)
            precision, recall, _ = precision_recall_curve(y_test, probs)
            auc_pr = auc(recall, precision)
            f1 = f1_score(y_test, preds, zero_division=0)
        except ValueError:
            auc_roc = auc_pr = f1 = float('nan')
            
        print(f"Results for {name}:")
        print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  AUC-PR:  {auc_pr:.4f}")
        print(f"  F1 Score: {f1:.4f}")

if __name__ == "__main__":
    run_baselines()

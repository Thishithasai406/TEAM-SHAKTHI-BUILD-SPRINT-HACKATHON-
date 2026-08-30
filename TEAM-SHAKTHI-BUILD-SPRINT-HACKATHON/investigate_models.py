import json
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, auc, matthews_corrcoef, balanced_accuracy_score
)
from deal_scorer import extract_normalized_features, calculate_ece
from main import load_json_data

hist = load_json_data('historical_deals.json')
live = load_json_data('live_deals.json')

X = np.array([extract_normalized_features(d) for d in hist])
y = np.array([1 if d.get('outcome', '').lower() in ['stalled', 'lost'] else 0 for d in hist])

# Fixed 80/20 train/holdout split with fixed random_state
X_tr, X_holdout, y_tr, y_holdout = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

prevalence = np.mean(y_tr)
brier_baseline = np.mean((y_holdout - prevalence) ** 2)

candidate_models = {
    "RandomForest (depth=5, balanced)": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42),
    "RandomForest (depth=3, balanced)": RandomForestClassifier(n_estimators=100, max_depth=3, class_weight="balanced", random_state=42),
    "RandomForest (unconstrained, default)": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression (balanced)": LogisticRegression(class_weight="balanced", random_state=42),
    "GradientBoosting (depth=3)": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    "HistGradientBoosting (class_weight=balanced)": HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=42),
}

print(f'Train size: {len(y_tr)} (WON={sum(y_tr==0)}, FAIL={sum(y_tr==1)})')
print(f'Holdout size: {len(y_holdout)} (WON={sum(y_holdout==0)}, FAIL={sum(y_holdout==1)})\n')

print(f'{"Model":<42} | {"Calib":<8} | {"ROC-AUC":<7} | {"PR-AUC":<7} | {"MCC@.7":<7} | {"BalAcc@.7":<9} | {"Brier":<7} | {"ECE":<7}')
print('-' * 115)

for m_name, base_clf in candidate_models.items():
    for cal_method in ["none", "sigmoid", "isotonic"]:
        if cal_method == "none":
            clf = base_clf
            clf.fit(X_tr, y_tr)
            p_holdout = clf.predict_proba(X_holdout)[:, 1]
        else:
            clf = CalibratedClassifierCV(estimator=base_clf, method=cal_method, cv=5)
            clf.fit(X_tr, y_tr)
            p_holdout = clf.predict_proba(X_holdout)[:, 1]

        # Discrimination metrics
        roc_auc = roc_auc_score(y_holdout, p_holdout)
        prec_arr, rec_arr, _ = precision_recall_curve(y_holdout, p_holdout)
        pr_auc_val = auc(rec_arr, prec_arr)
        
        pred_70 = (p_holdout >= 0.70).astype(int)
        mcc_70 = matthews_corrcoef(y_holdout, pred_70)
        bal_acc_70 = balanced_accuracy_score(y_holdout, pred_70)
        
        brier = brier_score_loss(y_holdout, p_holdout)
        ece = calculate_ece(y_holdout, p_holdout)

        print(f'{m_name:<42} | {cal_method:<8} | {roc_auc:<7.4f} | {pr_auc_val:<7.4f} | {mcc_70:<7.4f} | {bal_acc_70:<9.4f} | {brier:<7.4f} | {ece:<7.4f}')

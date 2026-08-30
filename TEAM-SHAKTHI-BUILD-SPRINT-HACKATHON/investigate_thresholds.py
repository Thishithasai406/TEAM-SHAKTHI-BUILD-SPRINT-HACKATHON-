import json
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, auc, matthews_corrcoef, balanced_accuracy_score
)
from deal_scorer import extract_normalized_features
from main import load_json_data

hist = load_json_data('historical_deals.json')
X = np.array([extract_normalized_features(d) for d in hist])
y = np.array([1 if d.get('outcome', '').lower() in ['stalled', 'lost'] else 0 for d in hist])

X_tr, X_holdout, y_tr, y_holdout = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

rf_bal = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
cal_sig = CalibratedClassifierCV(estimator=rf_bal, method="sigmoid", cv=5)
cal_sig.fit(X_tr, y_tr)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p_cv = cross_val_predict(cal_sig, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
p_holdout = cal_sig.predict_proba(X_holdout)[:, 1]

print(f'=== 5-FOLD CV ON TRAINING SET (n={len(y_tr)}) ===')
print(f'{"TH":<5} | {"Acc":<6} | {"Prec":<6} | {"Recall":<6} | {"Spec":<6} | {"F1":<6} | {"BalAcc":<6} | {"MCC":<6} | {"TN":<4} {"FP":<4} {"FN":<4} {"TP":<4}')
print('-' * 85)

for th in [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
    pred_cv = (p_cv >= th).astype(int)
    cm = confusion_matrix(y_tr, pred_cv)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    rec = recall_score(y_tr, pred_cv, zero_division=0)
    prec = precision_score(y_tr, pred_cv, zero_division=0)
    f1 = f1_score(y_tr, pred_cv, zero_division=0)
    acc = accuracy_score(y_tr, pred_cv)
    bal_acc = balanced_accuracy_score(y_tr, pred_cv)
    mcc = matthews_corrcoef(y_tr, pred_cv)
    print(f'{th:<5.2f} | {acc:<6.4f} | {prec:<6.4f} | {rec:<6.4f} | {spec:<6.4f} | {f1:<6.4f} | {bal_acc:<6.4f} | {mcc:<6.4f} | {tn:<4} {fp:<4} {fn:<4} {tp:<4}')

print(f'\n=== UNSEEN HOLDOUT SET (n={len(y_holdout)}) ===')
print(f'{"TH":<5} | {"Acc":<6} | {"Prec":<6} | {"Recall":<6} | {"Spec":<6} | {"F1":<6} | {"BalAcc":<6} | {"MCC":<6} | {"TN":<4} {"FP":<4} {"FN":<4} {"TP":<4}')
print('-' * 85)

for th in [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
    pred_h = (p_holdout >= th).astype(int)
    cm = confusion_matrix(y_holdout, pred_h)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    rec = recall_score(y_holdout, pred_h, zero_division=0)
    prec = precision_score(y_holdout, pred_h, zero_division=0)
    f1 = f1_score(y_holdout, pred_h, zero_division=0)
    acc = accuracy_score(y_holdout, pred_h)
    bal_acc = balanced_accuracy_score(y_holdout, pred_h)
    mcc = matthews_corrcoef(y_holdout, pred_h)
    print(f'{th:<5.2f} | {acc:<6.4f} | {prec:<6.4f} | {rec:<6.4f} | {spec:<6.4f} | {f1:<6.4f} | {bal_acc:<6.4f} | {mcc:<6.4f} | {tn:<4} {fp:<4} {fn:<4} {tp:<4}')

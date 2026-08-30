import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix, precision_recall_curve, auc
from deal_scorer import extract_normalized_features
from main import load_json_data

hist = load_json_data('historical_deals.json')
live = load_json_data('live_deals.json')

X = np.array([extract_normalized_features(d) for d in hist])
y = np.array([1 if d.get('outcome', '').lower() in ['stalled', 'lost'] else 0 for d in hist])
X_live = np.array([extract_normalized_features(d) for d in live])

print(f'Total Historical: {len(y)}')
print(f'Class Distribution: WON (0) = {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%), STALLED/LOST (1) = {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)\n')

# Calibrated Classifier with RandomForestClassifier base
rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
cal_clf = CalibratedClassifierCV(estimator=rf, method='sigmoid', cv=5)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_proba = cross_val_predict(cal_clf, X, y, cv=cv, method='predict_proba')

# Dynamically map positive class index
classes = list(cal_clf.fit(X, y).classes_)
pos_idx = classes.index(1)
y_proba_pos = y_proba[:, pos_idx]
y_pred = (y_proba_pos >= 0.5).astype(int)

prec_arr, rec_arr, _ = precision_recall_curve(y, y_proba_pos)
pr_auc = auc(rec_arr, prec_arr)

cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

print('=== CROSS-VALIDATION METRICS (5-FOLD STRATIFIED) ===')
print(f'Accuracy   : {accuracy_score(y, y_pred):.4f}')
print(f'Precision  : {precision_score(y, y_pred):.4f}')
print(f'Recall     : {recall_score(y, y_pred):.4f}')
print(f'F1-Score   : {f1_score(y, y_pred):.4f}')
print(f'ROC-AUC    : {roc_auc_score(y, y_proba_pos):.4f}')
print(f'PR-AUC     : {pr_auc:.4f}')
print(f'Brier Score: {brier_score_loss(y, y_proba_pos):.4f}')

print('\n=== CONFUSION MATRIX ===')
print(f'True Negatives (WON correctly predicted): {tn}')
print(f'False Positives (WON misclassified as Risk): {fp}')
print(f'False Negatives (STALLED/LOST missed): {fn}')
print(f'True Positives (STALLED/LOST correctly predicted): {tp}')

cal_clf.fit(X, y)
rf_fitted = cal_clf.calibrated_classifiers_[0].estimator
importances = rf_fitted.feature_importances_
feature_names = ['deal_size', 'days_in_stage', 'response_latency_hrs', 'stakeholder_count', 'competitor_mentions', 'scope_change_flags', 'sentiment']

print('\n=== FEATURE IMPORTANCE ===')
for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f'  {name:<22}: {imp:.4f}')

hist_probs = cal_clf.predict_proba(X)[:, pos_idx] * 100
live_probs = cal_clf.predict_proba(X_live)[:, pos_idx] * 100

print('\n=== RISK SCORE DISTRIBUTION ===')
print('Historical Dataset Proportions:')
print(f'  Low Risk (<35%)   : {sum(hist_probs < 35)} ({sum(hist_probs < 35)/len(hist_probs)*100:.1f}%)')
print(f'  Med Risk (35-59%) : {sum((hist_probs >= 35) & (hist_probs < 60))} ({sum((hist_probs >= 35) & (hist_probs < 60))/len(hist_probs)*100:.1f}%)')
print(f'  High Risk (>=60%) : {sum(hist_probs >= 60)} ({sum(hist_probs >= 60)/len(hist_probs)*100:.1f}%)')

print('\nLive Deals Risk Distribution:')
print(f'  Min Risk: {min(live_probs):.1f}%, Max Risk: {max(live_probs):.1f}%, Avg Risk: {np.mean(live_probs):.1f}%')
print(f'  Low Risk (<35%)   : {sum(live_probs < 35)} ({sum(live_probs < 35)/len(live_probs)*100:.1f}%)')
print(f'  Med Risk (35-59%) : {sum((live_probs >= 35) & (live_probs < 60))} ({sum((live_probs >= 35) & (live_probs < 60))/len(live_probs)*100:.1f}%)')
print(f'  High Risk (>=60%) : {sum(live_probs >= 60)} ({sum(live_probs >= 60)/len(live_probs)*100:.1f}%)')

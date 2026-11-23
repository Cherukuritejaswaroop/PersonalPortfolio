"""Explainability demo for AIML-501 Artifact 9.
Re-trains the logistic regression model from the Model Evaluation report (same data,
split, and seed) and explains it three ways:
  1) global: permutation importance on the test set (drop in recall when a feature is shuffled)
  2) global: standardized coefficients (direction and strength)
  3) local: per-feature contributions for one missed malignant case (false negative)
Run: python explain_model.py"""
import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix

SEED = 42
data = load_breast_cancer()
X, y, names = data.data, 1 - data.target, list(data.feature_names)   # malignant = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)).fit(X_train, y_train)
pred = model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print("Accuracy %.3f  Precision %.3f  Recall %.3f  TN %d FP %d FN %d TP %d" % (
    accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred), tn, fp, fn, tp))

# 1) Permutation importance (30 repeats), scored by accuracy
pi = permutation_importance(model, X_test, y_test, scoring="accuracy", n_repeats=30, random_state=SEED)
order = np.argsort(pi.importances_mean)[::-1]
print("\nTop permutation importances (mean drop in test accuracy):")
for i in order[:8]:
    print(f"  {names[i]:<25} {pi.importances_mean[i]:.4f} ± {pi.importances_std[i]:.4f}")

# 2) Standardized coefficients
coef = model.named_steps["logisticregression"].coef_[0]
corder = np.argsort(np.abs(coef))[::-1]
print("\nLargest standardized coefficients (+ pushes toward malignant):")
for i in corder[:8]:
    print(f"  {names[i]:<25} {coef[i]:+.3f}")

# 3) Local explanation for the false negatives
scaler = model.named_steps["standardscaler"]
lr = model.named_steps["logisticregression"]
prob = model.predict_proba(X_test)[:, 1]
fn_idx = np.where((y_test == 1) & (pred == 0))[0]
print("\nFalse negatives (test index, predicted P(malignant)):", [(int(i), round(float(prob[i]), 3)) for i in fn_idx])
case = fn_idx[np.argmax(prob[fn_idx])]          # the closest miss
z = scaler.transform(X_test[case:case + 1])[0]
contrib = coef * z
corder2 = np.argsort(np.abs(contrib))[::-1]
print(f"\nLocal explanation for test case {case} (P(malignant) = {prob[case]:.3f}, intercept {lr.intercept_[0]:+.3f}):")
for i in corder2[:8]:
    print(f"  {names[i]:<25} value {X_test[case, i]:>9.4f}  contribution {contrib[i]:+.3f}")

json.dump({
    "metrics": {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
                "recall": recall_score(y_test, pred), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "perm": [[names[i], float(pi.importances_mean[i]), float(pi.importances_std[i])] for i in order[:8]],
    "coef": [[names[i], float(coef[i])] for i in corder[:8]],
    "local": {"case": int(case), "prob": float(prob[case]), "intercept": float(lr.intercept_[0]),
              "contrib": [[names[i], float(X_test[case, i]), float(contrib[i])] for i in corder2[:8]]},
    "fn": [[int(i), float(prob[i])] for i in fn_idx],
}, open("xai_results.json", "w"), indent=2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

# --- LOAD TRAINING DATA ---
train_df = pd.read_csv('training_data.csv')
test_df = pd.read_csv('test_data.csv')

# Remove unnamed or extra columns if present
if 'Unnamed: 133' in train_df.columns:
    train_df = train_df.drop('Unnamed: 133', axis=1)
if 'Unnamed: 133' in test_df.columns:
    test_df = test_df.drop('Unnamed: 133', axis=1)

# Features: All columns except 'prognosis'
feature_cols = [c for c in train_df.columns if c != 'prognosis']

X_train = train_df[feature_cols].astype(int)
y_train = train_df['prognosis']

X_test = test_df[feature_cols].astype(int)
y_test = test_df['prognosis']

# --- LABEL ENCODING ---
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
# Only evaluate on test rows with disease labels seen in train
mask = y_test.isin(le.classes_)
X_test_final = X_test[mask]
y_test_final = y_test[mask]
y_test_enc = le.transform(y_test_final)

print(f"Test samples kept: {len(y_test_final)} / {len(y_test)}")

# --- RANDOM FOREST ---
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_enc)
rf_preds = rf.predict(X_test_final)
rf_probs = rf.predict_proba(X_test_final)

# --- EFFICIENT SVM WITH REDUCED GRIDSEARCH ---
# Small grid for speed (as SVM is slow on large data)
param_grid = {'C': [1, 10], 'gamma': ['scale', 0.01]}
svm = SVC(kernel='rbf', probability=True, random_state=42)
grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train_enc)
svm_best = grid.best_estimator_
svm_preds = svm_best.predict(X_test_final)
svm_probs = svm_best.predict_proba(X_test_final)
print("Best SVM params:", grid.best_params_)

# --- EVALUATION FUNCTION ---
def print_metrics(name, y_true, y_pred):
    print(f"\n{name} Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print("Macro Precision: {:.2%}".format(precision_score(y_true, y_pred, average='macro')))
    print("Weighted Precision: {:.2%}".format(precision_score(y_true, y_pred, average='weighted')))
    print("Macro Recall: {:.2%}".format(recall_score(y_true, y_pred, average='macro')))
    print("Weighted Recall: {:.2%}".format(recall_score(y_true, y_pred, average='weighted')))
    print("Macro F1-score: {:.2%}".format(f1_score(y_true, y_pred, average='macro')))
    print("Weighted F1-score: {:.2%}".format(f1_score(y_true, y_pred, average='weighted')))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

print_metrics("Random Forest (External Test)", y_test_enc, rf_preds)
print_metrics("SVM (External Test)", y_test_enc, svm_preds)

# --- CONFUSION MATRIX ---
def plot_cm(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, annot=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    return cm

cm_rf = plot_cm(y_test_enc, rf_preds, "Random Forest (External Test)")
cm_svm = plot_cm(y_test_enc, svm_preds, "SVM (External Test)")

print("Random Forest Confusion Matrix (Raw):\n", cm_rf)
print("SVM Confusion Matrix (Raw):\n", cm_svm)

# --- FEATURE IMPORTANCE (RF) ---
importances = rf.feature_importances_
indices = importances.argsort()[::-1]
feature_names = np.array(feature_cols)

plt.figure(figsize=(12, 5))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), feature_names[indices[:10]], rotation=45)
plt.tight_layout()
plt.show()
print("Top features:", [feature_names[i] for i in indices[:10]])

# --- SAVE RESULTS ---
results = pd.DataFrame({
    'True_Disease': le.inverse_transform(y_test_enc),
    'RF_Prediction': le.inverse_transform(rf_preds),
    'SVM_Prediction': le.inverse_transform(svm_preds),
    'RF_Confidence': rf_probs.max(axis=1),
    'SVM_Confidence': svm_probs.max(axis=1)
})
results.to_csv('external_structured_test_predictions.csv', index=False)
print("Saved predictions to external_structured_test_predictions.csv")

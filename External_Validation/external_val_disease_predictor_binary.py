import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

# --- LOAD TRAINING DATA ---
train_df = pd.read_csv('Datasets/external_val_training_data_binary.csv')
test_df = pd.read_csv('Datasets/external_val_test_data_binary.csv')

# Remove unnamed or extra columns if present
for col in ['Unnamed: 133', 'Unnamed: 0']:
    if col in train_df.columns:
        train_df = train_df.drop(col, axis=1)
    if col in test_df.columns:
        test_df = test_df.drop(col, axis=1)

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

# --- SVM WITH REDUCED GRIDSEARCH ---
param_grid = {'C': [1, 10], 'gamma': ['scale', 0.01]}
svm = SVC(kernel='rbf', probability=True, random_state=42)
grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train_enc)
svm_best = grid.best_estimator_
svm_preds = svm_best.predict(X_test_final)
svm_probs = svm_best.predict_proba(X_test_final)
print("Best SVM params:", grid.best_params_)

# --- MLPClassifier ---
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train_enc)
mlp_preds = mlp.predict(X_test_final)
try:
    mlp_probs = mlp.predict_proba(X_test_final)
    mlp_conf = mlp_probs.max(axis=1)
except Exception:
    mlp_conf = np.nan * np.ones_like(rf_probs.max(axis=1))

# --- Utility: Get Top N Class Indices by Frequency ---
def get_top_n_classes(y_true, n=20):
    freq = Counter(y_true)
    return [label for label, _ in freq.most_common(n)]

# --- EVALUATION FUNCTION (Top 20 Only) ---
def print_metrics_top_n(name, y_true, y_pred, encoder, top_n=20):
    print(f"\n{name} Metrics (Top {top_n} Classes):")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print("Macro Precision: {:.2%}".format(precision_score(y_true, y_pred, average='macro')))
    print("Weighted Precision: {:.2%}".format(precision_score(y_true, y_pred, average='weighted')))
    print("Macro Recall: {:.2%}".format(recall_score(y_true, y_pred, average='macro')))
    print("Weighted Recall: {:.2%}".format(recall_score(y_true, y_pred, average='weighted')))
    print("Macro F1-score: {:.2%}".format(f1_score(y_true, y_pred, average='macro')))
    print("Weighted F1-score: {:.2%}".format(f1_score(y_true, y_pred, average='weighted')))
    # Only show top_n most common classes in y_true
    top_classes = get_top_n_classes(y_true, top_n)
    target_names = [encoder.classes_[i] for i in top_classes]
    print(f"\nClassification Report (Top {top_n} by support):")
    print(classification_report(
        y_true, y_pred,
        labels=top_classes,
        target_names=target_names,
        zero_division=0
    ))

print_metrics_top_n("Random Forest (External Test)", y_test_enc, rf_preds, le, top_n=20)
print_metrics_top_n("SVM (External Test)", y_test_enc, svm_preds, le, top_n=20)
print_metrics_top_n("MLP (External Test)", y_test_enc, mlp_preds, le, top_n=20)

# --- CONFUSION MATRIX & SAVE (Top 20 Only) ---
def plot_cm_top_n(y_true, y_pred, encoder, top_n=20, model_name="Model", save_path=None):
    top_classes = get_top_n_classes(y_true, top_n)
    cm = confusion_matrix(y_true, y_pred, labels=top_classes)
    class_names = [encoder.classes_[i] for i in top_classes]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                annot=True,
                fmt='d',
                cbar_kws={'label': 'Count'})
    plt.title(f"Confusion Matrix (Top {top_n}) - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()
    return cm

plot_cm_top_n(y_test_enc, rf_preds, le, top_n=20, model_name="Random Forest (External Test)", save_path="Outputs/External_cms/Binary/rf_confusion_matrix_binary.png")
plot_cm_top_n(y_test_enc, svm_preds, le, top_n=20, model_name="SVM (External Test)", save_path="Outputs/External_cms/Binary/svm_confusion_matrix_binary.png")
plot_cm_top_n(y_test_enc, mlp_preds, le, top_n=20, model_name="MLP (External Test)", save_path="Outputs/External_cms/Binary/mlp_confusion_matrix_binary.png")

# --- FEATURE IMPORTANCE (RF only) ---
importances = rf.feature_importances_
indices = importances.argsort()[::-1]
feature_names = np.array(feature_cols)

plt.figure(figsize=(12, 5))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), feature_names[indices[:10]], rotation=45)
plt.tight_layout()
plt.show()
print("Top features (Random Forest):", [feature_names[i] for i in indices[:10]])

# --- SAVE RESULTS ---
results = pd.DataFrame({
    'True_Disease': le.inverse_transform(y_test_enc),
    'RF_Prediction': le.inverse_transform(rf_preds),
    'SVM_Prediction': le.inverse_transform(svm_preds),
    'MLP_Prediction': le.inverse_transform(mlp_preds),
    'RF_Confidence': rf_probs.max(axis=1),
    'SVM_Confidence': svm_probs.max(axis=1),
    'MLP_Confidence': mlp_conf
})
results.to_csv('Outputs/external_structured_test_predictions.csv', index=False)
print("Saved predictions to external_structured_test_predictions.csv")

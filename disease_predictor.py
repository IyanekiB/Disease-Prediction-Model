import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import WordNetLemmatizer
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
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('wordnet')

# --- Preprocessing Function ---
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalpha() and len(w) > 2]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

# --- Load and Preprocess Training Data ---
df_train = pd.read_csv('/mnt/data/symptom-disease-train-dataset.csv')
texts_train = df_train['text'].astype(str).apply(preprocess_text)
labels_train = df_train['label']

# --- Load and Preprocess External Test Data ---
df_test = pd.read_csv('/mnt/data/symptom-disease-test-dataset.csv')
texts_test = df_test['text'].astype(str).apply(preprocess_text)
labels_test = df_test['label']

# --- Vectorizer: Fit only on training data, apply to test data ---
vectorizer = TfidfVectorizer(
    binary=True,
    stop_words='english',
    max_features=200,
    token_pattern=r'\b[a-zA-Z]{3,}\b'
)
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)

# --- Label Encoder: Fit only on training data, apply to test data ---
le = LabelEncoder()
y_train = le.fit_transform(labels_train)

# Map test labels to train label space (skip examples not in train set)
test_label_map = {label: idx for idx, label in enumerate(le.classes_)}
mask = [lbl in test_label_map for lbl in labels_test]
X_test_final = X_test[mask]
labels_test_final = labels_test[mask].values
y_test_final = [test_label_map[lbl] for lbl in labels_test_final]

print(f"Test samples kept: {len(y_test_final)} / {len(labels_test)}")

# ----- Model Training -----
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test_final)
rf_probs = rf.predict_proba(X_test_final)

param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1, 1]}
svm = SVC(kernel='rbf', probability=True, random_state=42)
grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
svm_best = grid.best_estimator_
svm_preds = svm_best.predict(X_test_final)
svm_probs = svm_best.predict_proba(X_test_final)
print("Best SVM params:", grid.best_params_)

# ----- Evaluation -----
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

print_metrics("Random Forest (External Test)", y_test_final, rf_preds)
print_metrics("SVM (External Test)", y_test_final, svm_preds)

# Confusion matrices (heatmaps and raw)
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

cm_rf = plot_cm(y_test_final, rf_preds, "Random Forest (External Test)")
cm_svm = plot_cm(y_test_final, svm_preds, "SVM (External Test)")

print("Random Forest Confusion Matrix (Raw):\n", cm_rf)
print("SVM Confusion Matrix (Raw):\n", cm_svm)

# Top-10 feature importances (Random Forest)
importances = rf.feature_importances_
indices = importances.argsort()[::-1]
feature_names = vectorizer.get_feature_names_out()

plt.figure(figsize=(12, 5))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), feature_names[indices[:10]], rotation=45)
plt.tight_layout()
plt.show()
print("Top features:", [feature_names[i] for i in indices[:10]])

# Save predictions and probabilities for visual error analysis
results = pd.DataFrame({
    'Original_Text': df_test['text'].values[mask],
    'True_Disease': labels_test_final,
    'RF_Prediction': le.inverse_transform(rf_preds),
    'SVM_Prediction': le.inverse_transform(svm_preds),
    'RF_Confidence': rf_probs.max(axis=1),
    'SVM_Confidence': svm_probs.max(axis=1)
})
results.to_csv('external_test_disease_predictions.csv', index=False)
print("Saved predictions to external_test_disease_predictions.csv")

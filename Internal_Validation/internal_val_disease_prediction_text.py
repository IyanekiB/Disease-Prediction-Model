import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import os
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('wordnet')

# ----- Data Loading and Preprocessing -----
df = pd.read_csv('Datasets/Symptom2Disease.csv')
texts = df['text'].astype(str)
labels = df['label']

lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalpha() and len(w) > 2]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

texts_clean = texts.apply(preprocess_text)

vectorizer = TfidfVectorizer(
    binary=True,
    stop_words='english',
    max_features=200,
    token_pattern=r'\b[a-zA-Z]{3,}\b'
)
X = vectorizer.fit_transform(texts_clean)
le = LabelEncoder()
y = le.fit_transform(labels)

# 1. Split into Train+Val and Test first
X_temp, X_test, y_temp, y_test, texts_temp, texts_test = train_test_split(
    X, y, texts_clean, test_size=0.2, random_state=42, stratify=y
)
# 2. Split Train+Val into Train and Validation
X_train, X_val, y_train, y_val, texts_train, texts_val = train_test_split(
    X_temp, y_temp, texts_temp, test_size=0.2, random_state=42, stratify=y_temp
)
# Now: Train=64%, Val=16%, Test=20%

print(f"Train samples: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# ----- Model Training -----
# Random Forest (fit on train, tune if needed using val, final eval on test)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_val_preds = rf.predict(X_val)
rf_test_preds = rf.predict(X_test)

# SVM with parameter tuning (GridSearch on validation set only!)
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1, 1]}
svm = SVC(kernel='rbf', probability=True, random_state=42)
grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
svm_best = grid.best_estimator_

# Evaluate on validation set
svm_val_preds = svm_best.predict(X_val)
print("Best SVM params (by val set):", grid.best_params_)

# Now test on *never-seen* test set
svm_test_preds = svm_best.predict(X_test)

# ----- MLPClassifier (Neural Network) -----
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
mlp_val_preds = mlp.predict(X_val)
mlp_test_preds = mlp.predict(X_test)

# ----- Utility: Get Top N Class Indices by Frequency -----
def get_top_n_classes(y_true, n=20):
    freq = Counter(y_true)
    return [label for label, _ in freq.most_common(n)]

# ----- Evaluation Functions -----
def print_metrics_top_n(name, y_true, y_pred, encoder, top_n=20):
    """
    Prints metrics and classification report for the top_n most frequent classes.
    """
    print(f"\n{name} Metrics:")
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

# ----- Print results (on test set, for top 20) -----
print_metrics_top_n("Random Forest (Test)", y_test, rf_test_preds, le, top_n=20)
print_metrics_top_n("SVM (Test)", y_test, svm_test_preds, le, top_n=20)
print_metrics_top_n("MLP (Test)", y_test, mlp_test_preds, le, top_n=20)

# ----- Confusion matrices (for test set, top 20) -----
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

print("\nRandom Forest Test Confusion Matrix (Top 20):\n", plot_cm_top_n(y_test, rf_test_preds, le, top_n=20, model_name="Random Forest", save_path="Outputs/Internal_cms/internal_rf_confusion_matrix_top20.png"))
print("\nSVM Test Confusion Matrix (Top 20):\n", plot_cm_top_n(y_test, svm_test_preds, le, top_n=20, model_name="SVM", save_path="Outputs/Internal_cms/internal_svm_confusion_matrix_top20.png"))
print("\nMLP Test Confusion Matrix (Top 20):\n", plot_cm_top_n(y_test, mlp_test_preds, le, top_n=20, model_name="MLP", save_path="Outputs/Internal_cms/internal_mlp_confusion_matrix_top20.png"))

# Top-10 feature importances (Random Forest on test set)
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

# Save predictions for visual error analysis
results = pd.DataFrame({
    'Original_Text': texts_test.values,
    'True_Disease': le.inverse_transform(y_test),
    'RF_Prediction': le.inverse_transform(rf_test_preds),
    'SVM_Prediction': le.inverse_transform(svm_test_preds),
    'MLP_Prediction': le.inverse_transform(mlp_test_preds),
})
results.to_csv('Outputs/internal_disease_predictions_split.csv', index=False)
print("Saved predictions to internal_disease_predictions_split.csv")

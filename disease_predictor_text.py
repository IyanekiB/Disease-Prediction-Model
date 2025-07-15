import pandas as pd
import numpy as np
import re
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

# --- Symptom Extraction Utility ---
def extract_symptoms(text):
    if pd.isnull(text) or str(text).strip() == "":
        return []
    text = str(text).lower()
    text = re.sub(r'[^\w\s,]', ',', text)
    tokens = [t.strip() for t in re.split(r',|\n', text) if t.strip()]
    return [re.sub(r'[^a-z0-9_]', '', tok) for tok in tokens if len(tok) > 2]

def build_symptom_vocab(df_train, df_test):
    vocab = set()
    for col in ['text']:
        for s in pd.concat([df_train[col], df_test[col]]):
            for symptom in extract_symptoms(s):
                vocab.add(symptom)
    return sorted(list(vocab))

# --- Load Datasets ---
df_train = pd.read_csv('symptom-disease-train-dataset.csv')
df_test = pd.read_csv('symptom-disease-test-dataset.csv')

# --- Reduce Test Set for Speed and Reset Indexes ---
SAMPLE_N = 50
if len(df_test) > SAMPLE_N:
    df_test = df_test.sample(n=SAMPLE_N, random_state=42).reset_index(drop=True)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# --- Build Symptom Vocabulary ---
symptom_vocab = build_symptom_vocab(df_train, df_test)
print(f"Total unique symptoms: {len(symptom_vocab)}")

# --- Convert Text to Binary Matrix ---
def text_to_binary_matrix(df, vocab):
    arr = np.zeros((len(df), len(vocab)), dtype=int)
    for i, text in enumerate(df['text']):
        found = set(extract_symptoms(text))
        for j, sym in enumerate(vocab):
            if sym in found:
                arr[i, j] = 1
    return pd.DataFrame(arr, columns=vocab)

X_train = text_to_binary_matrix(df_train, symptom_vocab).reset_index(drop=True)
X_test = text_to_binary_matrix(df_test, symptom_vocab).reset_index(drop=True)

le = LabelEncoder()
y_train = le.fit_transform(df_train['label'].astype(str))

# --- Mask Only Test Labels Seen in Training, Align Indices ---
mask = df_test['label'].astype(str).isin(le.classes_).values

X_test_final = X_test.loc[mask].reset_index(drop=True)
y_test_final = df_test.loc[mask, 'label'].astype(str).reset_index(drop=True)
y_test_enc = le.transform(y_test_final)

print(f"Test samples kept: {len(y_test_final)} / {len(df_test)}")

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test_final)
rf_probs = rf.predict_proba(X_test_final)

# --- SVM (Efficient grid) ---
param_grid = {'C': [1], 'gamma': ['scale']}
svm = SVC(kernel='rbf', probability=True, random_state=42)
grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
svm_best = grid.best_estimator_
svm_preds = svm_best.predict(X_test_final)
svm_probs = svm_best.predict_proba(X_test_final)
print("Best SVM params:", grid.best_params_)

# --- Evaluation ---
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

importances = rf.feature_importances_
indices = importances.argsort()[::-1]
feature_names = np.array(symptom_vocab)
plt.figure(figsize=(12, 5))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), feature_names[indices[:10]], rotation=45)
plt.tight_layout()
plt.show()
print("Top features:", [feature_names[i] for i in indices[:10]])

results = pd.DataFrame({
    'Original_Text': df_test.loc[mask, 'text'].reset_index(drop=True),
    'True_Disease': y_test_final.values,
    'RF_Prediction': le.inverse_transform(rf_preds),
    'SVM_Prediction': le.inverse_transform(svm_preds),
    'RF_Confidence': rf_probs.max(axis=1),
    'SVM_Confidence': svm_probs.max(axis=1)
})
results.to_csv('external_mixedsymptom_test_predictions_sampled.csv', index=False)
print("Saved predictions to external_mixedsymptom_test_predictions_sampled.csv")

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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import os

# 1. DOWNLOAD NLTK RESOURCES
nltk.download('punkt_tab')
nltk.download('wordnet')

# 2. TEXT PREPROCESSING
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# 3. LOAD & PREPARE CSV
def load_and_prepare_csv(path):
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    text_col  = next((c for c in df.columns if 'text' in c.lower() or 'desc' in c.lower()), None)
    label_col = next((c for c in df.columns if 'label' in c.lower() or 'disease' in c.lower()), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Cannot find text/label columns in {path}")
    texts  = df[text_col].astype(str).apply(preprocess_text)
    labels = df[label_col].astype(str)
    return texts, labels, df

train_path = 'Datasets/symptom-disease-train-reformat.csv'
test_path  = 'Datasets/symptom-disease-test-reformat.csv'

texts_train, labels_train, df_train = load_and_prepare_csv(train_path)
texts_test,  labels_test,  df_test  = load_and_prepare_csv(test_path)

# 4. VECTORIZE & ENCODE
vectorizer = TfidfVectorizer(
    binary=True,
    stop_words='english',
    max_features=200,
    token_pattern=r'\b[a-zA-Z]{3,}\b'
)
X_train = vectorizer.fit_transform(texts_train)
X_test  = vectorizer.transform(texts_test)
le = LabelEncoder()
y_train = le.fit_transform(labels_train)

# 5. ALIGN TEST LABELS
label_map = {lbl: idx for idx, lbl in enumerate(le.classes_)}
mask = np.array([lbl in label_map for lbl in labels_test])
X_test_final      = X_test[mask]
labels_test_final = labels_test[mask].values
y_test_final      = np.array([label_map[lbl] for lbl in labels_test_final])
texts_test_final  = df_test.iloc[np.where(mask)[0]][texts_test.name]

print(f"Evaluating on {len(y_test_final)} / {len(labels_test)} test samples (labels seen in training).")

# 6. MODEL TRAINING

# 6a. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test_final)
rf_probs = rf.predict_proba(X_test_final)

# 6b. SVM with light GridSearch
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1, 1]}
svm = SVC(kernel='rbf', probability=True, random_state=42)
grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
svm_best  = grid.best_estimator_
svm_preds = svm_best.predict(X_test_final)
svm_probs = svm_best.predict_proba(X_test_final)
print("Best SVM params:", grid.best_params_)

# 6c. MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
mlp_preds = mlp.predict(X_test_final)
# Optional: MLP doesn't have predict_proba with some settings on multi-class. You can skip confidences for MLP if it throws error.
try:
    mlp_probs = mlp.predict_proba(X_test_final)
    mlp_conf = mlp_probs.max(axis=1)
except Exception:
    mlp_conf = np.nan * np.ones_like(rf_probs.max(axis=1))

# 7. METRICS & REPORTING
def print_metrics_top_n(name, y_true, y_pred, encoder, top_n=20):
    print(f"\n=== {name} ===")
    print(f"Accuracy:          {accuracy_score(y_true, y_pred):.2%}")
    print(f"Macro Precision:   {precision_score(y_true, y_pred, average='macro'):.2%}")
    print(f"Weighted Precision:{precision_score(y_true, y_pred, average='weighted'):.2%}")
    print(f"Macro Recall:      {recall_score(y_true, y_pred, average='macro'):.2%}")
    print(f"Weighted Recall:   {recall_score(y_true, y_pred, average='weighted'):.2%}")
    print(f"Macro F1-score:    {f1_score(y_true, y_pred, average='macro'):.2%}")
    print(f"Weighted F1-score: {f1_score(y_true, y_pred, average='weighted'):.2%}")

    # Show only top N classes in y_true
    freq      = Counter(y_true)
    top_labels = [label for label, _ in freq.most_common(top_n)]
    target_names = [encoder.classes_[lbl] for lbl in top_labels]

    print(f"\nClassification Report (Top {top_n} by support):")
    print(classification_report(
        y_true, y_pred,
        labels=top_labels,
        target_names=target_names,
        zero_division=0
    ))

print_metrics_top_n("Random Forest (External Test)", y_test_final, rf_preds, le, top_n=20)
print_metrics_top_n("SVM (External Test)",           y_test_final, svm_preds, le, top_n=20)
print_metrics_top_n("MLP (External Test)",           y_test_final, mlp_preds, le, top_n=20)

# 8. CONFUSION MATRICES (save to Outputs/)
def plot_top_n_cm(y_true, y_pred, encoder, code_to_disease, n=20, model_name="Model", save_path=None):
    freq        = Counter(y_true)
    top_n_vals  = [lab for lab,_ in freq.most_common(n)]
    unique_vals = sorted(set(y_true))
    cm_full     = confusion_matrix(y_true, y_pred, labels=unique_vals)
    val_to_pos  = {val:i for i,val in enumerate(unique_vals)}
    idx         = [val_to_pos[v] for v in top_n_vals]
    cm_n        = cm_full[np.ix_(idx, idx)]
    class_names = []
    for v in top_n_vals:
        code_str = encoder.classes_[v]
        disease  = code_to_disease.get(code_str, code_str)
        class_names.append(disease)
    size = max(8, n * 0.4)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        cm_n,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot=True,
        fmt="d",
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    ax.set_title(f"Top {n} Confusion Matrix – {model_name}")
    ax.set_xlabel("Predicted Disease")
    ax.set_ylabel("Actual Disease")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()
    return cm_n

code_to_disease = {
    "515": "Impetigo",
    "596": "Malaria",
    "72":  "Rheumatoid Arthritis",
    "447":"Myocardial Infarction",
    "394":"Urticaria (Hives)",
    "308":"Migraine",
    "297":"Hemorrhoids",
    "412":"GERD",
    "1035":"UTI",
    "541":"Hepatitis",
    "822":"Psoriasis",
    "33":"Liver Cirrhosis",
    "275":"Dengue Fever",
    "718":"Stroke",
    "1047":"Deep Vein Thrombosis",
    "468":"Acute Liver Failure",
    "700":"Osteoarthritis",
    "504":"Hypoglycemia",
    "766":"Pneumonia",
    "502":"Hyperthyroidism",
}

# Save confusion matrices
plot_top_n_cm(y_test_final, rf_preds,  le, code_to_disease, n=20, model_name="Random Forest",
              save_path="Outputs/External_cms/Text/rf_confusion_matrix_top20.png")
plot_top_n_cm(y_test_final, svm_preds, le, code_to_disease, n=20, model_name="SVM",
              save_path="Outputs/External_cms/Text/svm_confusion_matrix_top20.png")
plot_top_n_cm(y_test_final, mlp_preds, le, code_to_disease, n=20, model_name="MLP",
              save_path="Outputs/External_cms/Text/mlp_confusion_matrix_top20.png")

# 9. FEATURE IMPORTANCES (Random Forest only)
importances   = rf.feature_importances_
indices       = importances.argsort()[::-1]
feature_names = vectorizer.get_feature_names_out()

plt.figure(figsize=(12, 5))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), feature_names[indices[:10]], rotation=45)
plt.tight_layout()
plt.show()
print("Top features (Random Forest):", [feature_names[i] for i in indices[:10]])

# 10. SAVE RESULTS
results = pd.DataFrame({
    'Text':      texts_test_final.values,
    'True':      labels_test_final,
    'RF_Pred':   le.inverse_transform(rf_preds),
    'SVM_Pred':  le.inverse_transform(svm_preds),
    'MLP_Pred':  le.inverse_transform(mlp_preds),
    'RF_Conf':   rf_probs.max(axis=1),
    'SVM_Conf':  svm_probs.max(axis=1),
    'MLP_Conf':  mlp_conf
})
results.to_csv('Outputs/external_test_disease_predictions.csv', index=False)
print("Saved external_test_disease_predictions.csv")

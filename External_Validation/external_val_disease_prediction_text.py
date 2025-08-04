import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter, defaultdict
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer

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

train_path = 'Datasets/Symptom2Disease_augmented_train.csv'
test_path  = 'Datasets/Symptom2Disease_augmented_test_shuffled.csv'

texts_train, labels_train, df_train = load_and_prepare_csv(train_path)
texts_test,  labels_test,  df_test  = load_and_prepare_csv(test_path)

vectorizer = TfidfVectorizer(
    binary=True,
    stop_words='english',
    max_features=200,
    token_pattern=r'\b[a-zA-Z]{3,}\b'
)

X_train_full = vectorizer.fit_transform(texts_train)
X_test_full  = vectorizer.transform(texts_test)
le = LabelEncoder()
y_train_full = le.fit_transform(labels_train)

# --- Per-class Support Plot (on TRAIN set) ---
support_count = Counter(y_train_full)
classes_sorted = [i for i, _ in support_count.most_common(10)]
class_labels = [le.classes_[i] for i in classes_sorted]
support_vals = [support_count[i] for i in classes_sorted]
plt.figure(figsize=(14, 6))
sns.barplot(x=support_vals, y=class_labels, orient='h', color='skyblue')
plt.title('External: Per-class Support (Top 10 Diseases, Training Set)')
plt.xlabel('Number of Samples')
plt.ylabel('Disease')
plt.tight_layout()
plt.savefig("Outputs/external_val_per_class_support.png", dpi=300)
plt.close()
print("Saved per-class support plot as Outputs/external_val_per_class_support.png")

# --- Helper Functions ---
def get_top_n_classes(y_true, n=20):
    freq = Counter(y_true)
    return [label for label, _ in freq.most_common(n)]

def print_metrics_top_n(name, y_true, y_pred, encoder, top_n=20):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_prec = precision_score(y_true, y_pred, average='macro')
    macro_rec = recall_score(y_true, y_pred, average='macro')
    return acc, macro_f1, macro_prec, macro_rec

def plot_cm_top_n_agg(cm_sum, class_names, model_name="Model", save_path=None):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_sum, cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                annot=True,
                fmt='d',
                cbar_kws={'label': 'Total Count Across Runs'})
    plt.title(f"Confusion Matrix (Top 20, Summed over 10 seeds) - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    plt.close()

def add_confusion_pairs(cm, class_names, global_counter):
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                key = (class_names[i], class_names[j])
                global_counter[key] += cm[i, j]

def plot_confused_pairs_bar_global(pair_counts, model_name, save_path=None, top_n=7):
    pairs_sorted = sorted(pair_counts.items(), key=lambda x: -x[1])[:top_n]
    pairs = [f"{a}→{b}" for (a, b), _ in pairs_sorted]
    counts = [c for _, c in pairs_sorted]
    plt.figure(figsize=(9, 5))
    sns.barplot(x=counts, y=pairs, orient='h', color='tomato')
    plt.title(f"Top {top_n} Most Confused Disease Pairs — {model_name} (All Seeds)")
    plt.xlabel('Count (total over 10 seeds)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Most confused pairs barplot saved to {save_path}")
    plt.close()

# --- (2) Run 10 SEEDS and Aggregate Results ---
all_metrics = {'RF': [], 'SVM': [], 'MLP': []}
confusion_rf_sum = None
confusion_svm_sum = None
confusion_mlp_sum = None
pair_rf = defaultdict(int)
pair_svm = defaultdict(int)
pair_mlp = defaultdict(int)
seeds = list(range(10))

for seed in seeds:
    # For each run, shuffle train/test by seed (labels in test always aligned to train)
    # Align test to only those classes in train for each seed
    label_map = {lbl: idx for idx, lbl in enumerate(le.classes_)}
    mask = np.array([lbl in label_map for lbl in labels_test])
    X_test = X_test_full[mask]
    labels_test_seed = labels_test[mask].values
    y_test = np.array([label_map[lbl] for lbl in labels_test_seed])
    # Train on full train, always the same split (simulate: just change seed for models)
    X_train = X_train_full
    y_train = y_train_full

    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(X_train, y_train)
    rf_test_preds = rf.predict(X_test)

    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1, 1]}
    svm = SVC(kernel='rbf', probability=True, random_state=seed)
    grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    svm_best = grid.best_estimator_
    svm_test_preds = svm_best.predict(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed)
    mlp.fit(X_train, y_train)
    mlp_test_preds = mlp.predict(X_test)

    # Metrics for model comparison (top 20 classes)
    rf_stats = print_metrics_top_n("Random Forest (External)", y_test, rf_test_preds, le, top_n=20)
    svm_stats = print_metrics_top_n("SVM (External)", y_test, svm_test_preds, le, top_n=20)
    mlp_stats = print_metrics_top_n("MLP (External)", y_test, mlp_test_preds, le, top_n=20)
    all_metrics['RF'].append(rf_stats)
    all_metrics['SVM'].append(svm_stats)
    all_metrics['MLP'].append(mlp_stats)

    # Summed confusion matrices (top 20 classes only)
    top_classes = get_top_n_classes(y_test, 20)
    class_names = [le.classes_[i] for i in top_classes]
    cm_rf = confusion_matrix(y_test, rf_test_preds, labels=top_classes)
    cm_svm = confusion_matrix(y_test, svm_test_preds, labels=top_classes)
    cm_mlp = confusion_matrix(y_test, mlp_test_preds, labels=top_classes)
    if confusion_rf_sum is None:
        confusion_rf_sum = np.zeros_like(cm_rf)
        confusion_svm_sum = np.zeros_like(cm_svm)
        confusion_mlp_sum = np.zeros_like(cm_mlp)
    confusion_rf_sum += cm_rf
    confusion_svm_sum += cm_svm
    confusion_mlp_sum += cm_mlp

    # Error pairs (diagonal-zeroed)
    cm_rf_err = cm_rf.copy()
    np.fill_diagonal(cm_rf_err, 0)
    cm_svm_err = cm_svm.copy()
    np.fill_diagonal(cm_svm_err, 0)
    cm_mlp_err = cm_mlp.copy()
    np.fill_diagonal(cm_mlp_err, 0)
    add_confusion_pairs(cm_rf_err, class_names, pair_rf)
    add_confusion_pairs(cm_svm_err, class_names, pair_svm)
    add_confusion_pairs(cm_mlp_err, class_names, pair_mlp)

# --- (3) Model Comparison: Aggregate and Plot ---
def aggregate_metrics(metric_list):
    arr = np.array(metric_list)
    return arr.mean(axis=0), arr.std(axis=0)

rf_mean, rf_std = aggregate_metrics(all_metrics['RF'])
svm_mean, svm_std = aggregate_metrics(all_metrics['SVM'])
mlp_mean, mlp_std = aggregate_metrics(all_metrics['MLP'])

metrics_names = ['Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall']
for i, name in enumerate(metrics_names):
    plt.figure(figsize=(7, 4))
    plt.bar(['RF', 'SVM', 'MLP'],
            [rf_mean[i], svm_mean[i], mlp_mean[i]],
            yerr=[rf_std[i], svm_std[i], mlp_std[i]],
            capsize=10)
    plt.ylabel(name)
    plt.title(f'Model Comparison: {name} (Mean ± Std over 10 runs)')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"Outputs/Comparison/external_model_comparison_{name.replace(' ', '_').lower()}.png", dpi=300)
    plt.close()
    print(f"Saved model comparison barplot for {name}.")

# --- (4) Confusion Matrices: Aggregate and Save ---
plot_cm_top_n_agg(confusion_rf_sum, class_names, model_name="Random Forest",
                  save_path="Outputs/External_cms/Text/external_rf_confusion_matrix_top20_SUM.png")
plot_cm_top_n_agg(confusion_svm_sum, class_names, model_name="SVM",
                  save_path="Outputs/External_cms/Text/external_svm_confusion_matrix_top20_SUM.png")
plot_cm_top_n_agg(confusion_mlp_sum, class_names, model_name="MLP",
                  save_path="Outputs/External_cms/Text/external_mlp_confusion_matrix_top20_SUM.png")

# --- (5) Error Analysis: Most Confused Pairs Aggregated ---
plot_confused_pairs_bar_global(pair_rf, "Random Forest",
    save_path="Outputs/Error_Analysis/external_rf_confused_bar_AGGREGATED.png", top_n=7)
plot_confused_pairs_bar_global(pair_svm, "SVM",
    save_path="Outputs/Error_Analysis/external_svm_confused_bar_AGGREGATED.png", top_n=7)
plot_confused_pairs_bar_global(pair_mlp, "MLP",
    save_path="Outputs/Error_Analysis/external_mlp_confused_bar_AGGREGATED.png", top_n=7)

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

# --- (6) Save summary tables ---
summary = pd.DataFrame({
    'Model': ['RF', 'SVM', 'MLP'],
    'Accuracy Mean': [rf_mean[0], svm_mean[0], mlp_mean[0]],
    'Accuracy Std': [rf_std[0], svm_std[0], mlp_std[0]],
    'Macro F1 Mean': [rf_mean[1], svm_mean[1], mlp_mean[1]],
    'Macro F1 Std': [rf_std[1], svm_std[1], mlp_std[1]],
    'Macro Precision Mean': [rf_mean[2], svm_mean[2], mlp_mean[2]],
    'Macro Precision Std': [rf_std[2], svm_std[2], mlp_std[2]],
    'Macro Recall Mean': [rf_mean[3], svm_mean[3], mlp_mean[3]],
    'Macro Recall Std': [rf_std[3], svm_std[3], mlp_std[3]],
})
summary.to_csv("Outputs/external_val_metric_summary.csv", index=False)
print("Saved summary metric table as Outputs/external_val_metric_summary.csv")

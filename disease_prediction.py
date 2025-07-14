import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# ----- Data Loading and Preprocessing -----
df = pd.read_csv('Symptom2Disease.csv')
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

# Try different values for max_features here (e.g., 100, 200, 300)
vectorizer = TfidfVectorizer(
    binary=True,
    stop_words='english',
    max_features=200, 
    token_pattern=r'\b[a-zA-Z]{3,}\b'
)
X = vectorizer.fit_transform(texts_clean)

le = LabelEncoder()
y = le.fit_transform(labels)

# Split into train/test sets
X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
    X, y, texts_clean, test_size=0.2, random_state=42, stratify=y
)

# ----- Model Training -----
# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)

# SVM with parameter tuning
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1, 1]}
svm = SVC(kernel='rbf', probability=True, random_state=42)
grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
svm_best = grid.best_estimator_
svm_preds = svm_best.predict(X_test)
svm_probs = svm_best.predict_proba(X_test)
print("Best SVM params:", grid.best_params_)

# ----- Evaluation Functions -----
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

print_metrics("Random Forest", y_test, rf_preds)
print_metrics("SVM", y_test, svm_preds)

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

cm_rf = plot_cm(y_test, rf_preds, "Random Forest")
cm_svm = plot_cm(y_test, svm_preds, "SVM")

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
    'Original_Text': texts_test.values,
    'True_Disease': le.inverse_transform(y_test),
    'RF_Prediction': le.inverse_transform(rf_preds),
    'SVM_Prediction': le.inverse_transform(svm_preds),
    'RF_Confidence': rf_probs.max(axis=1),
    'SVM_Confidence': svm_probs.max(axis=1)
})
results.to_csv('disease_predictions.csv', index=False)
print("Saved predictions to disease_predictions.csv")

# ----- Interactive User Prediction and Evaluation -----
# Store all interactive predictions for evaluation
user_true_labels = []
user_rf_preds = []
user_svm_preds = []

def predict_user_input():
    while True:
        user_text = input("\nEnter your symptom description (or type 'exit' to finish):\n")
        if user_text.lower().strip() == 'exit':
            break
        true_disease = input("What is the *true* disease label? (for evaluation):\n").strip()
        # Preprocess input
        text_clean = preprocess_text(user_text)
        X_input = vectorizer.transform([text_clean])
        rf_pred = le.inverse_transform(rf.predict(X_input))[0]
        svm_pred = le.inverse_transform(svm_best.predict(X_input))[0]
        print(f"\nRandom Forest prediction: {rf_pred}")
        print(f"SVM prediction: {svm_pred}")
        # Show top-3 likely diseases from Random Forest
        top3_idx = rf.predict_proba(X_input)[0].argsort()[-3:][::-1]
        print("Top 3 probable diseases (RF):")
        for i in top3_idx:
            print(f" - {le.inverse_transform([i])[0]} ({rf.predict_proba(X_input)[0][i]*100:.1f}%)")
        # Record for confusion matrix and metrics
        user_true_labels.append(true_disease)
        user_rf_preds.append(rf_pred)
        user_svm_preds.append(svm_pred)

print("\n---- User Input Mode ----")
print("Enter your symptom descriptions to get predictions. Type 'exit' to stop and evaluate.")
predict_user_input()

if user_true_labels:
    # Convert user true labels to numeric
    user_true_y = le.transform(user_true_labels)
    user_rf_y = le.transform(user_rf_preds)
    user_svm_y = le.transform(user_svm_preds)

    print("\nUser Input Evaluation (Random Forest):")
    print_metrics("Random Forest (User)", user_true_y, user_rf_y)
    print("\nUser Input Evaluation (SVM):")
    print_metrics("SVM (User)", user_true_y, user_svm_y)

    # Plot confusion matrices for user session
    plot_cm(user_true_y, user_rf_y, "Random Forest (User Input)")
    plot_cm(user_true_y, user_svm_y, "SVM (User Input)")
else:
    print("No user predictions to evaluate.")

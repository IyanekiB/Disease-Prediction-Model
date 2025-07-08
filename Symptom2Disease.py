import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. Load the Data
df = pd.read_csv('Symptom2Disease.csv')

# 2. Basic Preprocessing
# We will use the 'label' column for diseases and 'text' for the symptoms description.
labels = df['label']
texts = df['text']

# 3. Encode Disease Labels as Numbers
le = LabelEncoder()
y = le.fit_transform(labels)

# 4. Convert Free Text to Binary Features (Bag-of-Words)
# CountVectorizer will convert text into a binary (0/1) word occurrence matrix.
vectorizer = CountVectorizer(binary=True, stop_words='english', max_features=100)  # Use the 100 most common words
X = vectorizer.fit_transform(texts)

print("Feature names (words):", vectorizer.get_feature_names_out())

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)

# 7. Train SVM (with parameter tuning)
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1, 1]}
svm = SVC(kernel='rbf', probability=True, random_state=42)
grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
svm_best = grid.best_estimator_
svm_preds = svm_best.predict(X_test)
svm_probs = svm_best.predict_proba(X_test)
print("Best SVM params:", grid.best_params_)

# 8. Evaluation Functions
def print_metrics(name, y_true, y_pred):
    print(f"\n{name} Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print("Macro Precision: {:.2%}".format(precision_score(y_true, y_pred, average='macro')))
    print("Macro Recall: {:.2%}".format(recall_score(y_true, y_pred, average='macro')))
    print("Macro F1-score: {:.2%}".format(f1_score(y_true, y_pred, average='macro')))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

print_metrics("Random Forest", y_test, rf_preds)
print_metrics("SVM", y_test, svm_preds)

# 9. Confusion Matrix Plots
def plot_cm(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, annot=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

plot_cm(y_test, rf_preds, "Random Forest")
plot_cm(y_test, svm_preds, "SVM")

# 10. Feature Importances (Random Forest)
importances = rf.feature_importances_
indices = importances.argsort()[::-1]
feature_names = vectorizer.get_feature_names_out()

plt.figure(figsize=(10, 5))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), feature_names[indices[:10]], rotation=45)
plt.tight_layout()
plt.show()

# 11. Save Predictions and Probabilities
results = pd.DataFrame({
    'True_Disease': le.inverse_transform(y_test),
    'RF_Prediction': le.inverse_transform(rf_preds),
    'SVM_Prediction': le.inverse_transform(svm_preds),
    'RF_Confidence': rf_probs.max(axis=1),
    'SVM_Confidence': svm_probs.max(axis=1)
})
results.to_csv('disease_predictions.csv', index=False)
print("Saved predictions to disease_predictions.csv")

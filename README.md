# Disease Prediction Model

Early and accurate disease prediction from textual symptom descriptions can support triage and clinical decision-making, especially in digital and telehealth applications. In this work, I systematically benchmark Random Forest, Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP) classifiers using the public Symptom2Disease dataset. I focus on addressing class imbalance, reproducibility, and error analysis through a rigorous multi-seed evaluation protocol. Both internal (stratified train/validation/test) and external validation splits are used to assess generalization. My analysis includes visualization of class distribution, confusion matrices, and the most frequently confused disease pairs, providing actionable insights into model behavior. The results highlight the strengths and limitations of each approach for clinical NLP and suggest directions for future work.

## Table of Contents
- [Performance Results](#performance-results)
- [Important Validation Notes](#important-validation-notes)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Validation Strategy](#validation-strategy)
- [Results & Limitations](#results--limitations)
- [Future Work](#future-work)

## Overview

This systematic benchmarking study evaluates three machine learning approaches for disease prediction from symptom descriptions. The project implements rigorous evaluation methodologies including multi-seed validation, class imbalance analysis, and comprehensive error assessment to provide actionable insights for clinical NLP applications. Initial results demonstrate promising performance across all algorithms, though additional validation is required for clinical deployment.

## Performance Results

### Initial Model Performance
- **Preliminary results show >99% accuracy** across all three algorithms
- **Consistent performance** observed across random splits with minimal standard deviation
- **Apparent strong generalization** between internal and external validation sets
- **10-seed averaging** demonstrates statistical consistency in initial testing

### Model Comparison Summary
| Model | Internal Accuracy | External Accuracy | Initial Ranking |
|-------|------------------|-------------------|-----------------|
| **SVM** | 99.82% ± 0.15% | 99.80% ± 0.00% | Highest |
| **Random Forest** | 99.40% ± 0.20% | 99.54% ± 0.05% | Most Stable |
| **MLP** | 99.40% ± 0.18% | 99.43% ± 0.06% | Balanced |

## Important Validation Notes

**Critical Disclaimer**: The >99% accuracy results shown above represent initial testing on specific datasets and require additional comprehensive validation before drawing definitive conclusions about model performance.

### Required Additional Testing
- **Independent dataset validation** on diverse patient populations
- **Cross-institutional testing** to verify generalizability
- **Clinical expert review** of prediction accuracy and medical validity
- **Edge case analysis** to identify potential failure modes
- **Bias assessment** across demographic groups and rare diseases
- **Real-world deployment testing** under clinical conditions

### Current Limitations
- **Limited scope validation** - results may not generalize to all medical scenarios
- **Dataset-specific performance** - accuracy may vary with different symptom patterns
- **Need for clinical validation** - medical expert review required for clinical applicability
- **Potential overfitting** despite external validation - broader testing needed

## Features

- **Multi-Algorithm Implementation**: Random Forest (RF), Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP)
- **Promising Initial Results**: >99% accuracy in preliminary testing (requires further validation)
- **Dual Validation Framework**: Internal and external validation methodologies
- **Statistical Analysis**: 10-seed averaging with confidence intervals
- **Data Augmentation**: Enhanced dataset through systematic augmentation techniques
- **Performance Monitoring**: Comprehensive metric tracking and analysis
- **Structured Outputs**: Organized results for clinical interpretation

## Dataset

### Primary Dataset
- **Source**: `Symptom2Disease.csv` - Original symptom-to-disease mapping dataset
- **Augmented Versions**: Multiple enhanced datasets for model training
  - `Symptom2Disease_augmented_v1.csv`
  - `Symptom2Disease_augmented_v2.csv` 
  - `Symptom2Disease_augmented_v3.csv`
  - `Symptom2Disease_augmented_v4.csv`

### Validation Data
- **External Test Sets**: Separate datasets for initial generalization testing
- **Binary Classification**: Specialized validation for disease presence/absence prediction
- **Note**: Additional diverse datasets needed for comprehensive validation

## Model Architecture

### Implemented Algorithms

#### 1. Support Vector Machine (SVM) - **Preliminary Top Performer**
- **Initial Internal Performance**: 99.82% ± 0.15% accuracy
- **Initial External Performance**: 99.80% ± 0.00% accuracy
- **Status**: Requires broader validation to confirm performance

#### 2. Random Forest (RF)
- **Initial Internal Performance**: 99.40% ± 0.20% accuracy  
- **Initial External Performance**: 99.54% ± 0.05% accuracy
- **Status**: Shows stability but needs clinical validation

#### 3. Multi-Layer Perceptron (MLP)
- **Initial Internal Performance**: 99.40% ± 0.18% accuracy
- **Initial External Performance**: 99.43% ± 0.06% accuracy
- **Status**: Balanced performance pending further testing

## Project Structure

```
Disease-Prediction-Model/
├── Datasets/
│   ├── Symptom2Disease.csv                    # Original dataset
│   ├── Symptom2Disease_augmented_*.csv        # Augmented training data
│   ├── external_val_test_data_binary.csv      # External test set
│   └── external_val_training_data_binary.csv  # External training set
│
├── External_Validation/
│   ├── external_val_disease_prediction_*.csv  # External validation predictions
│   └── external_val_disease_predictor_*.csv   # External validation models
│
├── Internal_Validation/
│   └── internal_val_disease_prediction_*.csv  # Internal validation results
│
└── Outputs/
    ├── Comparison/                            # Model comparison results
    ├── Error_Analysis/                        # Initial error analysis
    ├── External_cms/                          # External confusion matrices
    ├── Internal_cms/                          # Internal confusion matrices
    ├── external_structured_test_predictions.csv
    ├── external_val_metric_summary.csv       # Preliminary performance metrics
    ├── external_val_per_class_support.png    # Per-class performance analysis
    ├── internal_val_metric_summary.csv       # Initial performance metrics
    └── internal_val_per_class_support.png    # Internal per-class analysis
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Required Packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install jupyter notebook plotly
pip install imbalanced-learn
```

### Clone Repository
```bash
git clone https://github.com/IyanekiB/Disease-Prediction-Model.git
cd Disease-Prediction-Model
```

## Usage

### 1. Model Training - Internal Validation
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load and preprocess augmented dataset
df = pd.read_csv('Datasets/Symptom2Disease_augmented.csv')
texts_clean = df['text'].apply(preprocess_text)  # Custom preprocessing function

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    binary=True, stop_words='english', max_features=200,
    token_pattern=r'\b[a-zA-Z]{3,}\b'
)
X = vectorizer.fit_transform(texts_clean)
y = LabelEncoder().fit_transform(df['label'])

# 10-seed cross-validation for robust evaluation
all_metrics = {'RF': [], 'SVM': [], 'MLP': []}
for seed in range(10):
    # Stratified train/validation/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=seed, stratify=y_temp
    )
    
    # Train models with different random seeds
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    svm = SVC(kernel='rbf', probability=True, random_state=seed)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed)
    
    # Grid search for SVM hyperparameters
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1, 1]}
    grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    
    # Fit and evaluate models
    rf.fit(X_train, y_train)
    grid.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
```

### 2. External Validation
```python
# Load separate external test dataset
train_texts, train_labels = load_and_prepare_csv('Datasets/Symptom2Disease_augmented_train.csv')
test_texts, test_labels = load_and_prepare_csv('Datasets/Symptom2Disease_augmented_test_shuffled.csv')

# Fit vectorizer on training data only
X_train_full = vectorizer.fit_transform(train_texts)
X_test_full = vectorizer.transform(test_texts)  # Transform test with training vocab

# 10-seed evaluation for statistical robustness
for seed in range(10):
    # Train models on full training set
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    svm_best = GridSearchCV(svm, param_grid, cv=3, random_state=seed).fit(X_train_full, y_train_full).best_estimator_
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed)
    
    # Evaluate on external test set
    rf_preds = rf.predict(X_test_full)
    svm_preds = svm_best.predict(X_test_full) 
    mlp_preds = mlp.predict(X_test_full)
```

### 2. Validation Testing
```python
# Reproduce initial results (requires additional validation)
# Current 10-seed validation shows consistency
# Additional testing needed for clinical confidence
```

### 3. Further Testing Protocol
```python
# Implement additional validation measures:
# 1. Test on independent medical datasets
# 2. Clinical expert review of predictions
# 3. Cross-institutional validation
# 4. Bias and fairness assessment
```

## Model Performance

### Initial Testing Results

#### Preliminary Metrics
- **Initial accuracy >99%** across all models on current datasets
- **Macro F1 scores >99%** in preliminary testing
- **Consistent precision & recall** in initial validation
- **Low standard deviations** in 10-seed testing

#### Validation Status
- **Internal validation completed** with promising results
- **External validation performed** on available datasets
- **Additional validation required** for clinical deployment
- **Broader testing needed** to confirm generalizability

#### Statistical Notes
- **10-seed averaging** shows consistency in current datasets
- **Minimal variance** observed in preliminary testing
- **Results may not generalize** to all medical scenarios
- **Further statistical validation** required for confidence

## Validation Strategy

### Current Validation (Completed)
- **Internal Validation**: Cross-validation on augmented datasets
- **External Validation**: Testing on separate available datasets
- **Performance Range**: 99.40% - 99.82% in initial testing

### Required Additional Validation
- **Independent Medical Datasets**: Testing on diverse, real-world medical data
- **Clinical Expert Review**: Medical professional validation of predictions
- **Multi-Institutional Testing**: Validation across different healthcare systems
- **Demographic Bias Testing**: Performance assessment across patient groups
- **Rare Disease Testing**: Evaluation on uncommon conditions
- **Temporal Validation**: Testing on data from different time periods

## Results & Limitations

### Initial Promising Results
- **All models show >99% accuracy** in preliminary testing
- **SVM demonstrates highest initial performance**
- **Consistent results across random splits**
- **Good initial generalization** between internal and external sets

### Critical Limitations & Requirements
1. **Limited Dataset Scope**: Results based on specific datasets, may not generalize
2. **Need Clinical Validation**: Medical expert review required before deployment
3. **Demographic Bias Unknown**: Testing across diverse populations needed
4. **Real-World Performance Uncertain**: Clinical environment testing required
5. **Edge Case Analysis Pending**: Rare diseases and unusual symptoms need evaluation
6. **Interpretability Assessment**: Clinical decision-making transparency needed

## Future Work

### Immediate Priorities
1. **Comprehensive Validation**: Execute additional testing protocols
2. **Clinical Partnership**: Collaborate with medical institutions for validation
3. **Dataset Expansion**: Acquire diverse, real-world medical datasets
4. **Expert Review**: Engage medical professionals for accuracy assessment

### Long-term Development
- **Regulatory Preparation**: Work toward clinical approval standards
- **Interpretability Enhancement**: Improve model explainability for clinicians
- **Integration Planning**: Prepare for electronic health record integration
- **Continuous Monitoring**: Implement performance tracking in deployment

---

**Research Note**: While initial results are promising with >99% accuracy, comprehensive validation is essential before drawing conclusions about clinical applicability. This project serves as a foundation for rigorous medical AI validation studies.

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the original Symptom2Disease CSV
df = pd.read_csv('Symptom2Disease.csv')

# Stratified split by 'label' (disease)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label'],  # Ensures each disease is represented proportionally
    random_state=42
)

print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")

# Save to new CSVs for external validation
train_df.to_csv('symptom-disease-train-dataset.csv', index=False)
test_df.to_csv('symptom-disease-test-dataset.csv', index=False)

print("Saved symptom-disease-train-dataset.csv and symptom-disease-test-dataset.csv")

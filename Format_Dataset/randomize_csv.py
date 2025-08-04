import pandas as pd

# Load the test data
df = pd.read_csv('Datasets/Symptom2Disease_augmented_test.csv')

# Shuffle rows (preserves the correct label-text relationship)
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save shuffled test set
shuffled_df.to_csv('Datasets/Symptom2Disease_augmented_test_shuffled.csv', index=False)
print("Shuffled test file saved as 'Symptom2Disease_augmented_test_shuffled.csv'")

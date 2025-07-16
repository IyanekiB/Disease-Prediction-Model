import pandas as pd

# Paths to your files
train_in = 'symptom-disease-train-dataset_1.csv'
test_in = 'symptom-disease-test-dataset_1.csv'
train_out = 'symptom-disease-train-reformat.csv'
test_out = 'symptom-disease-test-reformat.csv'

def reformat_file(input_csv, output_csv):
    # Try to autodetect which columns are disease and which are symptom text
    df = pd.read_csv(input_csv)
    # Lowercase column names for easy matching
    cols = [c.lower() for c in df.columns]
    # Find label column
    label_col = None
    for possible in ['label', 'disease', 'diagnosis']:
        if possible in cols:
            label_col = df.columns[cols.index(possible)]
            break
    # Find text column
    text_col = None
    for possible in ['text', 'desc', 'symptom', 'symptoms']:
        if possible in cols:
            text_col = df.columns[cols.index(possible)]
            break
    if label_col is None or text_col is None:
        raise ValueError(f"Could not autodetect columns in {input_csv}. Found: {df.columns}")
    # Rename and keep only the right columns
    new_df = df[[label_col, text_col]].copy()
    new_df.columns = ['label', 'text']
    new_df.to_csv(output_csv, index=False)
    print(f"Saved reformatted: {output_csv} (rows: {len(new_df)})")

reformat_file(train_in, train_out)
reformat_file(test_in, test_out)

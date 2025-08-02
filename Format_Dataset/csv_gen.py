import pandas as pd
import numpy as np
import random
import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download("wordnet")

# Load data
orig_df = pd.read_csv("Datasets/Symptom2Disease.csv")
orig_df = orig_df[['label', 'text']].drop_duplicates().dropna()

# Clean text
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

orig_df['text'] = orig_df['text'].apply(clean_text)

# --- Augmentation Functions ---
def synonym_replacement(text, n=2):
    words = word_tokenize(text)
    candidates = [i for i, w in enumerate(words) if wn.synsets(w)]
    random.shuffle(candidates)
    replaced = 0
    for idx in candidates:
        syns = wn.synsets(words[idx])
        lemmas = set()
        for s in syns:
            for l in s.lemmas():
                if l.name().lower() != words[idx].lower():
                    lemmas.add(l.name().replace("_", " "))
        if lemmas:
            new_word = random.choice(list(lemmas))
            words[idx] = new_word
            replaced += 1
        if replaced >= n:
            break
    return " ".join(words)

def random_deletion(text, p=0.15):
    words = word_tokenize(text)
    if len(words) == 1:
        return text
    new_words = [w for w in words if random.random() > p]
    if len(new_words) == 0:
        new_words = [random.choice(words)]
    return " ".join(new_words)

def random_swap(text, n=2):
    words = word_tokenize(text)
    for _ in range(n):
        if len(words) < 2:
            break
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)

def augment_text(text):
    return [
        synonym_replacement(text, n=2),
        random_deletion(text, p=0.12),
        random_swap(text, n=2)
    ]

# --- Generate Augmented Dataset ---
augmented_rows = []
for idx, row in orig_df.iterrows():
    augmented_rows.append({'label': row['label'], 'text': row['text']})
    for aug_text in augment_text(row['text']):
        augmented_rows.append({'label': row['label'], 'text': aug_text})

# Repeat augmentations if not enough samples
augmented_df = pd.DataFrame(augmented_rows)
while len(augmented_df) < 10000:
    extra_rows = []
    for idx, row in orig_df.iterrows():
        for aug_text in augment_text(row['text']):
            extra_rows.append({'label': row['label'], 'text': aug_text})
            if len(augmented_df) + len(extra_rows) >= 10000:
                break
        if len(augmented_df) + len(extra_rows) >= 10000:
            break
    augmented_df = pd.concat([augmented_df, pd.DataFrame(extra_rows)], ignore_index=True)

# Shuffle and index
augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
augmented_df['index'] = augmented_df.index
augmented_df = augmented_df[['index', 'label', 'text']]

# Split and Save
train_size = int(0.8 * len(augmented_df))
train_df = augmented_df.iloc[:train_size].copy()
test_df  = augmented_df.iloc[train_size:].copy()

augmented_df.to_csv('Datasets/Symptom2Disease_augmented.csv', index=False)
train_df.to_csv('Datasets/Symptom2Disease_augmented_train.csv', index=False)
test_df.to_csv('Datasets/Symptom2Disease_augmented_test.csv', index=False)

print('Done! Files written.')

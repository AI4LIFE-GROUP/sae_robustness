import pandas as pd
import random
from collections import defaultdict
from itertools import combinations

# Load the CSV
df = pd.read_csv("ag_news_full.csv")  # replace with your actual file path

# Group texts by class label
class_to_texts = defaultdict(list)
for _, row in df.iterrows():
    class_to_texts[row["Class Index"]].append(row["Title"])

# Set random seed for reproducibility
# random.seed(42)
# print(len(class_to_texts[4]))
sample_size = 300
# Prepare to sample: all 6 unique class pairs
label_pairs = list(combinations(range(1, 5), 2))
# print(label_pairs)
pairs_per_type = sample_size // len(label_pairs)

# Sample pairs
pairs = []
for l1, l2 in label_pairs:
    
    texts1 = random.sample(class_to_texts[l1], pairs_per_type)
    texts2 = random.sample(class_to_texts[l2], pairs_per_type)
    pairs.extend([(x1, x2, l1, l2) for x1, x2 in zip(texts1, texts2)])

# Shuffle the final pairs
random.shuffle(pairs)

# Convert to DataFrame
pairs_df = pd.DataFrame(pairs, columns=["x1", "x2", "label1", "label2"])
# Remove leading/trailing double and single quotes from x1 and x2
pairs_df["x1"] = pairs_df["x1"].str.replace(r'^"+|"+$', '', regex=True)
pairs_df["x2"] = pairs_df["x2"].str.replace(r'^"+|"+$', '', regex=True)
pairs_df["x1"] = pairs_df["x1"].str.replace(r"^'+|'+$", '', regex=True)
pairs_df["x2"] = pairs_df["x2"].str.replace(r"^'+|'+$", '', regex=True)
# View or save
print(pairs_df.head())
pairs_df.to_csv("ag_news_sampled.csv", index=False)
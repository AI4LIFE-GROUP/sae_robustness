import pandas as pd
import random
from collections import defaultdict
from itertools import combinations

# Load the CSV
df = pd.read_csv("ag_news.csv")  # replace with your actual file path

for i in range(len(df)):
    df.at[i, 'x1'] = str(df.at[i, 'x1']).strip() + '.'
    df.at[i, 'x2'] = str(df.at[i, 'x2']).strip() + '.'
df.to_csv("ag_news_adjusted.csv", index=False)
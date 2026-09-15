#!/usr/bin/env python3
"""One-time script to split ConvoLearn into EFA, CFA, and human validation sets."""

import random
import pandas as pd
from pathlib import Path

SEED = 42
EFA_SIZE = 1000
HUMAN_SIZE = 500

csv_path = Path(__file__).parent / 'convolearn-full.csv'
df = pd.read_csv(csv_path)

# Add conversation_id based on original row index
df['conversation_id'] = [f'convolearn_{i:05d}' for i in df.index]

# Shuffle and assign splits
random.seed(SEED)
indices = list(range(len(df)))
random.shuffle(indices)

efa_set = set(indices[:EFA_SIZE])
human_set = set(indices[EFA_SIZE:EFA_SIZE + HUMAN_SIZE])

df['split'] = df.index.map(lambda i: 'efa' if i in efa_set else 'human' if i in human_set else 'cfa')

# Write split files
for split_name in ['efa', 'human', 'cfa']:
    split_df = df[df['split'] == split_name].drop(columns=['split'])
    split_df.to_csv(csv_path.parent / f'convolearn_{split_name}.csv', index=False)
    print(f'convolearn_{split_name}.csv: {len(split_df)} conversations')

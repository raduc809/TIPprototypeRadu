import os
import pandas as pd
from collections import Counter
import re

# === Config ===
asm_folder = "train"
csv_path = "available_train_samples.csv"
output_csv = "asm_features.csv"

# Opcodes we want to track
opcodes = ['mov', 'push', 'call', 'jmp', 'cmp', 'add', 'sub', 'lea', 'xor', 'test']

# Load sample list
df = pd.read_csv(csv_path)
features = []

for _, row in df.iterrows():
    sample_id = row["Id"]
    label = row["Class"]
    asm_path = os.path.join(asm_folder, f"{sample_id}.asm")

    opcode_counts = Counter()
    section_names = set()

    if os.path.exists(asm_path):
        with open(asm_path, 'r', errors='ignore') as f:
            for line in f:
                line = line.lower()
                # Count opcodes
                for op in opcodes:
                    if f" {op} " in line:
                        opcode_counts[op] += 1
                # Capture section headers
                if line.startswith('.'):
                    section = line.split()[0]
                    section_names.add(section)

        # Prepare row
        feature_row = {op: opcode_counts.get(op, 0) for op in opcodes}
        for s in ['.text', '.data', '.rdata', '.idata']:
            feature_row[f'section_{s}'] = 1 if s in section_names else 0
        feature_row['Id'] = sample_id
        feature_row['Class'] = label
        features.append(feature_row)

# Save to CSV
feature_df = pd.DataFrame(features)
feature_df.to_csv(output_csv, index=False)
print(f"\n✅ Saved feature matrix to: {output_csv}")


import os
import pandas as pd
 

# === CONFIG ===
csv_path = "trainLabels.csv"
data_folder = "train"  # where your .asm/.bytes files are
print("Hello radu1")        # Folder that contains .bytes and .asm files

# === Load train.csv ===

# === Load train.csv ===
df = pd.read_csv(csv_path, encoding='utf-8-sig')
df.rename(columns=lambda x: x.strip().replace('ï»¿', ''), inplace=True)
print("Hello radu")
print(df.columns.tolist())
print(f"Loaded {len(df)} rows from train.csv")

# === Setup ===
available_samples = []
missing = []

# Get all files in the folder once, in lowercase
files_in_dir = set(f.lower() for f in os.listdir(data_folder))

# === Check for matching .bytes and .asm files ===
for idx, row in df.iterrows():
    sample_id = str(row["Id"]).strip()
    class_label = row["Class"]

    bytes_file = f"{sample_id.lower()}.bytes"
    asm_file = f"{sample_id.lower()}.asm"

    print(f"Checking: {sample_id}")
    print(f" → Looking for: {bytes_file}, {asm_file}")

    if bytes_file in files_in_dir and asm_file in files_in_dir:
        available_samples.append((sample_id, class_label))
    else:
        missing.append(sample_id)

   # if idx > 20:  # stop early just to test
    #    break

# === Save to CSV ===
df_available = pd.DataFrame(available_samples, columns=["Id", "Class"])
df_available.to_csv("available_train_samples.csv", index=False)

print(f"\n✅ Found {len(df_available)} usable samples with both .bytes and .asm files.")
print(f"⚠️ Skipped {len(missing)} samples missing files.")
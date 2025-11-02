# install kagglehub if missing
# !pip install kagglehub --quiet

import os
import kagglehub

print("[INFO] Downloading IAM Handwriting Word Dataset...")
dataset_path = kagglehub.dataset_download("nibinv23/iam-handwriting-word-database")

print(f"\n[INFO] Dataset downloaded to: {dataset_path}\n")
print("[INFO] Directory structure:\n")

for root, dirs, files in os.walk(dataset_path):
    level = root.replace(dataset_path, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for f in files[:5]:  # show only first few files
        print(f"{subindent}{f}")
    if level >= 2:  # avoid too-deep trees
        break

# save dataset path for later scripts
output_file = r"C:\Users\dilip\OneDrive\Desktop\handwriting_to_text\handwriting_to_text\dataset_path.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    f.write(dataset_path.strip())

print(f"\n[INFO] Dataset path saved to: {output_file}")

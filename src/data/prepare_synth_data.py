import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_dataset_from_folder(folder_path):
    dataset = []
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith(".png"):
            img_path = os.path.join(folder_path, file)
            txt_path = img_path.replace(".png", ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    label = f.read().strip()
                dataset.append((img_path, label))
    return dataset

data_folder = "/home/ubuntu/model/ml-project-2-syc-group/data/synth/data"
dataset = load_dataset_from_folder(data_folder)

# Sanity check
print(f"Found {len(dataset)} samples")
print("Example:", dataset[0])

# Split
train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)

# Save
np.save("/home/ubuntu/model/ml-project-2-syc-group/src/data/synth_trainset.npy", train_data)
np.save("/home/ubuntu/model/ml-project-2-syc-group/src/data/synth_valset.npy", val_data)

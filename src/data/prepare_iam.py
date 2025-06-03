import random
import numpy as np

from src.train.iamwordsdataset import IAMWordsDataset

dataset = IAMWordsDataset(
    words_txt_path="data/iam_dataset/ascii/words.txt",
    root_dir="data/iam_dataset/words",
    transform=None
)

all_samples = dataset.samples
random.seed(42)
random.shuffle(all_samples)

total = len(all_samples)
train_end = int(0.9 * total)
val_end = int(1 * total)

train_samples = all_samples[:train_end]
val_samples = all_samples[train_end:val_end]

np.save("data/trainset.npy", train_samples)
np.save("data/valset.npy", val_samples)


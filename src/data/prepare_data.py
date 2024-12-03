import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
import torch
from src.data.dataset import CustomChessScoreSheetsDataset

# Paths to datasets
chess_reader_data_path = "data/chess_reader_dataset"
online_data_path = "data/online_dataset"

# Function to rescale and crop images
def rescale_img(img, target_height, target_width):
    orig_height, orig_width = img.shape

    # Calculate scaling factors
    scale_height = target_height / orig_height
    scale_width = target_width / orig_width

    # Use the smaller scale to preserve aspect ratio
    scale = max(scale_height, scale_width)  # Ensure the image is big enough to crop

    # Compute new dimensions
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    # Resize image while preserving aspect ratio
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Center crop to the target size
    start_x = (new_width - target_width) // 2
    start_y = (new_height - target_height) // 2
    cropped_img = resized_img[start_y:start_y + target_height, start_x:start_x + target_width]

    return cropped_img

# Function to resize images
def resize_images(images, target_height, target_width):
    return [
        rescale_img(img, target_height, target_width)
        for img in tqdm(images, desc="Resizing images", unit="image")
    ]

# Function to load images and labels
def load_images_labels(path):
    images = []
    df = pd.read_csv(path + "/labels.txt", delimiter=',')
    labels_of = {
        str(img_id): label for img_id, label in df.values
    }
    labels = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith('.png'):
                continue
            source = os.path.join(root, file)
            img = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img / 255.0)  # Normalize images to 0-1
                img_id = file.split('.')[0]
                labels.append(labels_of[img_id])

    return images, labels

# Shuffle function
def shuffle(a, b):
    assert len(a) == len(b)
    indices = np.random.permutation(len(a))
    return [a[i] for i in indices], [b[i] for i in indices]

if __name__ == "__main__":
    chess_reader_images, chess_reader_labels = load_images_labels(chess_reader_data_path)
    online_images, online_labels = load_images_labels(online_data_path)

    print("Loaded data")
    # Resize images
    max_width = max([img.shape[1] for img in chess_reader_images + online_images])
    max_height = max([img.shape[0] for img in chess_reader_images + online_images])

    print("Max width:", max_width)
    print("Max height:", max_height)

    chess_reader_images = resize_images(chess_reader_images, max_height, max_width)
    online_images = resize_images(online_images, max_height, max_width)

    print("Chess reader dataset")
    print("#images:", len(chess_reader_images))
    print("#labels:", len(chess_reader_labels))

    for i in range(5):
        plt.imshow(chess_reader_images[i], cmap='gray')
        plt.title(chess_reader_labels[i])
        plt.show()

    print("Online dataset")
    print("#images:", len(online_images))
    print("#labels:", len(online_labels))

    for i in range(5):
        plt.imshow(online_images[i], cmap='gray')
        plt.title(online_labels[i])
        plt.show()

    parser = argparse.ArgumentParser()
    parser.add_argument("--split_ratio_chess_reader", type=float, default=0.5)
    parser.add_argument("--split_ratio", type=float, default=0.8)

    split_ratio_chess_reader = parser.parse_args().split_ratio_chess_reader
    print(f"Using {split_ratio_chess_reader * 100}% of the data in chess_reader dataset for training and the rest for validation + testing")

    split_ratio = parser.parse_args().split_ratio
    print(f"Using {split_ratio * 100}% of the data for training and the rest for validation")

    chess_reader_images, chess_reader_labels = shuffle(chess_reader_images, chess_reader_labels)
    split = int(len(chess_reader_images) * split_ratio_chess_reader)
    test_images = chess_reader_images[split:]
    test_labels = chess_reader_labels[split:]
    test_dataset = CustomChessScoreSheetsDataset(test_images, test_labels)

    torch.save(test_dataset, "data/test_dataset.pt")
    print("Created test dataset")

    images = chess_reader_images[:split] + online_images
    labels = chess_reader_labels[:split] + online_labels
    images, labels = shuffle(images, labels)

    split = int(len(images) * split_ratio)
    train_images = images[:split]
    train_labels = labels[:split]
    train_dataset = CustomChessScoreSheetsDataset(train_images, train_labels)
    print("Created train dataset")

    val_images = images[split:]
    val_labels = labels[split:]
    val_dataset = CustomChessScoreSheetsDataset(val_images, val_labels)
    print("Created val dataset")

    torch.save(train_dataset, "data/train_dataset.pt")
    torch.save(val_dataset, "data/val_dataset.pt")

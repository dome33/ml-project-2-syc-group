import os   
import os
import pandas as pd
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse 
from tqdm import tqdm
import torch 
from torch.utils.data import Dataset 

chess_reader_data_path = "data/chess_reader_dataset" 
online_data_path = "data/online_dataset" 


def png_to_mat(path): 
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    
    if img is None:
        return None 
    img = img / 255.0 
    # print("max min", img.max(), img.min()) 
    
    return img 

def resize_img(img, target_height, target_width): 
    median = np.median(img)
    # Pad the image directly
    pad_top = (target_height - img.shape[0]) // 2
    pad_left = (target_width - img.shape[1]) // 2
    pad_bottom = target_height - img.shape[0] - pad_top
    pad_right = target_width - img.shape[1] - pad_left
    
    return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=median)

    
def resize_images(images, target_height, target_width): 
    return [ 
        resize_img(img, target_height, target_width) 
        for img in tqdm(images, desc="Resizing images", unit="image")
    ] 
    
def load_images_labels(path): 
    images = [] 
    df = pd.read_csv(path + "/labels.txt", delimiter=',' ) 
    
    labels_of = {
        str(img_id) : label for img_id, label in df.values
    }
    labels = []
    
    for root, dirs, files in os.walk(path): 
        
        for file in files:
            if not file.endswith('.png'):
                continue 
            source = os.path.join(root, file) 
            # print(source) 
            img = png_to_mat(source) 
            
            images.append(img) 
            
            img_id = file.split('.')[0]
            labels.append(labels_of[img_id]) 

    return images, labels 


class CustomChessScoreSheetsDataset(Dataset):
    def __init__(self, images, labels): 
        self.images = [torch.tensor(img, dtype=torch.float32) for img in images]
        self.labels = labels
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
    
def shuffle(a,b):
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
    
    print("max width", max_width)
    print("max height", max_height)
    
    chess_reader_images = resize_images(chess_reader_images, max_height, max_width)
    online_images = resize_images(online_images, max_height, max_width)
    
    
    
    print("Chess reader dataset")
    print("#images:", len(chess_reader_images))
    print("#labels", len(chess_reader_labels))
    
    for i in range(5): 
        plt.imshow(chess_reader_images[i]) 
        plt.title(chess_reader_labels[i])
        plt.show()
    
    print("Online dataset")
    print("#images:", len(online_images))
    print("#labels", len(online_labels))
    
    for i in range(5):
        plt.imshow(online_images[i]) 
        plt.title(online_labels[i])
        plt.show()
    
    
    parser = argparse.ArgumentParser() 
    
    parser.add_argument("--split_ratio_chess_reader", type=float, default=0.5) 
    parser.add_argument("--split_ratio", type=float, default=0.8) 
    
    # how much of the data in chess reader dataset to use for <training+val> vs <testing>
    split_ratio_chess_reader = parser.parse_args().split_ratio_chess_reader 
    print(f"Using {split_ratio_chess_reader * 100}% of the data in chess_reader dataset for training and the rest for validation + testing") 
    
    # how much of the rest of the data to use for training vs validation 
    split_ratio = parser.parse_args().split_ratio 
    print(f"Using {split_ratio * 100}% of the data for training and the rest for validation") 
    
    # Construct test set 
    chess_reader_images, chess_reader_labels = shuffle(chess_reader_images, chess_reader_labels)     
    split = int(len(chess_reader_images) * split_ratio_chess_reader)
    test_images = chess_reader_images[split:]
    test_labels = chess_reader_labels[split:] 
    test_dataset = CustomChessScoreSheetsDataset(test_images, test_labels) 
    
    torch.save(test_dataset, "data/test_dataset.pt") 
    print("created test dataset") 
    
    # rest of the data is for validation and testing 
    images = chess_reader_images[:split] + online_images
    labels = chess_reader_labels[:split] + online_labels
    images, labels = shuffle(images, labels) 
    
    # assert len(images) == len(labels) 
    # assert len(images) == len(chess_reader_images) + len(online_images) 
    # assert len(labels) == len(chess_reader_labels) + len(online_labels)
    
    
    # split train and val 
    split = int(len(images) * split_ratio)
    # train set
    train_images = images[:split]
    train_labels = labels[:split]
    train_dataset = CustomChessScoreSheetsDataset(train_images, train_labels) 
    print("created train dataset") 
    # val set 
    val_images = images[split:] 
    val_labels = labels[split:] 
    val_dataset = CustomChessScoreSheetsDataset(val_images, val_labels)  
    print("created val dataset") 
    # save alles 
    torch.save(train_dataset, "data/train_dataset.pt") 
    torch.save(val_dataset, "data/val_dataset.pt") 
    
    





import torch 
from torch.utils.data import Dataset 

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
    

def custom_collate_fn(batch:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    lengths = [len(label) for label in labels]
    flattened_labels = [c for label in labels for c in label]
    return images, torch.tensor(flattened_labels), torch.tensor(lengths)


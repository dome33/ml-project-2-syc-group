from src.models.cnn_bilstm import CNNBILSTM
import yaml 
import argparse
import torch 
from types import SimpleNamespace
from src.data.dataset import custom_collate_fn 
from torch.utils.data import DataLoader 
from src.utils import Tokenizer, compute_cer 
from jiwer import cer 
from tqdm import tqdm 

# PARSE ARGS 
parser = argparse.ArgumentParser() 
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()


# LOAD CONFIG 
with open(args.config, "r") as file: 
    conf = yaml.load(file, Loader=yaml.FullLoader) 
    conf = SimpleNamespace(**conf) 
    
# LOAD DATA 
train_set = torch.load("data/train_dataset.pt") 
tokenizer = Tokenizer(train_set.labels) 
train_set.labels = [tokenizer.encode(label) for label in train_set.labels] # TODO do better 
train_loader = DataLoader(train_set, batch_size=conf.batch_size, shuffle=True, collate_fn=custom_collate_fn)


# validation 
val_set = torch.load("data/val_dataset.pt") 


# INITIALIZE STUFF 
model = CNNBILSTM(
    num_chars= tokenizer.n_chars(),
    activation="leaky_relu", 
    dropout=0.3
).to(conf.device) 


optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr) 
criterion = torch.nn.CTCLoss(blank=tokenizer.blank)  
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=10, mode="min", min_lr=1e-6)

print(f"trainset shape: {train_set.images.shape}") 
print(f"valset shape: {val_set.images.shape}")
for i in range(conf.max_epochs): 
    
    mean_loss = 0 
    
    for image, label, target_lengths in tqdm(
        train_loader, 
        desc=f"Epoch {i}", 
        total=len(train_loader)
        ): 
        # move to device
        image = image.to(conf.device)   
        label = label.to(conf.device)
        target_lengths = target_lengths.to(conf.device).long()
        image = image.unsqueeze(1) # (b, 1, h, w) 

        # print(image.shape) 
    
        # start training 
        optimizer.zero_grad()
        log_probs = model(image)
        log_probs = log_probs.permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
        
        
        input_lengths = torch.full((image.shape[0],), log_probs.shape[0], dtype=torch.long)    
    
        loss = criterion(log_probs, label, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        
        mean_loss += loss.item()
    print(f"Epoch {i} loss: {mean_loss/len(train_loader)}") 
        
    if i % 10 == 0: 
        print(f"Validation CER: {compute_cer(model, val_set, tokenizer, conf.device)}") 
        
        
        
    
    
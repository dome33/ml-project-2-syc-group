import pandas as pd 
import random 
import numpy as np 
DEFAULT_HEIGHT = 90 
DEFAULT_WIDTH = 400 

CHESS_READER_SPLIT = 0.5 




# Paths to datasets
chess_reader_data_path = "data/chess_reader_dataset"
online_data_path = "data/online_dataset"


# Function to load images and labels
def load_images_labels(path):
    
    df = pd.read_csv(path + "/labels.txt", delimiter=',')
    
    return [
        (path + "/" + name , label) 
        for name, label in zip(
            list(df['id']), list(df['prediction'])
        ) 
    ]
    
     
if __name__ == "__main__":
    chess_reader_dataset = load_images_labels(chess_reader_data_path) 
    online_dataset = load_images_labels(online_data_path)
    
    random.shuffle(chess_reader_dataset) 
    random.shuffle(online_dataset)
    
    split = int(len(chess_reader_dataset) * CHESS_READER_SPLIT) 
    np.save("data/testset.npy", chess_reader_dataset[:split]) 
    
    dataset = chess_reader_dataset[split:] + online_dataset 
    np.save("data/trainset.npy", dataset) 
    
    
    

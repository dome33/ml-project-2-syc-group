
import numpy as np 
from jiwer import cer 
from torch.utils.data import DataLoader

class Tokenizer(): 
    
    def __init__(self, labels:list[str]):
        
        self.chars = list(set([c for label in labels for c in label])) 
        self.chars.sort() 
        
        self.blank = len(self.chars) 
        
        self.chars_to_idx = {c:i for i, c in enumerate(self.chars)} 
        self.idx_to_chars = {i:c for i, c in enumerate(self.chars)} 
        
    def encode(self, label:str) -> list[int]: 
        """
        Encode a label into a list of integers corresponding to the index of the character in the vocabulary. 
        """
        return [self.chars_to_idx[c] for c in label] 
    
    def decode(self, encoded:list[int]) -> str: 
        """
        Decode a list of integers into a string. 
        """
        return "".join([self.idx_to_chars[i] for i in encoded]) 


    def n_chars(self) -> int: 
        return len(self.chars) + 1 
    

def logprobs_to_seq_tokens(log_probs, blank_index):
    """
    Decode a sequence of tokens from the model output. 
    :param log_probs: Log probabilities from the model (T x V)
    :param blank_index: Index of the blank token
    :param tokenizer: Tokenizer object 
    
    :return: Decoded sequence as a string
    """
    max_indices = np.argmax(log_probs, axis=1)  # Argmax over vocabulary
    
    return [
        idx  
        for i, idx in enumerate(max_indices) 
        if idx != blank_index and (i == 0 or idx != max_indices[i-1])
    ]
    
    
def compute_cer(model,dataset,tokenizer,device): 
    model.eval()
    model = model.to("cpu")
    avg_cer = 0 
    
    for i,(images, labels ) in enumerate(dataset):  
        print(i) 
        images = images.unsqueeze(0).unsqueeze(0)
        log_probs = model(images).detach().cpu().numpy() 
    
        decoded = logprobs_to_seq_tokens(log_probs[0], tokenizer.blank) 
        decoded = tokenizer.decode(decoded)
        
        cer_score = cer(labels, decoded)
        avg_cer += cer_score 
    
    model = model.to(device) 
    model.train()
    return avg_cer / len(dataset) 
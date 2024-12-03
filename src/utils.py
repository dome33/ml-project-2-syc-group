

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
    
    
    
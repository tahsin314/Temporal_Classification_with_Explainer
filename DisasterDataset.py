import torch
from torch.utils.data import Dataset

class DisasterDataset(Dataset):
    def __init__(self, tokenizer, data, label=None, max_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.label = label
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.data[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        if self.label:
            return input_ids, attention_mask, torch.tensor(self.label[idx])
        else:
            return input_ids, attention_mask
        


from torch.utils.data import Dataset

class AccelerometerDataset(Dataset):
    def __init__(self, signal, label) -> None:
        super().__init__()
        self.signal = signal
        self.label = label
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        data = self.signal[index].transpose(1,0)
        label = self.label[index]
        return data, int(float(label))
    

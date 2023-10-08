from torch.utils.data import Dataset
from utils import onehot_encoder

class AccelerometerDataset(Dataset):
    def __init__(self, signal, label, downsampling_factor=1, one_hot=False) -> None:
        super().__init__()
        self.signals = signal
        self.labels = label
        self.downsampling_factor = downsampling_factor
        self.one_hot = one_hot

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        data = self.signals[index].transpose(1,0)[:, ::self.downsampling_factor]
        label = int(float(self.labels[index]))
        if self.one_hot:
            label = onehot_encoder(3, label)
        return data, label
    
    def get_labels(self):
        return list(self.labels)

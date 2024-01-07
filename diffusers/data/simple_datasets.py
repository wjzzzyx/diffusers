from torch.utils.data import Dataset


class NullImageDataset(Dataset):
    def __init__(self, len: int):
        self.len = len
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return {'index': index}
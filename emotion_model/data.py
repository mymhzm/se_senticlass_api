# data.py
# 定义 dataset 所需要的 '__init__', '__getitem__', '__len__'
# 好讓 dataloader 能使用
from torch.utils import data


class ReviewDataset(data.Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)

    __len__ will return the number of data
    """

    def __init__(self, X, y):
        self.data = X
        self.label = y

    # format the Dataset output item to fit the DataLoader
    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
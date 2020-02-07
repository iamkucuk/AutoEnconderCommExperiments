import torch
from torch.utils.data import Dataset


class DataGen(Dataset):
    def __init__(self, in_channels, epoch_len):
        self.epoch_len = epoch_len
        self.in_channels = in_channels
        self.labels = (torch.rand(self.epoch_len) * self.in_channels).long()
        self.data = torch.sparse.torch.eye(self.in_channels).index_select(dim=0, index=self.labels)

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        data = self.data[item]
        label = self.labels[item]

        return data, label


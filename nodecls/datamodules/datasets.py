import numpy as np
from torch.utils.data import Dataset


class NumPyDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split_type: str,
        apply_masking: bool=False,
    ):
        assert split_type in ["train", "val", "test"]
        data = np.load(data_path, allow_pickle=True)
        self.masks = data[f"{split_type}_masks"]
        self.feats = data["features"].astype(np.float32)
        self.labels = data["labels"].astype(np.int64)
        if apply_masking:
            self.feats = self.feats[self.masks]
            self.labels = self.labels[self.masks]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = {}
        
        data["index"] = index
        data["masks"] = self.masks[index]
        data["x"] = self.feats[index]
        data["y"] = self.labels[index]
        
        return data

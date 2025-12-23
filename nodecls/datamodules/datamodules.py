import lightning as L
import numpy as np
import torch

from .datasets import NumPyDataset
from torch.utils.data import DataLoader

from torch_geometric.utils import (
    subgraph,
    to_undirected,
    add_self_loops,
    remove_self_loops
)


class NumPyDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path,
        train_batch_size=4096,
        test_batch_size=4096 * 2,
        num_workers=4,
    ):
        super().__init__()
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.train_dataset = NumPyDataset(
            data_path,
            split_type="train",
            apply_masking=True,
        )
        self.val_dataset = NumPyDataset(
            data_path,
            split_type="val",
            apply_masking=True,
        )
        self.test_dataset = NumPyDataset(
            data_path,
            split_type="test",
            apply_masking=True,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
        )


class GraphDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path,
        train_batch_size=4096,
        test_batch_size=4096 * 2,
        num_workers=4,
    ):
        super().__init__()
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.train_dataset = NumPyDataset(
            data_path,
            split_type="train",
        )
        self.val_dataset = NumPyDataset(
            data_path,
            split_type="val",
        )
        self.test_dataset = NumPyDataset(
            data_path,
            split_type="test",
        )

        data = np.load(data_path, allow_pickle=True)
        self.num_nodes = torch.as_tensor(
            data["num_nodes"], dtype=torch.long)
        self.edge_indices = torch.as_tensor(
            data["edge_indices"], dtype=torch.long)
        if data["undirected"]:
            self.edge_indices = to_undirected(self.edge_indices)
        if data["add_self_loops"]:
            self.edge_indices, _ = remove_self_loops(self.edge_indices)
            self.edge_indices, _ = add_self_loops(
                self.edge_indices,
                num_nodes=self.num_nodes,
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True,
            pin_memory=True,
        )
    
    def on_before_batch_transfer(self, batch, dataloader_idx):
        edge_index, _ = subgraph(
            batch["index"],
            self.edge_indices,
            num_nodes=self.num_nodes,
            relabel_nodes=True,
        )
        batch["edge_index"] = edge_index
        return batch

import numpy as np
import os.path as osp
import pandas as pd
import scipy

from ogb.nodeproppred import NodePropPredDataset
from sklearn.preprocessing import StandardScaler


def convert_pokec(
    path_to_mat_file: str,
    path_to_splits: str,
    save_path: str,
    normalize_feats: bool=False,
):
    """Download the `pokec` dataset and `pokec-splits.npy` to the following directory:
        * https://drive.google.com/drive/folders/1rr3kewCBUvIuVxA6MJ90wzQuF-NnCRtf
          -> data/pokec/
        * https://github.com/LUOyk1999/tunedGNN/blob/main/large_graph/data/pokec/pokec-splits.npy
          -> data/pokec/
    
    Usage:
        from data.utils import convert_pokec
        
        convert_pokec(
            path_to_mat_file="data/pokec/pokec.mat",
            path_to_splits="data/pokec/pokec-splits.npy",
            save_path="data/pokec/pokec.npz",
            normalize_feats=False
        )
    """
    data = scipy.io.loadmat(path_to_mat_file)
    
    # Use the first split in the provided `pokec-splits.npy` file.
    splits = np.load(path_to_splits, allow_pickle=True)
    train_indices = splits[0]["train"]
    val_indices = splits[0]["valid"]
    test_indices = splits[0]["test"]
    
    num_nodes = data["num_nodes"].item()
    train_masks = np.zeros(num_nodes, dtype=bool)
    train_masks[train_indices] = True
    val_masks = np.zeros(num_nodes, dtype=bool)
    val_masks[val_indices] = True
    test_masks = np.zeros(num_nodes, dtype=bool)
    test_masks[test_indices] = True
    
    feats = data["node_feat"].astype(np.float32)
    if normalize_feats:
        # Log-normalize large feature counts
        # followed by standard scaling.        
        for col in [1, 13]:
            feats[:, col] = np.log(feats[:, col] + 1.0)
        feats = StandardScaler().fit_transform(feats)
    
    print(f"Saving to {save_path}...")
    np.savez_compressed(
        save_path,
        features=feats,
        labels=data["label"].reshape((-1,)).astype(np.int64),
        edge_indices=data["edge_index"].astype(np.int64),
        train_masks=train_masks,
        test_masks=test_masks,
        val_masks=val_masks,
        num_nodes=num_nodes,
        undirected=True,
        add_self_loops=True,
    )


def convert_ogbn_datasets(
    root_dir: str,
    data_name: str,
    save_path: str,
):
    """Download and unzip the following files into their
    respective directories:
        * http://snap.stanford.edu/ogb/data/nodeproppred/products.zip
          -> data/ogbn/products/
        * http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip
          -> data/ogbn/arxiv/
    Also download the `master.csv` file into the root directory `data/ogbn/`:
        * https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/master.csv
          -> data/ogbn/
    
    Usage:
        from data.utils import convert_ogbn_datasets
        
        convert_ogbn_datasets(
            root_dir="data/ogbn/",
            data_name="ogbn-products",
            save_path="data/ogbn/products/ogbn-products.npz"
        )
    """
    supported = [
        "ogbn-arxiv",
        "ogbn-products",
    ]
    assert data_name in supported, \
    f"`{data_name}` is not in the list of {supported} datasets."
    
    dir_name = data_name.split("-")[-1]
    dir_path = osp.join(root_dir, dir_name)
    master = pd.read_csv(osp.join(root_dir, "master.csv"), index_col=0, keep_default_na=False)
    meta_info = master[data_name]
    meta_info["dir_path"] = dir_path

    dataset = NodePropPredDataset(name=data_name, root=dir_path, meta_dict=meta_info)
    split_idx = dataset.get_idx_split()
    graph, labels = dataset[0]
    
    train_indices = split_idx["train"]
    val_indices = split_idx["valid"]
    test_indices = split_idx["test"]
    
    num_nodes = graph["num_nodes"]
    train_masks = np.zeros(num_nodes, dtype=bool)
    train_masks[train_indices] = True
    val_masks = np.zeros(num_nodes, dtype=bool)
    val_masks[val_indices] = True
    test_masks = np.zeros(num_nodes, dtype=bool)
    test_masks[test_indices] = True
    
    print(f"Saving to {save_path}...")
    np.savez_compressed(
        save_path,
        features=graph["node_feat"].astype(np.float32),
        labels=labels.reshape((-1,)).astype(np.int64),
        edge_indices=graph["edge_index"].astype(np.int64),
        train_masks=train_masks,
        test_masks=test_masks,
        val_masks=val_masks,
        num_nodes=num_nodes,
        undirected=meta_info["add_inverse_edge"],
        add_self_loops=True,
    )

# Supported Datasets
Refer to `utils.py` for more information on how to download and convert the source datasets.

| dataset | pokec | ogbn-arxiv | ogbn-products |
| :--- | ---: | ---: | ---: |
| num nodes | 1,632,803 | 169,343 |  2,449,029 |
| num edges | 30,622,564 | 1,166,243 |  61,859,140 |
| num classes | 2 | 40 | 47 |
| eval metric | accuracy | accuracy | accuracy |
| inverse edge | True | False | True |

## Directory Structure
The dataset artifacts reside in this directory with the following structure:

```
data/
|--pokec/
|  |--label.npy
|  |--node_feat.npy
|  |--pokec-splits.npy
|  |--pokec.mat
|  |--split_0.5_0.25/
|--ogbn/
|  |--master.csv
|  |--arxiv/
|  |--products/
```

## Usage
Convert the source datasets to the corresponding `.npz` formats required by this project.

```python
from data.utils import convert_pokec, convert_ogbn_datasets

convert_pokec(
    path_to_mat_file="data/pokec/pokec.mat",
    path_to_splits="data/pokec/pokec-splits.npy",
    save_path="data/pokec/pokec.npz",
    normalize_feats=False
)

# `data_name` is one of ["ogbn-arxiv", "ogbn-products"].
convert_ogbn_datasets(
    root_dir="data/ogbn/",
    data_name="ogbn-products",
    save_path="data/ogbn/products/ogbn-products.npz"
)
```
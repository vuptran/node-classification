from collections import defaultdict
import math
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
        self.feats = data["features"]
        self.labels = data["labels"]
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


class ClassBalancedDataset:
    """A wrapper of the dataset with repeat factor.

    Suitable for training on class imbalanced datasets. In each epoch, a
    sample may appear multiple times based on its "repeat factor".
    The repeat factor for a sample is a function of the frequency of the rare
    category. The "frequency of category c" in [0, 1] is defined by the
    fraction of samples in the training set (without repeats)
    in which category c appears.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction # of samples
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(thr/f(c)))`
    3. For each sample, assign the sample-level repeat factor

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which the sample is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling follows the square-root inverse frequency
            heuristic max(1, sqrt(thr/f(c)))
    """

    def __init__(self, dataset, oversample_thr):
        self.dataset = dataset

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

    def _get_repeat_factors(self, dataset, threshold):
        """Get repeat factor for each sample in the dataset.

        Args:
            dataset (:obj:`Dataset`): The dataset
            threshold (float): The threshold of frequency. If a sample
                contains the category with frequency below the threshold,
                it would be repeated.

        Returns:
            list[float]: The repeat factors for each sample in the dataset.
        """

        # 1. For each category c, compute the fraction # of samples
        #    that contain it: f(c)
        category_freq = defaultdict(int)
        num_samples = len(dataset)
        for idx in range(num_samples):
            cat_id = dataset[idx]["y"].squeeze(-1).item()
            category_freq[cat_id] += (1 / num_samples)

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(thr/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(threshold / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each sample, assign the sample-level repeat factor
        repeat_factors = []
        for idx in range(num_samples):
            cat_id = dataset[idx]["y"].squeeze(-1).item()
            repeat_factor = category_repeat[cat_id]
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        rep_index = self.repeat_indices[idx]
        return self.dataset[rep_index]

    def __len__(self):
        """Length after repetition."""
        return len(self.repeat_indices)

import pickle

import torch
import yaml
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, BatchSampler

import scanpy as sc


def replace_zeros_with_nan(series):
    if (series != 0).any():
        series = series.replace(0, np.nan)
    return series


def process_func(file_path: str, label_path: str, aug_rate=1, missing_ratio=0.1):
    if label_path is None:
        adata = sc.read_csv(file_path)
        adata.X = adata.X.astype('float64')
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata)
        cluster_info = pd.DataFrame(adata.obs['leiden'])
        data = pd.read_csv(file_path, index_col=0)
        data.insert(loc=len(data.columns),
                    column='cluster',
                    value=cluster_info.values.flatten()
                    )
        data = data.groupby('cluster', group_keys=False).apply(
            lambda x: x.apply(replace_zeros_with_nan))
        data = data.drop(data.columns[-1], axis=1)

    else:
        data = pd.read_csv(file_path, index_col=0)
        label = pd.read_csv(label_path, index_col=0)
        data.insert(loc=len(data.columns),
                    column='cluster',
                    value=label.values.flatten()
                    )
        data = data.groupby('cluster', group_keys=False).apply(
            lambda x: x.apply(replace_zeros_with_nan))

        data = data.drop(data.columns[-1], axis=1)

    data_aug = pd.concat([data] * aug_rate)

    observed_values = data_aug.values.astype("float32")
    observed_masks = ~np.isnan(observed_values)

    true_masks = np.isnan(observed_values)
    true_masks = np.where(true_masks, 1, 0)

    masks = observed_masks.copy()

    for col in range(observed_values.shape[1]):
        obs_indices = np.where(masks[:, col])[0]
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices, col] = False

    gt_masks = masks.reshape(observed_masks.shape)
    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype(int)
    gt_masks = gt_masks.astype(int)

    return observed_values, observed_masks, gt_masks, true_masks


class tabular_dataset(Dataset):

    def __init__(self, eval_length, use_index_list=None, aug_rate=1, missing_ratio=0.1, seed=0, file_path=None,
                 label_path=None):
        self.label_path = label_path
        self.file_path = file_path
        self.eval_length = eval_length

        np.random.seed(seed)

        processed_data_path = (
            f"./missing_ratio-{missing_ratio}_seed-{seed}.pk"
        )
        processed_data_path_norm = (
            f"./missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
        )

        if not os.path.isfile(processed_data_path):
            self.observed_values, \
                self.observed_masks, \
                self.gt_masks, \
                self.true_masks = process_func(
                file_path=file_path,
                label_path=label_path,
                aug_rate=aug_rate,
                missing_ratio=missing_ratio
            )

            with open(processed_data_path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks, self.true_masks], f
                )
            print("--------Dataset created--------")

        elif os.path.isfile(processed_data_path_norm):
            with open(processed_data_path_norm, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks, self.true_masks = pickle.load(
                    f
                )
            print("--------Normalized dataset loaded--------")

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
            "true_masks": self.true_masks[index]
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed, nfold, batch_size, missing_ratio, file_path, label_path, eva_length):
    dataset = tabular_dataset(missing_ratio=missing_ratio, seed=seed, file_path=file_path, label_path=label_path,
                              eval_length=eva_length)

    indlist = np.arange(len(dataset))
    np.random.seed(seed + 1)
    np.random.shuffle(indlist)

    tmp_ratio = 1 / nfold
    start = (int)((nfold - 1) * len(dataset) * tmp_ratio)
    end = (int)(nfold * len(dataset) * tmp_ratio)

    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))
    np.random.shuffle(remain_index)
    num_train = (int)(len(remain_index) * 1)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    max_arr = []

    processed_data_path_norm = (
        f"./missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
    )

    if not os.path.isfile(processed_data_path_norm):
        col_num = dataset.observed_values.shape[1]
        max_arr = np.zeros(col_num)
        min_arr = np.zeros(col_num)

        for k in range(col_num):
            obs_ind = dataset.observed_masks[train_index, k].astype(bool)
            temp = dataset.observed_values[train_index, k]
            max_arr[k] = max(temp[obs_ind])
            min_arr[k] = min(temp[obs_ind])

        dataset.observed_values = (
                                          (dataset.observed_values - 0 + 1) / (max_arr - 0 + 1)
                                  ) * dataset.observed_masks

        with open(processed_data_path_norm, "wb") as f:
            pickle.dump(
                [dataset.observed_values, dataset.observed_masks, dataset.gt_masks, dataset.true_masks], f
            )

    # Create datasets and corresponding data loaders objects.
    train_dataset = tabular_dataset(use_index_list=train_index,
                                    missing_ratio=missing_ratio,
                                    seed=seed,
                                    file_path=file_path,
                                    label_path=label_path,
                                    eval_length=eva_length)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)

    valid_dataset = tabular_dataset(use_index_list=valid_index,
                                    missing_ratio=missing_ratio,
                                    seed=seed,
                                    file_path=file_path,
                                    label_path=label_path,
                                    eval_length=eva_length)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False)

    test_dataset = tabular_dataset(use_index_list=test_index,
                                   missing_ratio=missing_ratio,
                                   seed=seed,
                                   file_path=file_path,
                                   label_path=label_path,
                                   eval_length=eva_length)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    genera_loader = DataLoader(dataset,
                               batch_size=batch_size,
                               shuffle=False)

    return train_loader, valid_loader, test_loader, genera_loader, dataset, max_arr

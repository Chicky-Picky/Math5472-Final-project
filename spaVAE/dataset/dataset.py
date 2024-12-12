import torch
import numpy as np
import h5py
import os

from sklearn.preprocessing import MinMaxScaler
import scanpy as sc

from typing import Union

from torch.utils.data import Dataset

AnyPath = Union[str, bytes, os.PathLike]

# Normalize procedure is copied from spaVAE project
def normalize(adata,
              filter_min_counts=True,
              size_factors=True,
              normalize_input=True,
              logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

class SpatialCountDataset(Dataset):

    def __init__(self,
                 raw_data_filename: AnyPath,
                 root: AnyPath = ".",
                 loc_range: float = 20.,
                 ):
        """
        :param root: where dataset should be stored. This folder contains
                     raw_filename (original dataset) and processed_dir (processed data)
        :param raw_filename: raw info filename
        """
        self.root = root
        self.raw_filename = os.path.join(self.root, raw_data_filename)
        self.processed_dir = os.path.join(self.root, "processed")

        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
            
            data_mat = h5py.File(self.raw_filename, 'r')
            Y = np.array(data_mat['X']).astype('float64') # count matrix
            X = np.array(data_mat['pos']).astype('float64') # location information
            data_mat.close()

            scaler = MinMaxScaler()
            X = scaler.fit_transform(X) * loc_range

            adata = sc.AnnData(Y, dtype="float64")
            adata = normalize(adata,
                              size_factors=True,
                              normalize_input=True,
                              logtrans_input=True)

            for i, (x, y, y_raw, size_factors) in enumerate(zip(X, adata.X, adata.raw.X, adata.obs.size_factors)):
                torch.save({
                    'loc': torch.tensor(x),
                    'norm_count': torch.tensor(y),
                    'raw_count': torch.tensor(y_raw),
                    'size_factors': torch.tensor(size_factors)},
                    os.path.join(self.processed_dir, f"{i}.pt"))

    def __getitem__(self, index):
        r"""Gets the data object at index :obj:`idx`."""
        file_pt = os.path.join(self.processed_dir, f"{index}.pt")
        data = torch.load(os.path.join(file_pt), weights_only=True)
        
        x = data['loc']
        y = data['norm_count']
        y_raw = data['raw_count']
        size_factors = data['size_factors']

        return x, y, y_raw, size_factors

    def __len__(self) -> int:
        return len(os.listdir(self.processed_dir))
    

class SpatialCountNormalizedDataset(Dataset):

    def __init__(self,
                 raw_data_filename: AnyPath,
                 root: AnyPath = ".",
                 ):
        """
        :param root: where dataset should be stored. This folder contains
                     raw_filename (original dataset) and processed_dir (processed data)
        :param raw_filename: raw info filename
        """
        self.root = root
        self.raw_filename = os.path.join(self.root, raw_data_filename)

        name = raw_data_filename.split('.')[0]
        self.processed_dir = os.path.join(self.root, "processed_" + name)

        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
            
            data_mat = h5py.File(self.raw_filename, 'r')
            Y = np.array(data_mat['X']).astype('float64') # normalized count matrix
            raw_Y = np.array(data_mat['X_raw']).astype('float64') # raw count matrix
            X = np.array(data_mat['pos']).astype('float64') # scaled location information
            SF = np.array(data_mat['sf']).astype('float64') # size factors
            data_mat.close()

            for i, (x, y, y_raw, size_factors) in enumerate(zip(X, Y, raw_Y, SF)):
                torch.save({
                    'loc': torch.tensor(x),
                    'norm_count': torch.tensor(y),
                    'raw_count': torch.tensor(y_raw),
                    'size_factors': torch.tensor(size_factors)},
                    os.path.join(self.processed_dir, name + f"_{i}.pt"))

    def __getitem__(self, index):
        r"""Gets the data object at index :obj:`idx`."""
        name = os.path.basename(self.raw_filename).split(".")[0]
        file_pt = os.path.join(self.processed_dir, name + f"_{index}.pt")
        data = torch.load(os.path.join(file_pt), weights_only=True)
        
        x = data['loc']
        y = data['norm_count']
        y_raw = data['raw_count']
        size_factors = data['size_factors']

        return x, y, y_raw, size_factors

    def __len__(self) -> int:
        return len(os.listdir(self.processed_dir))
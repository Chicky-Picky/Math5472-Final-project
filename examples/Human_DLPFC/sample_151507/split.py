import h5py
import numpy as np
import scanpy as sc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

data_mat = h5py.File("/Users/akazovskaia/Documents/MATH5472/Final/examples/Human_DLPFC/sample_151507/sample_151507.h5", 'r')
Y = np.array(data_mat['X']).astype('float64')
X = np.array(data_mat['pos']).astype('float64')
L = np.array(data_mat['Y']).astype('S26')
data_mat.close()

adata = sc.AnnData(Y)
adata = normalize(adata,
                  size_factors=True,
                  normalize_input=True,
                  logtrans_input=True)

scaler = MinMaxScaler()
X = scaler.fit_transform(X) * 20

(count_train, count_test, raw_count_train, raw_count_test, pos_train_scaled, pos_test_scaled,
 size_factors_train, size_factors_test, labels_train, labels_test) = train_test_split(adata.X, adata.raw.X, X, adata.obs.size_factors, L, test_size=0.1, train_size=0.9)

with h5py.File('/Users/akazovskaia/Documents/MATH5472/Final/examples/Human_DLPFC/sample_151507/train.h5', 'w') as f:
    f.create_dataset('X', data=count_train)                                                                                                                                                                       
    f.create_dataset('X_raw', data=raw_count_train)
    f.create_dataset('pos', data=pos_train_scaled)
    f.create_dataset('sf', data=size_factors_train)
    f.create_dataset('Y', data=labels_train)

with h5py.File('/Users/akazovskaia/Documents/MATH5472/Final/examples/Human_DLPFC/sample_151507/test.h5', 'w') as f:
    f.create_dataset('X', data=count_test)
    f.create_dataset('X_raw', data=raw_count_test)
    f.create_dataset('pos', data=pos_test_scaled)
    f.create_dataset('sf', data=size_factors_test)
    f.create_dataset('Y', data=labels_test)
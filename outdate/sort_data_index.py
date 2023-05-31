import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


def sort_data_index(data):
    # Embed data into 1D first
    # pca = PCA(n_components=1)
    # for i in range(data.shape[0]):
    #     data[i, :] = torch.Tensor(pca.fit_transform(data[i, :, :]))

    dist = torch.cdist(data, data)
    for i in range(data.shape[0]):
        idx = torch.zeros([data.shape[1]], dtype=torch.int64)
        node = 0
        for j in range(idx.shape[0]):
            dist_sum = dist[i, node, :] + dist[i, 0, :]
            node = torch.argmin(dist_sum)
            idx[j] = node
            dist[i, :, node] = torch.inf
        data[i, :, :] = data[i, idx, :]
    return data


def pca_sort(data):
    pca = PCA(n_components=1)
    for i in range(data.shape[0]):
        data_pca = pca.fit_transform(data[i, :, :])
        dist = np.power(data_pca - data_pca[0, :], 2)
        idx = np.squeeze(np.argsort(dist, axis=0))
        data[i, :, :] = data[i, idx, :]
    return data


def mds_sort(data):
    mds = MDS(n_components=1)
    for i in range(data.shape[0]):
        data_mds = mds.fit_transform(data[i, :, :])
        dist = np.power(data_mds - data_mds[0, :], 2)
        idx = np.squeeze(np.argsort(dist, axis=0))
        data[i, :, :] = data[i, idx, :]
    return data


if __name__ == '__main__':
    data = torch.rand(size=[10, 5, 2])
    # sort_data_index(data)
    # sorted_data = pca_sort(data)
    sorted_data = mds_sort(data)
    print()
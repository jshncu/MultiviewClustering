from __future__ import absolute_import
import scipy.io as scio
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import torch
import torch.nn as nn

def load_data(args):
    if args.dataset == '3sources':
        data = scio.loadmat('./data/3sources.mat')
        label = data['truelabel'][0][0].squeeze()
        # sorted_indices = np.argsort(label)
        # sorted_label = label[sorted_indices]
        # label = sorted_label
        # sort_feature0 = data['data'][0][0][sorted_indices]
        # sort_feature1 = data['data'][0][1][sorted_indices]
        # sort_feature2 = data['data'][0][2][sorted_indices]
        # data['data'][0][0] = sort_feature0
        # data['data'][0][1] = sort_feature1
        # data['data'][0][2] = sort_feature2
        feature = data['data'][0]
        for i in range(feature.shape[0]):
            feature[i] = feature[i].transpose()

    elif args.dataset == 'BDGP':
        data = scio.loadmat('./data/BDGP.mat')
        label = data['Y'][0].squeeze()

        # sorted_indices = np.argsort(label)
        # sorted_label = label[sorted_indices]
        # label = sorted_label
        #
        # sort_feature0 = data['X1'][sorted_indices]
        # sort_feature1 = data['X2'][sorted_indices]
        # data['X1'] = sort_feature0
        # data['X2'] = sort_feature1
        data['X1'] = data['X1'].transpose()
        data['X2'] = data['X2'].transpose()
        feature = np.array([data['X1'], data['X2']], dtype=object)
        for i in range(feature.shape[0]):
            feature[i] = feature[i].transpose()

    elif args.dataset == 'Caltech-5V':
        "X1:(1400, 40), X2:(1400, 254) X3:(1400, 1984), X4:(1400, 512), X5:(1400, 928)"
        data = scio.loadmat('./data/Caltech-5V.mat')
        array1 = data['X1'].transpose()
        array2 = data['X2'].transpose()
        array3 = data['X3'].transpose()
        array4 = data['X4'].transpose()
        array5 = data['X5'].transpose()
        feature = np.array([array1, array2], dtype=object)  #Caltech-2V [40, 254]
        # feature = np.array([array1, array2, array5], dtype=object)   #Caltech-3V [40, 254,928]
        # feature = np.array([array1, array2, array5, array4], dtype=object)  #Caltech-4V [40, 254, 928, 512]
        #feature = np.array([array1, array2, array5,array4, array3], dtype=object)  #Caltech-5V [40, 254, 928, 512, 1984]
        for i in range(feature.shape[0]):
            feature[i] = feature[i].transpose()
        label = data['Y'].squeeze()


    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    Data = Construct_View(min_k=3, k=10, feature=feature, label=label, mask_T=args.mask_T, mask_rate=args.mask_rate)
    return Data

class Construct_View(object):
    def __init__(self, min_k, k, feature, label,mask_T, mask_rate):
        self.k = k
        self.min_k = min_k
        self.feature = feature
        self.label = label
        self.idx = np.arange(self.feature[0].shape[0])
        self.graph_dict = {}
        self.mi_dict = {}
        self.mask_T = mask_T
        self.p = mask_rate
        self.A_c = []

        err = []
        err_mi = []

        for i in range(self.feature.shape[0]):
            metric = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
            g_all = []
            e_all = []
            mi_idx_all = []
            error_mi_all = []
            for met in metric:
                g, e, min_idx, error_mi = KNN_graph(self.min_k, self.k, self.feature[i], self.label, metrix=met)
                # print("g.shape: ", g.shape)
                # print("g.type: ", type(g))
                g_all.append(g)
                e_all.append(e)
                mi_idx_all.append(min_idx)
                error_mi_all.append(error_mi)

            err.append(min(e_all))
            err_mi.append(min(error_mi_all))
            self.mi_dict[i] = mi_idx_all[error_mi_all.index(min(error_mi_all))]
            self.graph_dict[i] = g_all[e_all.index(min(e_all))]


        for i in range(len(err)):
            self._load(self.feature[i], self.label, self.idx, self.graph_dict[i], i)

    def _load(self, feature, label, idx, graph, i):
        features = sp.csr_matrix(feature, dtype=np.float32)
        labels = _encode_onehot(label)
        self.num_labels = labels.shape[1]

        # Constructing the graph
        idx = np.asarray(idx, dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = graph
        edges = np.asarray(list(map(idx_map.get, edges_unordered.flatten())),
                           dtype=np.int32).reshape(edges_unordered.shape)

        # Constructing a symmetric adjacency matrix
        adj = sp.coo_matrix((np.ones(edges.shape[0]),
                             (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        if self.mask_T:
            bernoulli_samples = np.random.binomial(1, self.p, size=adj.data.shape)
            new_data = adj.data * bernoulli_samples
            adj = coo_matrix((new_data, (adj.row, adj.col)),
                                        shape=adj.shape)

        # from sklearn.metrics.pairwise import cosine_similarity
        # adj_matrix_dense = adj.toarray()
        # similarity_matrix = cosine_similarity(adj_matrix_dense)
        # similarity_matrix_coo = coo_matrix(similarity_matrix)
        # result = similarity_matrix_coo.dot(similarity_matrix_coo)
        # similarity_matrix_coo = result.tocoo()
        # row = similarity_matrix_coo.row
        # col = similarity_matrix_coo.col
        # data = similarity_matrix_coo.data
        # np.savetxt('similarity_matrix_coo_BDGP30.txt', np.column_stack((row, col, data)), fmt='%d %d %.6f')

        #Create a graph from adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        self.graph_dict[i] = nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph())
        features = _normalize(features)
        self.feature[i] = np.asarray(features.todense())
        self.label = np.where(labels)[1]



def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.asarray(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def _encode_onehot(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.asarray(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def KNN_graph(min_k=3, knn=10, data='', label='', metrix='minkowski'):
    """
    Define the function KNN_graph, which takes four parameters: mik (default value 3),
    knn (default value 10), data (default empty string), label (default empty string), metrix (default value 'Minkowski')
    : param min_k:  The minimum number of neighbors used to construct a graph
    : param knn: The K value in the KNN algorithm, which is the number of nearest neighbors considered
    : param data:  Features of training dataset
    : param label:  Label the training dataset
    : param metrix:  A measurement method used to calculate distance
    : return:   Return the constructed graph, error rate, minimum neighbor index, and error rate of the minimum neighbor.
    """
    x_train, y_train = data, label
    n_train = len(x_train)
    x_train_flat = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))[:n_train]
    train_neighbors = NearestNeighbors(n_neighbors=knn + 1, metric=metrix).fit(x_train_flat)
    _, idx = train_neighbors.kneighbors(x_train_flat)

    new_idx = np.empty((idx.shape[0], idx.shape[1] - 1))
    assert (idx >= 0).all()

    for i in range(idx.shape[0]):
        try:
            new_idx[i] = idx[i, idx[i] != i][:idx.shape[1] - 1]
        except Exception as e:
            print(idx[i, ...], new_idx.shape, idx.shape)
            raise e

    idx = new_idx.astype(int)
    mi_idx = idx[:, :min_k]

    counter = 0
    counter_mi = 0

    for i, v in enumerate(idx):
        for vv in v:
            if y_train[i] != y_train[vv]:
                counter += 1
    error = counter / (y_train.shape[0] * knn)

    for i, v in enumerate(mi_idx):
        for vv in v:
            if y_train[i] != y_train[vv]:
                counter_mi += 1
    error_mi = counter_mi / (y_train.shape[0] * min_k)

    graph = np.empty(shape=[0, 2], dtype=int)
    for i, m in enumerate(idx):
        for mm in m:
            # print(i, mm)
            graph = np.append(graph, [[i, mm]], axis=0)
    return graph, error, mi_idx, error_mi

def KNN_Hypergraph(min_k=3, knn=10, data='', label='', metrix='minkowski'):
    """
    Define the function KNN_graph, which takes four parameters: mik (default value 3),
    knn (default value 10), data (default empty string), label (default empty string), metrix (default value 'Minkowski')
    : param min_k:  The minimum number of neighbors used to construct a graph
    : param knn: The K value in the KNN algorithm, which is the number of nearest neighbors considered
    : param data:  Features of training dataset
    : param label:  Label the training dataset
    : param metrix:  A measurement method used to calculate distance
    : return:   Return the constructed graph, error rate, minimum neighbor index, and error rate of the minimum neighbor.
    """
    x_train, y_train = data, label
    n_train = len(x_train)
    x_train_flat = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))[:n_train]
    train_neighbors = NearestNeighbors(n_neighbors=knn + 1, metric=metrix).fit(x_train_flat)
    _, idx = train_neighbors.kneighbors(x_train_flat)

    new_idx = np.empty((idx.shape[0], idx.shape[1] - 1))
    assert (idx >= 0).all()

    for i in range(idx.shape[0]):
        try:
            new_idx[i] = idx[i, idx[i] != i][:idx.shape[1] - 1]
        except Exception as e:
            print(idx[i, ...], new_idx.shape, idx.shape)
            raise e

    idx = new_idx.astype(int)
    mi_idx = idx[:, :min_k]

    counter = 0
    counter_mi = 0

    for i, v in enumerate(idx):
        for vv in v:
            if y_train[i] != y_train[vv]:
                counter += 1
    error = counter / (y_train.shape[0] * knn)

    for i, v in enumerate(mi_idx):
        for vv in v:
            if y_train[i] != y_train[vv]:
                counter_mi += 1
    error_mi = counter_mi / (y_train.shape[0] * min_k)

    graph = np.empty(shape=[0, 2], dtype=int)
    for i, m in enumerate(idx):
        for mm in m:
            # print(i, mm)
            graph = np.append(graph, [[i, mm]], axis=0)
    return graph, error, mi_idx, error_mi


def calculate_cosine_similarity(x_i, x_j):
    dot_product = np.dot(x_i, x_j)
    norm_i = np.linalg.norm(x_i)
    norm_j = np.linalg.norm(x_j)
    if norm_i == 0 or norm_j == 0:
        return 0
    return dot_product / (norm_i * norm_j)
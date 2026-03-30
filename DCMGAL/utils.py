import numpy as np
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import adjusted_mutual_info_score as ami_score
from sklearn import metrics
import torch
import torch.nn as nn
import dgl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if num_class1 != numclass2:
        print('error')
        return

    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    return acc, f1_macro, precision_macro, recall_macro

def eva(y_true, y_pred, state='', save_path=''):
    acc, f1, precision, recall = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    pur = purity_score(y_true, y_pred)

    with open(save_path, mode='a') as f:
        f.write(str(state) + ':acc {:.5f}'.format(acc) + ', nmi {:.5f}'.format(nmi) + ', pur {:.5f}'.format(pur) + '\n')
        if state == 'Final':
            f.write('\n')

    print(state, ':acc {:.5f}'.format(acc), ', nmi {:.5f}'.format(nmi), ', pur {:.5f}'.format(pur))

    return acc, nmi, pur

def purity_score(y_true, y_pred):
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]

    y_voted_labels = np.zeros(y_true.shape)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster],
                               bins=np.concatenate((np.unique(y_true), [np.max(labels) + 1])))
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def trace_loss(adj, k):
    adj = torch.clamp(adj, 0, 1)
    adj = torch.round(adj)
    rowsum = adj.sum(axis=1).detach().cpu().numpy()
    d = torch.zeros(adj.shape).numpy()
    row, col = np.diag_indices_from(d)
    d[row, col] = rowsum
    l = d - adj.detach().cpu().numpy()
    e_vals, e_vecs = np.linalg.eig(l)
    sorted_indices = np.argsort(e_vals)
    q = torch.tensor(e_vecs[:, sorted_indices[0:k:]].real.astype(np.float32)).to(
        device)
    m = torch.mm(torch.t(q), adj)
    m = torch.mm(m, q)
    return torch.trace(m)

def contrastive_loss(output_anchor, output_positive, margin=1.0):
    euclidean_distance = nn.functional.pairwise_distance(output_anchor, output_positive)
    losses = torch.relu(margin - euclidean_distance)
    return losses.mean()

def compare_adjacency_matrices(A_c, A_star):
    """
    Compare two adjacency matrices to find positive and negative sample pairs.

    Parameters:
    A_c (torch.Tensor): The first adjacency matrix.
    A_star (torch.Tensor): The second adjacency matrix.

    Returns:
    positive_pairs (torch.Tensor): Indices of positive sample pairs.
    negative_pairs (torch.Tensor): Indices of negative sample pairs.
    """
    # Ensure the adjacency matrices are binary
    A_c = (A_c > 0).int()
    A_star = (A_star > 0).int()

    # Find the indices where both matrices agree (positive pairs)
    both_neighbors = torch.logical_and(A_c, A_star)
    positive_pairs = torch.nonzero(both_neighbors, as_tuple=False)

    # Find the indices where one matrix indicates a neighbor but the other does not (negative pairs)
    xor_neighbors = torch.logical_xor(A_c, A_star)
    negative_pairs = torch.nonzero(xor_neighbors, as_tuple=False)

    return positive_pairs, negative_pairs

# def local_topological_structure_loss(Z, positive_pairs, negative_pairs, tau=0.1):
#     """
#     Calculate the local topological structure loss.
#
#     Parameters:
#     Z (torch.Tensor): The feature vectors of all samples.
#     positive_pairs (torch.Tensor): The indices of the positive sample pairs.
#     negative_pairs (torch.Tensor): The indices of the negative sample pairs.
#     tau (float): The temperature parameter.
#
#     Returns:
#     loss (torch.Tensor): The local topological structure loss.
#     """
#     device = Z.device
#     num_samples = Z.shape[0]
#     loss = torch.zeros(1, device=device)
#
#     for i in range(num_samples):
#         z_i = Z[i].to(device)  # Anchor feature vector
#
#         # Get the positive samples for the current anchor
#         pos_indices = torch.where(positive_pairs[:, 0].to(device) == i)[0]
#         if pos_indices.numel() > 0:
#             p2 = positive_pairs[pos_indices, 1].to(device)
#             # Check if p2 is within the bounds of Z
#             p2 = p2[p2 < num_samples]
#             if p2.numel() > 0:
#                 Z_i_plus = Z[torch.cat((torch.tensor([i], device=device), p2), dim=0)]  # Positive sample feature vectors
#                 corr_pos = torch.sum(Z_i_plus * z_i, dim=1) / (
#                     torch.norm(Z_i_plus, dim=1) * torch.norm(z_i)
#                 )
#                 sum_pos = torch.sum(torch.exp(corr_pos / tau))
#                 corr_pos_norm = corr_pos / tau - torch.log(sum_pos)
#                 loss = loss - torch.sum(corr_pos_norm)
#
#         # Get the negative samples for the current anchor
#         if negative_pairs.dim() == 2:  # Check if negative_pairs is 2D
#             neg_indices = torch.where(negative_pairs[:, 0].to(device) == i)[0]
#         else:  # If negative_pairs is 1D, use it directly
#             neg_indices = torch.where(negative_pairs.to(device) == i)[0]
#         if neg_indices.numel() > 0:
#             n1 = negative_pairs[neg_indices, 1] if negative_pairs.dim() == 2 else negative_pairs[neg_indices]
#             # Check if n1 is within the bounds of Z
#             n1 = n1[n1 < num_samples]
#             if n1.numel() > 0:
#                 Z_i_minus = Z[n1]  # Negative sample feature vectors
#                 corr_neg = torch.sum(Z_i_minus * z_i, dim=1) / (
#                         torch.norm(Z_i_minus, dim=1) * torch.norm(z_i)
#                 )
#                 sum_neg = torch.sum(torch.exp(corr_neg / tau))
#                 corr_neg_norm = corr_neg / tau - torch.log(sum_neg)
#                 loss = loss + torch.sum(corr_neg_norm)
#
#     return loss / num_samples




def visualize_clustering(save_name, X, y):
    X_detached = X.detach()
    X_numpy = X_detached.cpu().numpy()
    X_scaled = StandardScaler().fit_transform(X_numpy)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 6), dpi=100)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab20', marker='o', s=5)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                    labelleft=False)
    plt.title(save_name)

    plt.savefig(save_name, format='PNG')
    plt.show()

def shuffling(x, latent):
    idxs = torch.arange(0, x.shape[0]).to(device)
    a = torch.randperm(idxs.size(0)).to(device)
    aa = idxs[a].unsqueeze(1)
    aaa = aa.repeat(1, latent)
    return torch.gather(x, 0, aaa)
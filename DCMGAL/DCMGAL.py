from dgl.nn.pytorch import GraphConv as GCN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import dgl
import networkx as nx
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

class DCMGAL(nn.Module):
    def __init__(self, in_dim, hidden_dims_v, hidden_dims, views, v=1):
        super(DCMGAL, self).__init__()
        self.views = views
        self.layers = create_layers(in_dim, hidden_dims_v, views)
        self.rx_layers = create_decoder_layers(hidden_dims_v, in_dim, views)
        self.layer_m = create_last_layers(hidden_dims, hidden_dims_v, views)
        self.dualfusion_H = DualFeatureFusion(m=hidden_dims_v[-1])
        self.decoder = InnerProductDecoder(activation=lambda x: x, size=hidden_dims[-1])
        self.v = v

    def forward(self, graphs, features, Graph):
        graphs = [g.to(device) for g in graphs]
        features = [f.to(device) for f in features]
        Graph = Graph.to(device=device)

        hs = [feature for feature in features]
        rhs = []
        loss_xrec = []
        loss_fn = DynamicCosineSimilarityLoss()

        for i in range(self.views):
            for conv in self.layers[i]:
                hs[i] = conv(graphs[i], hs[i])
        for i in range(self.views):
            for conv in self.rx_layers[i]:
                rhs.append(conv(graphs[i], hs[i]))
        for i in range(self.views):
            loss_xrec.append(loss_fn(hs[i], rhs[i]))


        H_c = self.dualfusion_H(hs)
        H_c = H_c.to(device=device)

        # self-expression layer:
        lambda_0 = 0.1
        ser = SelfExpressionReconstruction(H_c.clone(), lambda_0)
        H_c = ser.fit()

        for conv in self.layer_m:
            if conv == self.layer_m[0]:
                H_c = conv(dgl.add_self_loop(Graph), H_c.clone())
            else:
                H_c = conv(dgl.add_self_loop(Graph), H_c.clone())

        adj_rec = {}
        for i in range(self.views):
            adj_rec[i] = self.decoder(H_c)

        return loss_xrec, adj_rec, H_c

class DynamicCosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(DynamicCosineSimilarityLoss, self).__init__()
    def forward(self, h, rh):
        rh_dim = rh.size(1)
        if h.size(1) != rh_dim:
            linear_layer = nn.Linear(h.size(1), rh_dim).to(device)
            h = linear_layer(h)

        assert h.size() == rh.size(), "The size of h and rh must be the same after dimension matching."

        dot_product = torch.sum(h * rh, dim=1)
        norm_h = torch.norm(h, p=2, dim=1)
        norm_rh = torch.norm(rh, p=2, dim=1)
        cosine_similarity = dot_product / (norm_h * norm_rh)
        loss = 1 - cosine_similarity.mean()

        return loss

class DualFeatureFusion(nn.Module):
    def __init__(self, m):
        super(DualFeatureFusion, self).__init__()
        self.m = m
        self.K_v = nn.Parameter(torch.randn(m+1, m))
        self.q_v = nn.Parameter(torch.randn(m))
        self.Theta_c = nn.Parameter(torch.randn(m, m))

    def compute_alpha(self, H_c):
        H_c = H_c.detach()  # 隔离输入张量
        H_c_expanded = torch.cat((H_c, torch.ones(H_c.size(0), 1, device=H_c.device)), dim=1)
        alpha_v = torch.tanh(torch.matmul(H_c_expanded.clone(), self.K_v.detach()) * self.q_v.detach())
        return alpha_v

    def compute_w_h(self, alpha_v):
        w_h_v = F.softmax(alpha_v, dim=0)
        return w_h_v

    def fuse_views(self, H_v_list):
        V = len(H_v_list)
        device = H_v_list[0].device
        batch_size = H_v_list[0].size(0)
        feature_dim = self.Theta_c.size(1)

        H_c = torch.zeros(batch_size, feature_dim, device=device)
        alpha_v_list = [self.compute_alpha(H_v.clone()) for H_v in H_v_list]
        w_h_v_list = [self.compute_w_h(alpha_v.clone()) for alpha_v in alpha_v_list]

        Theta_c = self.Theta_c.clone()
        H_v_processed = [torch.matmul(H_v.clone(), Theta_c) for H_v in H_v_list]
        H_v_processed = torch.stack(H_v_processed, dim=2)

        for v in range(V):
            w_h_v = w_h_v_list[v].unsqueeze(2)
            H_c += torch.sum(w_h_v * H_v_processed[:, :, v].unsqueeze(2), dim=1)

        return H_c

    def forward(self, H_v_list):
        return self.fuse_views(H_v_list)

class DualAdjacencyFusion(nn.Module):
    def __init__(self):
        super(DualAdjacencyFusion, self).__init__()

    def calculate_beta_v(self, S_v, l):
        n = S_v.shape[0]
        beta_v = 0
        epsilon = 1e-10
        for i in range(n):
            for j in range(n):
                if S_v[i, j] > epsilon and 1 - S_v[i, j] > epsilon:
                    beta_v += (-l[i, j] * np.log(S_v[i, j]) - (1 - l[i, j]) * np.log(1 - S_v[i, j]))
                else:
                    beta_v += 0
        return beta_v

    def calculate_w_v(self, beta_v, all_betas):
        """根据公式(8)计算w_v"""
        epsilon = 1e-10
        if any(b is None for b in all_betas):
            raise ValueError("all_betas contains None values")
        all_betas = np.clip(all_betas, None, 100)
        exp_all_betas = np.exp(all_betas)
        sum_exp_all_betas = np.sum(exp_all_betas)
        if sum_exp_all_betas < epsilon:
            w_v = np.ones_like(beta_v) / len(beta_v)
        else:
            w_v = exp_all_betas / sum_exp_all_betas
        return w_v

    def calculate_l_list(self, feature):
        l_list = []
        for x in feature:
            n = len(x)
            S_v = np.array([[calculate_cosine_similarity(x[i], x[j]) for j in range(n)] for i in range(n)])
            l = (S_v > 0.8).astype(int)
            l_list.append(l)
        return l_list

    def forward(self, A_v_list, feature):
        l_list = self.calculate_l_list(feature)
        V = len(A_v_list)
        n = len(feature[0])
        beta_v_list = []

        for i in range(V):
            S_v = cosine_similarity(A_v_list[i])
            beta_v = self.calculate_beta_v(S_v, l_list[i])
            beta_v_list.append(beta_v)

        try:
            w_v_list = self.calculate_w_v(beta_v_list[0], beta_v_list)
        except ValueError as e:
            print(e)
            return None
        A_c = torch.zeros((n, n))
        for w_v, A_v in zip(w_v_list, A_v_list):
            if w_v.ndim != 0 or A_v.ndim != 2 or A_v.shape != A_c.shape:
                raise ValueError("w_v must be a scalar and A_v must be a 2D array with shape matching A_c")
            A_c += w_v * A_v
        return A_c

class InnerProductDecoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout=0.1, size=10):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.weight = Parameter(torch.FloatTensor(size, size))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        t = torch.mm(z, self.weight)
        adj = self.activation(torch.mm(z, z.t()))
        return adj

class Dis(nn.Module):
    def __init__(self, latent_dim=120):
        super(Dis, self).__init__()
        self.latent_dim = latent_dim
        self.discriminator = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.discriminator(z)
        return out

class LabelConsistencyContrastive(nn.Module):
    def __init__(self, z):
        super(LabelConsistencyContrastive, self).__init__()
        self.fc1 = nn.Linear(z.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class LocalTopologicalStructureContrastive(nn.Module):
#     def __init__(self, tau=0.1):
#         super(LocalTopologicalStructureContrastive, self).__init__()
#         self.tau = tau
#
#     def forward(self, Z, positive_pairs, negative_pairs):
#         device = Z.device
#         num_samples = Z.shape[0]
#         total_loss = torch.tensor(0.0, device=device, requires_grad=True)
#
#         for i in range(num_samples):
#             z_i = Z[i].unsqueeze(0)  # Anchor feature vector
#
#             # Positive samples
#             pos_indices = torch.where(positive_pairs[:, 0].to(device) == i)[0]
#             if pos_indices.numel() > 0:
#                 p2 = positive_pairs[pos_indices, 1].to(device)
#                 p2 = p2[p2 < num_samples]
#                 if p2.numel() > 0:
#                     i_tensor = torch.tensor([i], dtype=torch.long, device=device)
#                     Z_i_plus = Z[torch.cat((i_tensor, p2), dim=0)]
#                     corr_pos = torch.sum(Z_i_plus * z_i, dim=1) / (
#                         torch.norm(Z_i_plus, dim=1) * torch.norm(z_i)
#                     )
#                     sum_pos = torch.sum(torch.exp(corr_pos / self.tau))
#                     corr_pos_norm = corr_pos / self.tau - torch.log(sum_pos)
#                     total_loss = total_loss - torch.sum(corr_pos_norm)
#
#             # Negative samples
#             if negative_pairs.dim() == 2:
#                 neg_indices = torch.where(negative_pairs[:, 0].to(device) == i)[0]
#             else:
#                 neg_indices = torch.where(negative_pairs.to(device) == i)[0]
#             if neg_indices.numel() > 0:
#                 n1 = negative_pairs[neg_indices, 1] if negative_pairs.dim() == 2 else negative_pairs[neg_indices]
#                 n1 = n1[n1 < num_samples]
#                 if n1.numel() > 0:
#                     Z_i_minus = Z[n1]
#                     corr_neg = torch.sum(Z_i_minus * z_i, dim=1) / (
#                             torch.norm(Z_i_minus, dim=1) * torch.norm(z_i)
#                     )
#                     sum_neg = torch.sum(torch.exp(corr_neg / self.tau))
#                     corr_neg_norm = corr_neg / self.tau - torch.log(sum_neg)
#                     total_loss = total_loss + torch.sum(corr_neg_norm)
#
#         return total_loss / num_samples

class LocalTopologicalStructureContrastive(nn.Module):
    def __init__(self, tau=0.1):
        super(LocalTopologicalStructureContrastive, self).__init__()
        self.tau = tau

    def forward(self, Z, positive_pairs, negative_pairs):
        device = Z.device
        num_samples = Z.shape[0]
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for i in range(num_samples):
            z_i = Z[i].unsqueeze(0)  # Anchor feature vector

            # Positive samples
            pos_indices = torch.where(positive_pairs[:, 0].to(device) == i)[0]
            if pos_indices.numel() > 0:
                p2 = positive_pairs[pos_indices, 1].to(device)
                p2 = p2[p2 < num_samples]
                if p2.numel() > 0:
                    Z_i_plus = Z[torch.cat((torch.tensor([i], dtype=torch.long, device=device), p2), dim=0)]
                    corr_pos = self.correlation(z_i, Z_i_plus)
                    neg_samples = self.get_negative_samples(Z, i, device)
                    corr_neg = self.correlation(z_i, neg_samples)
                    pos_loss = -torch.log(corr_pos / (corr_pos + corr_neg))
                    total_loss = total_loss + pos_loss
        loss = total_loss / num_samples

        return loss

    def correlation(self, z_i, Z):
        cos_sim = torch.sum(Z * z_i, dim=1) / (torch.norm(Z, dim=1) * torch.norm(z_i))
        return torch.sum(torch.exp(cos_sim / self.tau))

    def get_negative_samples(self, Z, i, device):
        # This function should return all negative samples for the anchor i
        neg_samples = torch.arange(Z.shape[0], device=device)
        neg_samples = neg_samples[neg_samples != i]
        return Z[neg_samples]
class SelfExpressionReconstruction(nn.Module):
    def __init__(self, Y, lambda_0, learning_rate=0.01, max_iter=1000):
        super(SelfExpressionReconstruction, self).__init__()
        self.Y = Y
        self.lambda_0 = lambda_0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.C = None
        self.Z = None

    def _initialize_C(self, n_features):
        return torch.rand(n_features, n_features, requires_grad=True)

    def _update_C(self, grad):
        C_new = self.C - self.learning_rate * grad
        C_new = torch.clamp(C_new - torch.diag_embed(torch.diag(C_new)), min=0)
        diag_mask = torch.ones_like(C_new)
        diag_mask.fill_diagonal_(0)
        C_new *= diag_mask
        return C_new.detach().requires_grad_(True)

    def optimize_C(self):
        n_samples, n_features = self.Y.shape
        self.C = self._initialize_C(n_features)

        for _ in range(self.max_iter):
            grad = -self.Y.T @ (self.Y - self.Y @ self.C) + self.lambda_0 * torch.sign(self.C)
            self.C = self._update_C(grad)

        return self.C

    def calculate_Z(self):
        if self.C is None:
            self.optimize_C()
        self.Z = self.Y @ self.C
        return self.Z

    def fit(self):
        return self.calculate_Z()

def create_layers(in_dim, hidden_dims_v, views):
    layers = nn.ModuleList()

    for view_index in range(views):
        if len(hidden_dims_v) >= 2:
            layer = [GCN(in_feats=in_dim[view_index], out_feats=hidden_dims_v[0], activation=F.relu)]
            for j in range(1, len(hidden_dims_v)):
                if j != len(hidden_dims_v) - 1:
                    layer.append(GCN(in_feats=hidden_dims_v[j - 1], out_feats=hidden_dims_v[j], activation=F.relu))
                else:
                    layer.append(GCN(in_feats=hidden_dims_v[j - 1], out_feats=hidden_dims_v[j], activation=lambda x: x))
            layers.append(nn.ModuleList(layer))
        else:
            single_layer = GCN(in_feats=in_dim[view_index], out_feats=hidden_dims_v[0], activation=lambda x: x)
            layers.append(single_layer)
    return layers

def create_decoder_layers(hidden_dims_v, out_dim, views):
    decoder_layers = nn.ModuleList()

    for view_index in range(views):
        if len(hidden_dims_v) >= 2:
            layer = [GCN(in_feats=hidden_dims_v[-1], out_feats=hidden_dims_v[-2])]
            for j in range(len(hidden_dims_v) - 2, 0, -1):
                if j != 0:
                    layer.append(GCN(in_feats=hidden_dims_v[j], out_feats=hidden_dims_v[j - 1]))
                else:
                    layer.append(GCN(in_feats=hidden_dims_v[j], out_feats=out_dim[view_index]))
            decoder_layers.append(nn.ModuleList(layer[::-1]))
        else:
            single_layer = GCN(in_feats=hidden_dims_v[0], out_feats=out_dim[view_index])
            decoder_layers.append(single_layer)
    return decoder_layers

def create_last_layers(hidden_dims, hidden_dims_v, views):
    layer_m = nn.ModuleList()
    if len(hidden_dims) >= 2:
        layer_m.append(GCN(in_feats=int(hidden_dims_v[-1]), out_feats=hidden_dims[0], activation=F.relu))
        for j in range(1, len(hidden_dims)):
            if j != len(hidden_dims) - 1:
                layer_m.append(GCN(in_feats=hidden_dims[j - 1], out_feats=hidden_dims[j], activation=F.relu))
            else:
                layer_m.append(GCN(in_feats=hidden_dims[j - 1], out_feats=hidden_dims[j], activation=lambda x: x))
    else:
        layer_m.append(GCN(in_feats=int(hidden_dims_v[-1] * views), out_feats=hidden_dims[0], activation=lambda x: x))
    return layer_m

def calculate_cosine_similarity(x_i, x_j):
    dot_product = np.dot(x_i, x_j)
    norm_i = np.linalg.norm(x_i)
    norm_j = np.linalg.norm(x_j)
    if norm_i == 0 or norm_j == 0:
        return 0
    return dot_product / (norm_i * norm_j)
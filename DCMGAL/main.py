import argparse
import datetime
from sklearn.cluster import KMeans
from DCMGAL import DCMGAL, Dis, LabelConsistencyContrastive, DualAdjacencyFusion, LocalTopologicalStructureContrastive
from datasetloader import *
from utils import *
import networkx as nx
from tqdm import tqdm
import os
os.environ["OMP_NUM_THREADS"] = '1'

parser = argparse.ArgumentParser(description='DCMGAL')
parser.add_argument('--dataset', type=str, default='BDGP', help='choose a dataset')
parser.add_argument('--mask_T', type=lambda x: (str(x).lower() == 'true'), default=False, help='Choose whether to mask the view')
parser.add_argument('--mask_rate', type=float, default=0.70, help='mask rate of edges')
parser.add_argument('--ptrain_epochs', '-pe', type=int, default=500, help='number of pre-train_epochs')
parser.add_argument('--train_epochs', '-te', type=int, default=500, help='number of train_epochs')
parser.add_argument('--hidden_dimsV', type=int, nargs='+', default=[64, 32], help='list of V1 hidden dimensions')
parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 10], help='list of feature hidden dimensions')
parser.add_argument('--plr', type=float, default=0.0005, help='Adam learning rate')
parser.add_argument('--tlr', type=float, default=0.01, help='Adam learning rate')
parser.add_argument('--lambda1', type=float, default=0.1, help='Rate for ser')
parser.add_argument('--lambda2', type=float, default=1, help='Rate for rec')
parser.add_argument('--lambda3', type=float, default= 0.01, help='Rate for clu')
parser.add_argument('--lambda4', type=float, default= 10, help='Rate for con')
parser.add_argument('--save_index', type=str, default='001', help='choose a result index')
parser.add_argument('--repetitions', '-re', type=int, default=10, help='Experimental repetition times')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("use cuda: {}".format(args.cuda))

def main():
    global args
    data = load_data(args)
    views = len(data.feature)
    print('Number of Samples: {:02d}'.format(data.feature[0].shape[0]))
    print('Views:', views)

    features = [torch.FloatTensor(data.feature[i]) for i in range(views)]
    in_feats = [features[i].shape[1] for i in range(views)]
    graphs = [dgl.from_networkx(data.graph_dict[i]) for i in range(views)]

    for i in range(views):
        print('Feature-{:02d} feature shape: {}'.format(i+1, features[i].shape))
    for i in range(views):
        print('View-{:02d} graph edge: {}'.format(i+1, graphs[i].number_of_edges()))


    miks = [data.mi_dict[i] for i in range(views)]
    Mik = np.hstack(miks)

    degs = [graphs[i].in_degrees().float() for i in range(views)]
    norms = [torch.pow(degs[i], -0.5) for i in range(views)]
    for i in range(views):
        norms[i][torch.isinf(norms[i])] = 0
    for i in range(views):
        graphs[i].ndata['norm'] = norms[i].unsqueeze(1)

    adjs = [graphs[i].adjacency_matrix().to_dense() for i in range(views)]
    edges = sum(g.number_of_edges() for g in graphs)
    y = data.label
    n_clusters = len(np.unique(y))

    model = DCMGAL(in_feats, args.hidden_dimsV, args.hidden_dims, views)
    model.train()

    model_d = Dis(latent_dim=args.hidden_dims[-1])
    model_d.train()

    fusion_model = DualAdjacencyFusion()
    fusion_model.train()

    # optimizer
    optim_gmae_p = torch.optim.Adam(model.parameters(), lr=args.plr)
    optim_gmae_t = torch.optim.Adam(model.parameters(), lr=args.tlr)

    # loss
    pos_weight = torch.Tensor([float(graphs[0].adjacency_matrix().to_dense().shape[0] ** 2 - edges / 2) / edges * 2])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_m = torch.nn.MSELoss()

    # To GPU
    criterion.to(device=device)
    criterion_m.to(device=device)
    model = model.to(device=device)
    model_d = model_d.to(device=device)
    fusion_model = fusion_model.to(device=device)
    for i in range(views):
        graphs[i] = graphs[i].to(device)
        features[i] = features[i].to(device)
        adjs[i] = adjs[i].to(device)

    adj_r = fusion_model.forward(adjs, features)
    adj_p = torch.clamp(adj_r, 0, 1)
    adj_p = torch.round(adj_p + 0.1)
    adj_pn = adj_p.detach().cpu().numpy()
    adj_pn += adj_pn.T
    Graph = nx.from_numpy_array(adj_pn, create_using=nx.DiGraph())
    Graph = dgl.from_networkx(Graph)
    Graph = Graph.to(device=device)
    print('graph.number_of_edges', Graph.number_of_edges())

    print('----------------------DCMGAL Pre-Training Start----------------------')
    with tqdm(total=args.ptrain_epochs, desc="Pre-Training Progress", unit="epoch") as pbar:
        for epoch in range(args.ptrain_epochs):
            model.train()
            lrx, adj_logits, h = model.forward(graphs, features, Graph)
            loss_rec = sum(l for l in lrx) / views

            optim_gmae_p.zero_grad()
            loss_rec.backward(retain_graph=True)
            optim_gmae_p.step()

            pbar.update(1)
            pbar.set_postfix({"Current Epoch": epoch + 1})

    # Obtain the initial cluster center.
    model.eval()
    _, _, z = model.forward(graphs, features, Graph)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    _, _, _ = eva(y, y_pred, 'Pre-train', save_path=save_path)

    # Training
    print('----------------------Training Start----------------------')
    with tqdm(total=args.train_epochs, desc="Training Progress", unit="epoch") as pbar:
        # loss = []
        for epoch in range(args.train_epochs):
            model.train()
            adj_r = fusion_model.forward(adjs, features)
            loss_gre = sum(criterion_m(adj_r, adj) for adj in adjs) / views
            loss_gtr = trace_loss(adj_r, n_clusters) ** 2
            loss_ge = loss_gre + args.lambda1 * loss_gtr

            # normalization
            adj_p = torch.clamp(adj_r, 0, 1)
            adj_p = torch.round(adj_p + 0.1)
            adj_pn = adj_p.detach().cpu().numpy()
            adj_pn += adj_pn.T
            Graph = nx.from_numpy_array(adj_pn)
            Graph = dgl.from_networkx(Graph)
            Graph = Graph.to(device=device)

            lrx, adj_logits, h = model.forward(graphs, features, Graph)
            loss_rec = sum(l for l in lrx) / views

            global_info_loss = 0
            for i in range(Mik.shape[1]):
                h_shuffle = shuffling(h, latent=args.hidden_dims[-1])
                h_h_shuffle = torch.cat((h, h_shuffle), 1)
                h_h_shuffle_scores = model_d(h_h_shuffle)
                h_h = torch.cat((h, h[Mik[:, i]]), 1)
                h_h_scores = model_d(h_h)
                global_info_loss += - torch.mean(
                    torch.log(h_h_scores + 1e-6) + torch.log(1 - h_h_shuffle_scores + 1e-6))

            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
            y_pred_ = kmeans.fit_predict(h.data.cpu().numpy())

            #Cross-view Label Consistency Contrastive Learning
            model_c = LabelConsistencyContrastive(h)
            model_c = model_c.to(device=device)
            model_c.train()

            indices = np.random.choice(z.shape[0], 100, replace=False)
            anchor_indices = indices[:50]
            positive_indices = indices[50:100]
            anchors = h[anchor_indices]
            positives = h[positive_indices]

            same_cluster_indices = y_pred_[anchor_indices] == y_pred_[positive_indices]
            anchors = anchors[same_cluster_indices].to(device=device)
            positives = positives[same_cluster_indices].to(device=device)

            output_anchor = model_c(anchors)
            output_positive = model_c(positives)
            loss_clu = contrastive_loss(output_anchor, output_positive)

            #Local Topological Structure Contrastive Learning
            model_l = LocalTopologicalStructureContrastive(tau=0.1)
            model_l = model_l.to(device=device)
            model_l.train()
            G, _, _, _ = KNN_graph(3, 10, z.detach().numpy(), y_pred_ , metrix='cosine')
            G_nx = nx.DiGraph()
            for edge in G:
                start_node, end_node = edge
                G_nx.add_edge(start_node, end_node)
            G_star = dgl.from_networkx(G_nx)
            A_star = G_star.adjacency_matrix().to_dense()
            positive_pairs, negative_pairs = compare_adjacency_matrices(adj_r, A_star)
            loss_con = model_l(z, positive_pairs, negative_pairs)

            #Calculate the total loss
            loss_gmae =loss_ge + loss_rec + args.lambda2 * global_info_loss + args.lambda3 * loss_clu + args.lambda4 * loss_con
            # loss.append(float(loss_gmae))

            optim_gmae_t.zero_grad()
            loss_gmae.backward(retain_graph=True)
            optim_gmae_t.step()

            if (epoch + 1) % 100 == 0:
                model.eval()
                _, _, z = model.forward(graphs, features, Graph)
                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
                y_pred = kmeans.fit_predict(z.data.cpu().numpy())
                _, _, _ = eva(y, y_pred, 'Mid-train', save_path=save_path)
            pbar.update(1)
            pbar.set_postfix({"Current Epoch": epoch + 1})

    # plt.figure(figsize=(10, 5))
    # epochs = [i for i in range(1, args.train_epochs + 1)]
    # plt.plot(epochs, loss, label='Line Plot')
    # plt.legend()
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.show()

    model.eval()
    _, _, z = model.forward(graphs, features, Graph)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    # save_name = f'./result/{args.dataset}/{args.dataset}_{str(args.mask_rate*100)}.png'
    # visualize_clustering(save_name, z, y_pred)

    acc, nmi, pur = eva(y, y_pred, 'Fin-train', save_path=save_path)
    return acc, nmi, pur

if __name__ == '__main__':
    if args.mask_T:
        save_path = f'./result/{args.dataset}/{args.dataset}_{args.save_index}_mask_rate_{str(int(args.mask_rate * 100))}.txt'
    else:
        save_path = f'./result/{args.dataset}/{args.dataset}_{args.save_index}_raw_graph.txt'
    accA = []
    nmiA = []
    purA = []
    for i in range(args.repetitions):
        acc, nmi, pur = main()
        accA.append(acc)
        nmiA.append(nmi)
        purA.append(pur)
    ACC = np.mean(accA)
    NMI = np.mean(nmiA)
    PUR = np.mean(purA)

    print(ACC, "\n", NMI, "\n",  PUR)
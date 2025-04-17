from cgi import test
import warnings
warnings.filterwarnings('ignore')

from utils.dataloader import CustomDataset
import torch
from sklearn.model_selection import StratifiedKFold, KFold
from model.model import MLP, GIN, GraphTrans
from model.loss import AUCPRHingeLoss,aucpr_hinge_loss
import torch.optim as optim
from torch.nn import CrossEntropyLoss, BCELoss
import torch.nn as nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
from model.loss import contrastive_loss_cell, mae_loss_cell
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score
from glob import glob
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as DataLoader_PyG
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import math
import json
from tqdm import tqdm
from torch_geometric.nn.pool import global_add_pool
from argparse import ArgumentParser
import gc
gc.enable()
from torch_geometric.data import Data

import logging

import networkx as nx
import torch
from torch_geometric.utils import to_networkx, subgraph
from torch_geometric.data import Data
import pymetis

def partition_graph(data: Data, part_size=500):
    G = to_networkx(data, to_undirected=True)
    num_parts =  math.ceil(data.num_nodes / part_size)

    # PyMetis requires adjacency list
    node_to_idx = {node: i for i, node in enumerate(G.nodes)}
    adj_list = [[] for _ in range(G.number_of_nodes())]
    for u, v in G.edges:
        adj_list[node_to_idx[u]].append(node_to_idx[v])
        adj_list[node_to_idx[v]].append(node_to_idx[u])

    _, membership = pymetis.part_graph(num_parts, adjacency=adj_list)

    # Group nodes by partition id
    partitions = [[] for _ in range(num_parts)]
    for node, part_id in zip(G.nodes, membership):
        partitions[part_id].append(node_to_idx[node])

    # Create subgraphs
    subgraphs = []
    for part_nodes in partitions:
        part_nodes = torch.tensor(part_nodes, dtype=torch.long)
        edge_idx, _ = subgraph(part_nodes, data.edge_index, relabel_nodes=True)
        X = data.X[part_nodes]
        pos = data.pos[part_nodes]
        batch = torch.zeros(part_nodes.size(0), dtype=torch.long)
        sub_data = Data(X=X, pos=pos, edge_index=edge_idx, batch=batch)
        sub_data.cell_type = data.cell_type[part_nodes] if hasattr(data, "cell_type") else None
        subgraphs.append(sub_data)

    return subgraphs

parser = ArgumentParser(description="SCGFM")
parser.add_argument('--data_name', type=str, help="Name of the dataset")
parser.add_argument('--random_state', type=int, default=0, help="Random state")
parser.add_argument('--label_name', type=str, help="Name of the label")
parser.add_argument('--input_modality', type=str, default = "x", help="Name of the label")
parser.add_argument('--init_dim', type=int, default=64, help="Hidden dim for the MLP")
parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden dim for the MLP")
parser.add_argument('--latent_dim', type=int, default=64, help="Output dim for the MLP")
parser.add_argument('--num_layers', type=int, default=10, help="Number of MLP layers")
parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning Rate")
parser.add_argument('--wd', type=float, default=1e-4, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs")
parser.add_argument('--gpu', type=int, default=0, help="GPU index")
parser.add_argument('--model_type', type=str, default='GIN', choices=['MLP', 'GIN'], help="Model type to train (MLP or GIN)")

args = parser.parse_args()
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'
torch.autograd.set_detect_anomaly(True)

class GraphMAE(nn.Module):
    def __init__(self, input_dim, init_dim, hidden_dim, num_layers=10, mask_ratio=0.4, input_modality='x'):
        super(GraphMAE, self).__init__()
        self.input_modality = input_modality
        self.mask_ratio = mask_ratio
        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.mlp_x = nn.Linear(input_dim, init_dim)
        self.mlp_x_2 = nn.Linear(1, input_dim)
        self.mlp_pos = nn.Linear(2, init_dim)
        self.encoder = GraphTrans(init_dim, hidden_dim, input_dim, 4, num_layers)
        
        self.decoder_x = GIN(input_dim, hidden_dim, input_dim, 1)
        self.decoder_pos = GIN(input_dim, hidden_dim, 2, 1)

        # If only one input modality is used, learn to predict the other latent for contrastive loss
        self.cross_modal_mlp = nn.Linear(hidden_dim, hidden_dim)

    def mask_features(self, x, gene=True):
        if gene:
            mask = torch.bernoulli(torch.ones_like(x)*0.3).long().to(args.device)
            masked_x = x * (1 - mask.long())
        else:
            mask = torch.bernoulli(torch.ones(x.shape[0])*0.3).long().to(args.device)#(torch.rand(x.size(0)) < self.mask_ratio).to()
            masked_x = x.clone()
            masked_x[mask] = 0
        return masked_x.to(args.device), mask.long().to(args.device)
    
    def forward(self, x, pos, edge_index, cell_type, batch):
        # Mask both x and pos (you may need both for recon, even if not used for encoding)
        masked_x, mask_x = self.mask_features(x, gene=True)
        masked_pos, mask_pos = self.mask_features(pos, gene=False)

        # Encode using only one modality
        if self.input_modality == 'x':
            _x = self.mlp_x(masked_x)
            latent_x = self.encoder(_x, edge_index, batch)
            latent_pos = self.cross_modal_mlp(latent_x)
            latent_x = latent_x.masked_fill(mask_x.bool(), 0)
            latent_pos = latent_pos.masked_fill(mask_pos.unsqueeze(-1).bool(), 0)
            reconstructed_x = self.decoder_x(latent_x, edge_index, batch)
            reconstructed_pos = self.decoder_pos(latent_x, edge_index, batch)
        elif self.input_modality == 'pos':
            _pos = self.mlp_pos(masked_pos)
            latent_pos = self.encoder(_pos, edge_index, batch)
            latent_x = self.cross_modal_mlp(latent_pos)
            latent_x = latent_x.masked_fill(mask_x.bool(), 0)
            latent_pos = latent_pos.masked_fill(mask_pos.unsqueeze(-1).bool(), 0)
            reconstructed_x = self.decoder_x(latent_pos, edge_index, batch)
            reconstructed_pos = self.decoder_pos(latent_pos, edge_index, batch)
        else:
            raise ValueError("input_modality must be either 'x' or 'pos'")

        # Losses

        cells, genes = latent_x.shape
        latent_x = self.mlp_x_2(latent_x.view(cells*genes, 1))
        batch = Data(x=None, edge_index=None, batch=torch.arange(cells).repeat_interleave(genes)).to(latent_x.device)
        contrastive = contrastive_loss_cell(cell_type, latent_pos, batch, latent_x, 2)
        import pdb; pdb.set_trace()
        recon = mae_loss_cell(pos, x.view(genes*cells,1), reconstructed_pos, reconstructed_x.view(genes*cells,1), mask_pos.view(len(mask_pos), 1), mask_x.view(genes*cells, 1))
        # orthogonal = (
        #     0.1 * (latent_x.T @ latent_x - torch.eye(latent_x.shape[1]).to(args.device)).square().mean() +
        #     0.1 * (latent_pos.T @ latent_pos - torch.eye(latent_pos.shape[1]).to(args.device)).square().mean()
        # )

        loss = F.sigmoid(self.alpha) * contrastive + (1 - F.sigmoid(self.alpha)) * recon #+ orthogonal
        return loss


def train(model, data_loader, val_loader, optimizer, device):# -> Any:
    model.train()
    best_val_loss = torch.inf
    for epoch in (range(args.num_epochs)):
            total_loss = 0
            print(f"Training Epoch {epoch+1}")
            for data in tqdm(data_loader):
                try:
                    data = data.to(device)
                    optimizer.zero_grad()
                    loss = model(data.X, data.pos, data.edge_index, data.cell_type, data.batch)
                    if torch.isnan(loss):
                        print("Skipping batch due to NaN loss")
                        continue
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    total_loss += loss.item()
                except torch.cuda.OutOfMemoryError as e:
                    logging.error(f"CUDA OOM occurred:", e)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
            print(f"Validating Epoch {epoch+1}")
            val_loss = test(model, val_loader, device)
            if(best_val_loss>val_loss):
                best_val_loss = val_loss
                best_val_loss = val_loss
                model_path = f"saved_models/{args.input_modality}_only.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'args': args,
                    'epoch': epoch,
                    'best_val_loss':best_val_loss
                }, model_path)
                print(f"✅ New best model saved at Epoch {epoch+1} with Val Loss {val_loss:.4f}")
                    
            print(f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss = {total_loss:.4f}, Validation loss = {val_loss:.4f}, Best Validation loss = {best_val_loss:.4f}")
    return model

def test(model, graphs, device):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        for data in tqdm(graphs):
            data = data.to(device)
            loss = model(data.X, data.pos, data.edge_index, data.cell_type, data.batch)
            if torch.isnan(loss):
                    print("Skipping batch due to NaN loss")
                    continue
            total_loss += loss.item()
    return total_loss

if __name__ == '__main__':
    print(args)
    print("Loading graphs")
    graphs_list = [torch.load(file) for file in tqdm(sorted(glob("data/pretraining/*/*"))[:5])]
    cell_graphs = []
    print("Preprocessing datasets")
    for i in tqdm(range(len(graphs_list))):
        graphs = graphs_list[i]
        X = []
        for k in range(1, len(graphs)):
            X.append(graphs[k].X.squeeze(1).tolist())
        cell_graph = graphs[0]
        cell_graph.pos = cell_graph.X.float()
        cell_graph.X = torch.Tensor(X).float()
        if(cell_graph.X.shape[1]>128):
            cell_graph.X = cell_graph.X[:, :128]
        elif(cell_graph.X.shape[1]<128):
            pad_size = 128 - cell_graph.X.shape[1]
            padding = torch.zeros(cell_graph.X.shape[0], pad_size, device=cell_graph.X.device, dtype=cell_graph.X.dtype)
            cell_graph.X = torch.cat([cell_graph.X, padding], dim=1)
        cell_graphs.append(cell_graph)
        del(graphs)
    train_idx, val_idx = train_test_split(np.arange(len(cell_graphs)), test_size = 0.2, random_state = 42)

    all_train_subgraphs = []
    print("Partitioning the train graphs")
    for i in tqdm(train_idx):
        subgraphs = partition_graph(cell_graphs[i], part_size=10000)
        all_train_subgraphs.extend(subgraphs)

    train_loader = DataLoader_PyG(all_train_subgraphs, batch_size=1, shuffle=True)
    print("Partitioning the val graphs")
    all_val_subgraphs = []
    for i in tqdm(val_idx):
        subgraphs = partition_graph(cell_graphs[i], part_size=10000)
        all_val_subgraphs.extend(subgraphs)

    val_loader = DataLoader_PyG(all_val_subgraphs, batch_size=1, shuffle=False)
    model = GraphMAE(cell_graphs[0].X.shape[1], cell_graphs[0].X.shape[1], cell_graphs[0].X.shape[1], args.num_layers, input_modality=args.input_modality).to(args.device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay = args.wd)
    model = train(model, train_loader, val_loader, optimizer, args.device)
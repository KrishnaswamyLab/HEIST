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
from model.loss import contrastive_loss_cell_single_view, mae_loss_cell
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

parser = ArgumentParser(description="SCGFM")
parser.add_argument('--data_name', type=str, help="Name of the dataset")
parser.add_argument('--random_state', type=int, default=0, help="Random state")
parser.add_argument('--label_name', type=str, help="Name of the label")
parser.add_argument('--init_dim', type=int, default=64, help="Hidden dim for the MLP")
parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden dim for the MLP")
parser.add_argument('--latent_dim', type=int, default=64, help="Output dim for the MLP")
parser.add_argument('--num_layers', type=int, default=10, help="Number of MLP layers")
parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
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
    def __init__(self, input_dim, init_dim, hidden_dim, num_layers = 10, mask_ratio = 0.3):
        super(GraphMAE, self).__init__()
        self.mlp_x = nn.Linear(input_dim, init_dim)
        self.mlp_pos = nn.Linear(2, init_dim)
        self.encoder = GraphTrans(init_dim, hidden_dim, input_dim, 4, num_layers)
        self.decoder_x = GIN(input_dim, hidden_dim, input_dim, 1)
        self.decoder_pos = GIN(input_dim, hidden_dim, 2, 1)
        self.mask_ratio = mask_ratio
        self.alpha = nn.Parameter(torch.tensor(1.0))  

    def mask_features(self, x, gene = True):
        if(gene):
            mask = (torch.rand(x.size()) < self.mask_ratio).to(args.device)
        else:
            mask = torch.rand(x.size(0)) < self.mask_ratio
        masked_x = x.clone()
        if(gene):
            masked_x = x * (1 - mask.long())
        else:
            masked_x[torch.where(mask.long())[0]] = 0
        return masked_x.to(args.device), mask.long().to(args.device)

    def forward(self, x, pos, edge_index, cell_type, batch):
        masked_x, mask_x = self.mask_features(x)
        masked_pos, mask_pos = self.mask_features(pos, False)
        _x = self.mlp_x(masked_x)
        _pos = self.mlp_pos(masked_pos)
        latent_x = self.encoder(_x, edge_index, batch)
        latent_pos = self.encoder(_pos, edge_index, batch)
        contrastive_loss = contrastive_loss_cell_single_view(cell_type, latent_pos, latent_x, 2)

        latent_x = latent_x.masked_fill(mask_x.bool(), 0)
        latent_pos = latent_pos.masked_fill(mask_pos.unsqueeze(-1).bool(), 0)

        reconstructed_x = self.decoder_x(latent_x, edge_index, batch)
        reconstructed_pos = self.decoder_pos(latent_pos, edge_index, batch)
        recon_loss = mae_loss_cell(pos, x, reconstructed_pos, reconstructed_x, mask_pos.view(len(mask_pos),1), mask_x)
        orthogonal_loss = 0.1*(latent_x.T@latent_x - torch.eye(latent_x.shape[1]).to(args.device)).square().mean() + 0.1*(latent_pos.T@latent_pos - torch.eye(latent_pos.shape[1]).to(args.device)).square().mean()
        loss = F.sigmoid(self.alpha) * contrastive_loss + (1 - F.sigmoid(self.alpha)) * recon_loss + orthogonal_loss

        return loss

def train(model, data_loader, val_loader, optimizer, device):# -> Any:
    model.train()
    for epoch in (range(args.num_epochs)):
            total_loss = 0
            print(f"Training Epoch {epoch+1}")
            for data in tqdm(data_loader):
                data = data.to(device)
                optimizer.zero_grad()
                loss = model(data.X, data.pos, data.edge_index, data.cell_type, data.batch)
                loss.backward(retain_graph=True)
                optimizer.step()
                total_loss += loss.item()
            print(f"Validating Epoch {epoch+1}")
            val_loss = test(model, val_loader, device)
            print(f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {total_loss:.4f}, Validation loss = {val_loss:.4f}")
    return model

def test(model, graphs, device):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        for data in tqdm(graphs):
            data = data.to(device)
            loss = model(data.X, data.pos, data.edge_index, data.cell_type, data.batch)
            total_loss += loss.item()
    return total_loss

if __name__ == '__main__':
    print(args)
    print("Loading graphs")
    graphs_list = [torch.load(file) for file in tqdm(sorted(glob("data/pretraining/*/*")))]
    cell_graphs = []
    print("Preprocessing datasets")
    for i in tqdm(range(len(graphs_list))):
        if(args.data_name == "sea" and i==29):
            continue
        graphs = graphs_list[i]
        X = []
        for k in range(1, len(graphs)):
            X.append(graphs[k].X.squeeze(1).tolist())
        cell_graph = graphs[0]
        cell_graph.pos = cell_graph.X.float()
        cell_graph.X = torch.Tensor(X).float()
        if(cell_graph.X.shape[1]>180):
            cell_graph.X = cell_graph.X[:, :180]
        elif(cell_graph.X.shape[1]!=180):
            continue
        cell_graphs.append(cell_graph)
        del(graphs)
    train_idx, val_idx = train_test_split(np.arange(len(cell_graphs)), test_size = 0.2, random_state = 42)

    train_loader = DataLoader_PyG([cell_graphs[i] for i in train_idx], batch_size=args.batch_size, shuffle=True, exclude_keys=['region_id', 'status', 'acquisition_id_visualizer', 'sample_label_visualizer'])
    val_loader = DataLoader_PyG([cell_graphs[i] for i in val_idx], batch_size=args.batch_size, shuffle=True, exclude_keys=['region_id', 'status', 'acquisition_id_visualizer', 'sample_label_visualizer'])
    model = GraphMAE(cell_graphs[0].X.shape[1], args.hidden_dim, args.latent_dim, args.num_layers).to(args.device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay = args.wd)
    model = train(model, train_loader, val_loader, optimizer, args.device)
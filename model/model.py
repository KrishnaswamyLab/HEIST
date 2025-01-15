import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GINConv
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from utils.augment import GraphAugmentor
import torch
import numpy as np
from model.loss import infoNCE_loss, cross_contrastive_loss

def calculate_pe(high_level_graph, low_level_graphs, pe_dim):
    # Step 1: Calculate cell positional encodings (Dist_i)
    num_nodes = high_level_graph.num_nodes
    cell_locations = high_level_graph.X  # Shape: [num_cells, 2]
    anchor_nodes = torch.randint(0, num_nodes, (pe_dim,))
    
    # Compute distance vectors (Dist_i) between each cell and the anchors
    dist_matrix = torch.cdist(cell_locations, cell_locations[anchor_nodes])  # Shape: [num_cells, num_cells]

    # Use the distance matrix as positional encoding for the high-level graph
    high_level_graph.pe = dist_matrix

    # Step 2: Calculate gene positional encodings (RankNorm * Dist_i)
    gene_expressions = low_level_graphs.X.squeeze(-1)  # Shape: [num_genes]
    gene_batches = low_level_graphs.batch  # Shape: [num_genes]

    # Filter by batch and calculate RankNorm for each gene batch
    rank_norm = torch.zeros_like(gene_expressions).to(high_level_graph.X.device).float()
    unique_batches = gene_batches.unique()
    for b in unique_batches:
        batch_mask = (gene_batches == b)
        batch_gene_expressions = gene_expressions[batch_mask]
        batch_ranks = torch.argsort(-batch_gene_expressions, dim=0)  # Descending order
        batch_rank_norm = torch.linspace(0, 1, steps=batch_ranks.size(0), device=batch_gene_expressions.device)
        rank_norm[torch.where(batch_mask)[0]] = batch_rank_norm[torch.argsort(batch_ranks)]

    # Apply sinusoidal encoding to the gene positional encodings
    div_term = torch.exp(torch.arange(0, pe_dim, 2, device=rank_norm.device).float() * -(torch.log(torch.tensor(10000.0)) / pe_dim))
    gene_sinusoidal_pe = torch.zeros(rank_norm.size(0), pe_dim, device=rank_norm.device)
    gene_sinusoidal_pe[:, 0::2] = torch.sin(rank_norm.unsqueeze(-1) * div_term)
    gene_sinusoidal_pe[:, 1::2] = torch.cos(rank_norm.unsqueeze(-1) * div_term)
    # Set low-level graph PE
    low_level_graphs.pe = gene_sinusoidal_pe

    return high_level_graph, low_level_graphs

class HierarchicalBlending(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 n_heads: int = 1, 
                 p: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.p = p
        
        self.attn_high = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_low  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, high_emb, low_emb, batch):
        device = high_emb.device

        num_feats = high_emb.shape[1]
        perm = torch.randperm(num_feats, device=device)
        half = num_feats // 2
        anchor1, anchor2 = perm[:half], perm[half:]

        high_emb1 = high_emb.clone()
        high_emb1[:, anchor1] = 0
        high_emb2 = high_emb.clone()
        high_emb2[:, anchor2] = 0

        low_mask = torch.rand(low_emb.shape[0], device=device) < self.p
        low_emb1 = low_emb.clone()
        low_emb1[low_mask] = 0
        low_emb2 = low_emb.clone()
        low_emb2[~low_mask] = 0

        global_low_emb2 = global_mean_pool(low_emb2, batch)

        Q_high = high_emb1.unsqueeze(1)         # [B, 1, d_model]
        K_high = global_low_emb2.unsqueeze(1)   # [B, 1, d_model]
        V_high = global_low_emb2.unsqueeze(1)   # [B, 1, d_model]

        attn_high_out, _ = self.attn_high(Q_high, K_high, V_high)
        attn_high_out = attn_high_out.squeeze(1)

        Q_low = low_emb1.unsqueeze(1)         # [N, 1, d_model]
        K_low = high_emb2[batch].unsqueeze(1) # [N, 1, d_model]
        V_low = high_emb2[batch].unsqueeze(1) # [N, 1, d_model]

        attn_low_out, _ = self.attn_low(Q_low, K_low, V_low)
        attn_low_out = attn_low_out.squeeze(1)  # => [N, d_model]

        return attn_high_out, attn_low_out

# def hierarchical_blending(high_emb, low_emb, batch):# -> tuple:# -> tuple:
#     # Number of features in `high_emb`:
#     num_feats = high_emb.shape[1]

#     perm = torch.randperm(num_feats, device=high_emb.device)
#     half = num_feats // 2
#     anchor1, anchor2 = perm[:half], perm[half:]

#     high_emb1 = high_emb.clone()
#     high_emb1[:, anchor1] = 0
#     high_emb2 = high_emb.clone()
#     high_emb2[:, anchor2] = 0

#     low_mask = torch.rand(low_emb.shape[0], device=low_emb.device) < 0.5
#     low_emb1 = low_emb.clone()
#     low_emb1[low_mask] = 0
#     low_emb2 = low_emb.clone()
#     low_emb2[~low_mask] = 0

#     high_emb = high_emb1 + global_mean_pool(low_emb2, batch)
#     low_emb = low_emb1 + high_emb2[batch]

#     return high_emb, low_emb
    
class MultiLevelGraphLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiLevelGraphLayer, self).__init__()
        self.conv = TransformerConv(input_dim, output_dim // num_heads, heads=num_heads)
        self.batch_norm = nn.LayerNorm(output_dim)

        self.attn_fclh = nn.Linear(output_dim * 2, 1, bias=False)
        self.attn_fchl = nn.Linear(output_dim * 2, 1, bias=False)

    def forward(self, high_emb, high_level_graph, low_emb, low_level_graphs):
        high_emb = F.relu(self.batch_norm(self.conv(high_emb, high_level_graph.edge_index)))
        low_emb = F.relu(self.batch_norm(self.conv(low_emb, low_level_graphs.edge_index)))

        _high_emb = high_emb#.clone().requires_grad_()
        x = global_mean_pool(low_emb, low_level_graphs.batch)
        sim_score = torch.sigmoid(self.attn_fclh(torch.cat([_high_emb, x], dim=1)))
        _high_emb = sim_score * _high_emb + (1 - sim_score) * x
        
        updated_low_emb = low_emb.clone().requires_grad_()
        for i in range(low_level_graphs.batch.max().item() + 1):
            mask = (low_level_graphs.batch == i)
            low_emb_i = low_emb[mask]
            high_emb_i = high_emb[i].unsqueeze(0)

            attn_scores_low = torch.sigmoid(self.attn_fchl(torch.cat([high_emb_i.expand_as(low_emb_i), low_emb_i], dim=1)))

            updated_low_emb[mask] = attn_scores_low * low_emb_i + (1 - attn_scores_low) * high_emb_i

        return _high_emb, updated_low_emb


class GraphEncoder(nn.Module):
    def __init__(self, input_dim_high, input_dim_low, pe_dim, init_dim, hidden_dim, output_dim):
        super(GraphEncoder, self).__init__()
        
        # self.augmentor = GraphAugmentor(0.2, 0.0)
        self.mlp_high = nn.Sequential(nn.Linear(pe_dim, init_dim), nn.ReLU())
        self.mlp_low = nn.Sequential(nn.Linear(pe_dim, init_dim), nn.ReLU())
        
        # self.pe_mlp = nn.Sequential(nn.Linear(10, init_dim), nn.ReLU())

        # self.mlp_high = nn.Sequential(nn.Linear(10, init_dim), nn.ReLU())
        # self.mlp_low = nn.Sequential(nn.Linear(10, init_dim), nn.ReLU())
        
        self.conv1 = MultiLevelGraphLayer(init_dim, hidden_dim, 4)
        self.conv2 = MultiLevelGraphLayer(hidden_dim, hidden_dim, 4)
        self.conv3 = MultiLevelGraphLayer(hidden_dim, output_dim, 4)

        self.projection_head = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU())
        self.beta = nn.Parameter(torch.tensor(0.5))  

        self.hierarchical_blending = HierarchicalBlending(pe_dim, 4)
    def forward(self, high_level_graph, low_level_graphs, pe_dim, high_mask=None, low_mask=None):
        # high_emb = self.mlp_high(high_level_graph.X.float() + high_level_graph.pe.float())
        # low_emb = self.mlp_low(low_level_graphs.X.float() + low_level_graphs.pe.float())
        # import pdb; pdb.set_trace()
        high_level_graph, low_level_graphs = calculate_pe(high_level_graph, low_level_graphs, pe_dim)
        high_emb = high_level_graph.X.repeat([1,pe_dim//2]).float() + high_level_graph.pe.float()
        low_level_graphs.pe = F.sigmoid(self.beta) * low_level_graphs.pe.float() + (1 - F.sigmoid(self.beta)) * high_level_graph.pe[low_level_graphs.batch].float()
        low_emb = low_level_graphs.X.float() + low_level_graphs.pe.float()
        high_emb, low_emb = self.hierarchical_blending(high_emb, low_emb, low_level_graphs.batch)
        high_emb = self.mlp_high(high_emb)
        low_emb = self.mlp_low(low_emb)
        
        
        if high_mask is not None and low_mask is not None:
            high_emb = high_emb * high_mask
            low_emb = low_emb * low_mask

        high_emb1, low_emb1 = self.conv1(high_emb, high_level_graph, low_emb, low_level_graphs)
        high_emb2, low_emb2 = self.conv2(high_emb1, high_level_graph, low_emb1, low_level_graphs)

        high_emb2 += high_emb1
        low_emb2 += low_emb1

        high_emb3, low_emb3 = self.conv3(high_emb2, high_level_graph, low_emb2, low_level_graphs)
        high_emb3 += high_emb2
        low_emb3 += low_emb2

        return self.projection_head(high_emb3), self.projection_head(low_emb3)

    def encode(self, high_level_graph, low_level_graphs, pe_dim):
        # Encode high-level graph
            
        # high_emb = self.mlp_high(high_level_graph.X.float() + high_level_graph.pe.float())
        # low_emb = self.mlp_low(low_level_graphs.X.float() + low_level_graphs.pe.float())

        high_level_graph, low_level_graphs = calculate_pe(high_level_graph, low_level_graphs, pe_dim)
        high_emb = high_level_graph.X.repeat([1,pe_dim//2]).float() + high_level_graph.pe.float()
        low_level_graphs.pe = F.sigmoid(self.beta) * low_level_graphs.pe.float() + (1 - F.sigmoid(self.beta)) * high_level_graph.pe[low_level_graphs.batch].float()
        low_emb = low_level_graphs.X.float() + low_level_graphs.pe.float()
        high_emb, low_emb = self.hierarchical_blending(high_emb, low_emb, low_level_graphs.batch)
        high_emb = self.mlp_high(high_emb)
        low_emb = self.mlp_low(low_emb)

        high_emb1, low_emb1 = self.conv1(high_emb, high_level_graph, low_emb, low_level_graphs)
        high_emb2, low_emb2 = self.conv2(high_emb1, high_level_graph, low_emb1, low_level_graphs)

        high_emb2 += high_emb1
        low_emb2 += low_emb1

        high_emb3, low_emb3 = self.conv3(high_emb2, high_level_graph, low_emb2, low_level_graphs)
        high_emb3 += high_emb2
        low_emb3 += low_emb2
        return high_emb3, low_emb3
    
    def forward_contrastive(self, high_level_graph, low_level_graphs):
        aug_high_graph_1, aug_high_graph_2 = self.augmentor.augment(high_level_graph)
        aug_low_graphs_1, aug_low_graphs_2 = self.augmentor.augment_batch(low_level_graphs)
        
        high_emb_1, low_emb_1 = self.forward(aug_high_graph_1, aug_low_graphs_1)
        high_emb_2, low_emb_2 = self.forward(aug_high_graph_2, aug_low_graphs_2)
        
        low_emb_1_pooled = global_mean_pool(low_emb_1, aug_low_graphs_1.batch)
        low_emb_2_pooled = global_mean_pool(low_emb_2, aug_low_graphs_2.batch)
        loss = infoNCE_loss(high_emb_1, high_emb_2) + 0.5 * infoNCE_loss(low_emb_1, low_emb_2)
        loss += infoNCE_loss(high_emb_1, low_emb_1_pooled) + infoNCE_loss(high_emb_2, low_emb_2_pooled)
        
        return loss

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.sf = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm1d(input_dim)
        if(num_layers==1):
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for i in range(num_layers-2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, X):
        X = self.bn(X)
        for i in range(len(self.layers)-1):
            X = F.relu(self.layers[i](X))
        return self.layers[-1](X)
   
class GIN_decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super(GIN_decoder, self).__init__()
        self.layers = nn.ModuleList([GINConv(nn.Linear(input_dim, hidden_dim), train_eps=True)])
        for i in range(num_layers-1):
            self.layers.append(GINConv(nn.Linear(hidden_dim, hidden_dim), train_eps=True))
        self.high_mlp = nn.Linear(hidden_dim, 2)
        self.low_mlp = nn.Linear(hidden_dim, 1)
        self.alpha = nn.Parameter(torch.tensor(1.0))  
        self.beta = nn.Parameter(torch.tensor(0.3))  

    def forward(self, high_emb, high_graph, low_emb, low_graph):
        for i in range(len(self.layers)):
            high_emb, low_emb = self.layers[i](high_emb, high_graph.edge_index), self.layers[i](low_emb, low_graph.edge_index)
            high_emb, low_emb = high_emb.relu(), low_emb.relu()
        high_emb = F.dropout(high_emb, p=0.5, training=self.training)
        low_emb = F.dropout(low_emb, p=0.5, training=self.training)
        high_emb = self.high_mlp(high_emb)
        low_emb = self.high_mlp(low_emb)
        return high_emb, low_emb
 
class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList([GINConv(nn.Linear(input_dim, hidden_dim), train_eps=True)])
        for i in range(num_layers-2):
            self.layers.append(GINConv(nn.Linear(hidden_dim, hidden_dim), train_eps=True))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.batch_norm = nn.BatchNorm1d(input_dim)

    def forward(self, x, edge_index, batch):
        x = self.batch_norm(x)
        for i in range(len(self.layers)-1):
            x = self.layers[i](x, edge_index)
            x = x.relu()
        # x = global_mean_pool(x, batch)  
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layers[-1](x)
        return x

class GraphTrans(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super(GraphTrans, self).__init__()
        self.layers = nn.ModuleList([TransformerConv(input_dim, hidden_dim // num_heads, heads=num_heads)])
        for i in range(num_layers-2):
            self.layers.append(TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.batch_norm = nn.BatchNorm1d(input_dim)

    def forward(self, x, edge_index, batch):
        x = self.batch_norm(x)
        for i in range(len(self.layers)-1):
            x = self.layers[i](x, edge_index)
            x = x.relu()
        # x = global_mean_pool(x, batch)  
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layers[-1](x)
        return x
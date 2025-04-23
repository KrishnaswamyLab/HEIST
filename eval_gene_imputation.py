import warnings
warnings.filterwarnings('ignore')

import torch
from utils.dataloader import create_dataloader
from model.model import GraphEncoder, GIN_decoder
import torch.optim as optim
from torch_geometric.data import DataLoader as DataLoader_PyG
import numpy as np
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import math
import json
import pandas as pd
from tqdm import tqdm
from torch_geometric.nn.pool import global_add_pool, global_mean_pool
from torchinfo import summary
import sys
from glob import glob
from argparse import ArgumentParser
import lovely_tensors as lt
lt.monkey_patch()
parser = ArgumentParser(description="SCGFM")
parser.add_argument('--data_name', type=str, default = 'gsm', help="Directory where the raw data is stored")
parser.add_argument('--pe_dim', type=int, default= 128, help="Dimension of the positional encodings")
parser.add_argument('--init_dim', type=int, default= 128, help="Hidden dim for the MLP")
parser.add_argument('--hidden_dim', type=int, default= 128, help="Hidden dim for the MLP")
parser.add_argument('--output_dim', type=int, default= 128, help="Output dim for the MLP")
parser.add_argument('--blending', action='store_true')
parser.add_argument('--fine_tune', action='store_true')
parser.add_argument('--anchor_pe', action='store_true')
parser.add_argument('--pe', action='store_true')
parser.add_argument('--cross_message_passing', action='store_true')
parser.add_argument('--num_layers', type=int, default= 10, help="Number of MLP layers")
parser.add_argument('--num_heads', type=int, default= 8, help="Number of transformer heads")
parser.add_argument('--batch_size', type=int, default= 512, help="Batch size")
parser.add_argument('--graph_idx', type=int, default= 0, help="Batch size")
parser.add_argument('--lr', type=float, default= 1e-3, help="Learnign Rate")
parser.add_argument('--wd', type=float, default= 3e-3, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default= 20, help="Number of epochs")
parser.add_argument('--gpu', type=int, default= 0, help="GPU index")
args = parser.parse_args()





if __name__ == '__main__':
    
    graphs_names = glob(f"data/{args.data_name}_preprocessed/*")[:1]
    
    model_path = f"saved_models/final_model_sea_pe_concat_softmax.pth"
    checkpoint = torch.load(model_path)
    fine_tune = args.fine_tune
    args = checkpoint['args']
    args.fine_tune = fine_tune

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    args.batch_size = 50
    model = GraphEncoder(args.pe_dim, args.init_dim, args.hidden_dim, args.output_dim, 
                            args.num_layers, args.num_heads, args.cross_message_passing, args.pe, args.anchor_pe, args.blending).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.wd)
    model.load_state_dict(checkpoint['model_state_dict'])
    decoder = GIN_decoder(args.output_dim, args.output_dim).to(args.device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model = model.eval()
    decoder = decoder.eval()
    summary(model)
    criterion = torch.nn.L1Loss()
    print("Extracting representations:")
    model.eval()
    _high_level_graphs = []
    total_loss = 0
    for i in tqdm(range(len(graphs_names))):
        graphs = torch.load(graphs_names[i], weights_only=False)
        dataloader = create_dataloader(graphs, args.batch_size, False)
        with torch.no_grad():
            for high_level_subgraph, low_level_batch, batch_idx in dataloader:
                high_level_subgraph = high_level_subgraph.to(args.device)
                low_level_batch = low_level_batch.to(args.device)
                low_level_batch.batch_idx = batch_idx.to(args.device)
                mask = torch.bernoulli(torch.ones(low_level_batch.num_nodes, 1)*0.3).long().to(args.device)
                
                high_emb, low_emb = model.encode(high_level_subgraph, low_level_batch, 1 - mask)
                _, imputed_gene, _ = decoder(high_emb, high_level_subgraph, low_emb, low_level_batch)
                import pdb; pdb.set_trace()
                total_loss+=criterion(imputed_gene*mask, low_level_batch.X*mask)
    print(total_loss)

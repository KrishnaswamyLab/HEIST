import warnings
warnings.filterwarnings('ignore')

import torch
import random
from utils.dataloader import create_dataloader
from model.model import GraphEncoder, GIN_decoder#, infoNCE_loss, cca_loss, cross_contrastive_loss
from model.loss import contrastive_loss_cell, mae_loss_cell, infoNCE_loss
import torch.optim as optim
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from tqdm import tqdm
from torch_geometric.nn.pool import global_add_pool
from torchinfo import summary
from glob import glob
from argparse import ArgumentParser
import psutil
import sys
import gc
gc.enable()
import logging

def print_ram_usage():
    process = psutil.Process()  # Get current process
    memory_info = process.memory_info()  # Get memory information
    ram_usage_mb = memory_info.rss / (1024 * 1024 * 1024)  # Convert bytes to MB
    print(f"RAM Usage: {ram_usage_mb:.2f} GB")

parser = ArgumentParser(description="SCGFM")
parser.add_argument('--data_dir', type=str, default = 'data/sea_preprocessed_new_new/', help="Directory where the raw data is stored")
parser.add_argument('--pe_dim', type=int, default= 128, help="Dimension of the positional encodings")
parser.add_argument('--init_dim', type=int, default= 256, help="Hidden dim for the MLP")
parser.add_argument('--hidden_dim', type=int, default= 512, help="Hidden dim for the MLP")
parser.add_argument('--output_dim', type=int, default= 512, help="Output dim for the MLP")
parser.add_argument('--num_layers', type=int, default= 3, help="Number of MLP layers")
parser.add_argument('--batch_size', type=int, default= 150, help="Batch size")
parser.add_argument('--graph_idx', type=int, default= 0, help="Batch size")
parser.add_argument('--lr', type=float, default= 1e-3, help="Learnign Rate")
parser.add_argument('--wd', type=float, default= 3e-3, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default= 20, help="Number of epochs")
parser.add_argument('--gpu', type=int, default= 0, help="GPU index")

def initialize_weights(layer):
    # Handle linear layers with Xavier initialization
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    # Handle TransformerConv layers
    elif isinstance(layer, pyg_nn.TransformerConv):
        # Loop over the named parameters and initialize them if possible
        for name, param in layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    # Handle Conv2d layers (if applicable)
    elif isinstance(layer, nn.Conv2d):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)



args = parser.parse_args()
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

INPUT_DIM_HIGH = 2
INPUT_DIM_LOW = 1

if __name__ == '__main__':
    graphs_list = [torch.load(file) for file in glob(args.data_dir+"*")]
    model_path = "saved_models/model_3M_attention_anchor_pe_ranknorm_cross_blending.pth"
    print(args)
    model = GraphEncoder(INPUT_DIM_HIGH, INPUT_DIM_LOW, args.pe_dim, args.init_dim, args.hidden_dim, args.output_dim).to(args.device)
    model.apply(initialize_weights)
    decoder = GIN_decoder(args.output_dim, args.output_dim).to(args.device)
    decoder.apply(initialize_weights)
    optimizer = optim.AdamW(list(model.parameters())+list(decoder.parameters()), lr=args.lr, weight_decay = args.wd)
    summary(model)
    model.train()
    for epoch in range(args.num_epochs):
        total_loss = 0
        order = list(range(len(graphs_list)))
        random.shuffle(order)

        for graph_idx in order:
            graphs = graphs_list[graph_idx]
            try:
                dataloader = create_dataloader(graphs, args.batch_size, args.device)
            except KeyError:
                continue

            for high_level_subgraph, low_level_batch, batch_idx in tqdm(dataloader):
                optimizer.zero_grad()
                # Move data to device only when needed
                high_level_subgraph = high_level_subgraph.to(args.device) # batch_size \times 2
                low_level_batch = low_level_batch.to(args.device) # batch_size * num_genes \times 1
                low_level_batch.batch_idx = batch_idx.to(args.device)

                high_mask = 1 - torch.bernoulli(torch.ones(high_level_subgraph.num_nodes, 1)*0.3).long().to(args.device)
                low_mask = 1 - torch.bernoulli(torch.ones(low_level_batch.num_nodes, 1)*0.3).long().to(args.device)

                high_emb, low_emb = model(high_level_subgraph, low_level_batch, args.pe_dim, high_mask, low_mask)
                contrastive_loss = contrastive_loss_cell(low_level_batch.cell_type, high_emb, low_level_batch, low_emb, 2)
                _high_emb = high_emb * high_mask
                _low_emb = low_emb * low_mask

                decoded_high, decoded_low = decoder(_high_emb, high_level_subgraph, _low_emb, low_level_batch)
                recon_loss = mae_loss_cell(high_level_subgraph.X, low_level_batch.X, decoded_high, decoded_low, 1 - high_mask, 1 - low_mask)
                loss = F.sigmoid(decoder.alpha) * contrastive_loss + (1 - F.sigmoid(decoder.alpha)) * recon_loss
                # loss = model.forward_contrastive(high_level_subgraph, low_level_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                del(high_level_subgraph, low_level_batch, loss, high_emb, low_emb, contrastive_loss, recon_loss, high_mask, low_mask, _high_emb, _low_emb)
                torch.cuda.empty_cache()
                gc.collect()
            del(dataloader)
            gc.collect()
        print(f"Epoch: {epoch + 1}, Loss: {total_loss}")

        torch.save({
            'epoch': epoch,  # Save the current epoch number
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,  # Optionally, save the loss
            'args': args
        }, model_path)
        print(f"Model saved at Epoch: {epoch+1}")
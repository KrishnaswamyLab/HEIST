import warnings
warnings.filterwarnings('ignore')

from utils.dataloader import CustomDataset
import torch
from sklearn.model_selection import StratifiedKFold, KFold
from model.model import MLP, GIN
from model.loss import AUCPRHingeLoss,aucpr_hinge_loss
import torch.optim as optim
from torch.nn import CrossEntropyLoss, BCELoss
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as DataLoader_PyG
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import math
from glob import glob
import json
from tqdm import tqdm
from torch_geometric.nn.pool import global_add_pool
from argparse import ArgumentParser

parser = ArgumentParser(description="SCGFM")
parser.add_argument('--data_name', type=str, help="Name of the dataset")
parser.add_argument('--random_state', type=int, default=0, help="Random state")
parser.add_argument('--label_name', type=str, help="Name of the label")
parser.add_argument('--init_dim', type=int, default=256, help="Hidden dim for the MLP")
parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dim for the MLP")
parser.add_argument('--output_dim', type=int, default=512, help="Output dim for the MLP")
parser.add_argument('--num_layers', type=int, default=3, help="Number of MLP layers")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning Rate")
parser.add_argument('--wd', type=float, default=1e-4, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default=1000, help="Number of epochs")
parser.add_argument('--gpu', type=int, default=0, help="GPU index")
parser.add_argument('--model_type', type=str, default='GIN', choices=['MLP', 'GIN'], help="Model type to train (MLP or GIN)")


def eval(model, loader):
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for X,y in loader:
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            total_correct += torch.sum(preds == y).float()
            total += len(y)
    return ((total_correct * 100) / total).item()

def eval_roc_auc(model, loader):
    model.eval()
    true = []
    pred = []
    with torch.no_grad():
        for X,y in loader:
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            true.append(y)
            pred.append(preds)
    true = torch.cat(true).cpu().detach().numpy()
    pred = torch.cat(pred).cpu().detach().numpy()
    return roc_auc_score(true, pred,average='weighted')

def train(model, train_loader, val_loader, test_loader):
    best_val_roc = eval_roc_auc(model, val_loader)
    best_test_acc = eval(model, test_loader)
    best_roc_auc = eval_roc_auc(model, test_loader)
    loss_fn = AUCPRHingeLoss()
    loss_fn = CrossEntropyLoss(weight = class_weights)
    # loss_fn = aucpr_hinge_loss
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=1000, verbose=True)

    with tqdm(range(args.num_epochs)) as tq:
        for e, epoch in enumerate(tq):
            model.train()
            for X,y in train_loader:
                opt.zero_grad()
                logits = model(X)
                loss = loss_fn(logits, y) #+ loss_fn2(logits, graphs[args.label_name].long())
                loss.backward()
                opt.step()
            scheduler.step(eval(model, val_loader))

            train_acc = eval(model, train_loader)
            train_roc = eval_roc_auc(model, train_loader)
            val_acc = eval(model, val_loader)
            val_roc = eval_roc_auc(model, val_loader)
            if val_roc>= best_val_roc:
                best_val_roc = val_roc
                best_test_acc = eval(model, test_loader)
                best_roc_auc = max(best_roc_auc, eval_roc_auc(model, test_loader))
            tq.set_description("Loss = %.4f, Train acc = %.4f, Train roc = %.4f, Val acc = %.4f, Val roc = %.4f, Best val roc = %.4f, Best acc = %.4f, Best ROC AUC score = %.4f" % (loss.item(), train_acc, train_roc, val_acc, val_roc, best_val_roc, best_test_acc, best_roc_auc))
    return best_test_acc, best_roc_auc

args = parser.parse_args()
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'
INPUT_DIM_HIGH = 2
INPUT_DIM_LOW = 1

if __name__ == '__main__':
    print(args)
    print("Loading graphs")
    # _high_level_graphs = torch.load('data/space-gm/{}_graphs_projection.pt'.format(args.data_name))
    _high_level_graphs = torch.load('data/space-gm/{}_graphs_3M_attention_anchor_pe_ranknorm_cross_blending.pt'.format(args.data_name))
    if(args.data_name == "charville"):
        split = [["c004"], ["c003"]]
    elif(args.data_name == "upmc"):
        split = [["c006", "c007"], ["c004", "c005"]]

    indices = []
    for i in range(len(_high_level_graphs)):
        if hasattr(_high_level_graphs[i], args.label_name) and not math.isnan(getattr(_high_level_graphs[i], args.label_name)):
            if(torch.any(torch.isnan(_high_level_graphs[i].X))):
                print(i)
                continue
            indices.append(i)
    _high_level_graphs = [_high_level_graphs[i] for i in indices]
    patient_c = [_high_level_graph.region_id.split("_")[1] for _high_level_graph in _high_level_graphs]
    labels = torch.LongTensor([getattr(i, args.label_name) for i in _high_level_graphs]).to(args.device)
    print(torch.bincount(labels))
    embeddings = []
    for graph in _high_level_graphs:
        embeddings.append(graph.X.mean(0).tolist())  # Pooling for MLP input
    embeddings = torch.FloatTensor(embeddings).to(args.device)
    embeddings[:, embeddings.shape[1]//2:] = embeddings[:, embeddings.shape[1]//2:]/(len(_high_level_graphs*40))
    class_weights = torch.tensor([(1 - labels.float().mean()), labels.float().mean()]).to(args.device)
    
    roc_scores = []

    for fold in range(2):
        model = MLP(2*args.output_dim, args.hidden_dim, 2, args.num_layers).to(args.device)
        train_idx = []
        val_idx = []
        for i in range(len(patient_c)):
            if(patient_c[i] not in split[fold]):
                train_idx.append(i)
            else:
                val_idx.append(i)
        val_idx = np.array(val_idx)
        val_idx, test_idx = train_test_split(val_idx, test_size=0.5, stratify=labels.cpu().detach().numpy()[val_idx])
        train_idx = torch.LongTensor(train_idx).to(args.device)
        val_idx = torch.LongTensor(val_idx).to(args.device)
        test_idx = torch.LongTensor(test_idx).to(args.device)
        print(f"Num samples: {len(labels)}")
        train_loader = DataLoader(CustomDataset(embeddings[train_idx], labels[train_idx]), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(CustomDataset(embeddings[val_idx], labels[val_idx]), batch_size=args.batch_size, shuffle=False)#, exclude_keys=['cell_type', 'region_id', 'status', 'acquisition_id_visualizer', 'sample_label_visualizer'])
        test_loader = DataLoader(CustomDataset(embeddings[test_idx], labels[test_idx]), batch_size=args.batch_size, shuffle=False)#x, exclude_keys=['cell_type', 'region_id', 'status', 'acquisition_id_visualizer', 'sample_label_visualizer'])

        best_acc, best_aoc_roc = train(model, train_loader, val_loader, test_loader)
        roc_scores.append(best_aoc_roc.item())
    roc_scores = np.array(roc_scores)
    print(f"Mean:{roc_scores.mean()}, Std:{roc_scores.std()}.")
    print(f"Max:{roc_scores.max()}.")

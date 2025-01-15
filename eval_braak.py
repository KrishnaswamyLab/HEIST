import warnings
warnings.filterwarnings('ignore')

import torch
from model.model import MLP, GIN
from sklearn.model_selection import StratifiedKFold, KFold
from model.loss import AUCPRHingeLoss, aucpr_hinge_loss
import torch.optim as optim
from torch.nn import BCELoss
from torch_geometric.data import DataLoader as DataLoader_PyG
from utils.dataloader import CustomDataset
from torch.nn import CrossEntropyLoss, BCELoss
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, LogisticRegression
import math
import json
from tqdm import tqdm
from torch_geometric.nn.pool import global_add_pool
from argparse import ArgumentParser

parser = ArgumentParser(description="SCGFM")
parser.add_argument('--model', type=str, default="sea_graphs_cross_contrastive_anchor_pe_ranknorm.pt", help="Which model to choose")
parser.add_argument('--init_dim', type=int, default=256, help="Hidden dim for the MLP")
parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dim for the MLP")
parser.add_argument('--output_dim', type=int, default=512, help="Output dim for the MLP")
parser.add_argument('--num_layers', type=int, default=3, help="Number of MLP layers")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning Rate")
parser.add_argument('--wd', type=float, default=1e-3, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs")
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

def eval_f1(model, loader):
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
    return f1_score(true, pred, average="macro")

def train(model, train_loader, val_loader, test_loader):
    best_val_acc = eval(model, val_loader)
    best_test_acc = eval(model, test_loader)
    best_test_f1 = eval_f1(model, test_loader)
    loss_fn = CrossEntropyLoss()
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
            val_acc = eval(model, val_loader)
            if val_acc>= best_val_acc:
                best_val_acc = val_acc
                best_test_acc = max(best_test_acc, eval(model, test_loader))
                best_test_f1 = max(best_test_acc, eval_f1(model, test_loader))
            tq.set_description("Loss = %.4f, Train acc = %.4f, Val acc = %.4f, Best acc = %.4f, Best f1 = %.4f" % (loss.item(), train_acc, val_acc, best_test_acc, best_test_f1))
    return best_test_acc, best_test_f1


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
    _high_level_graphs = torch.load("data/"+args.model)
    labels = [i.y for i in _high_level_graphs]
    labels = np.hstack(labels)
    X = torch.cat([i.X for i in _high_level_graphs], dim=0)
    y = torch.LongTensor(LabelEncoder().fit_transform(labels)).to(args.device)
    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y.cpu().numpy())
    test_idx, val_idx = train_test_split(val_idx, test_size=0.5, stratify=y[val_idx].cpu().numpy())
    train_loader = DataLoader(CustomDataset(X[train_idx], y[train_idx]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CustomDataset(X[val_idx], y[val_idx]), batch_size=args.batch_size, shuffle=False)#, exclude_keys=['cell_type', 'region_id', 'status', 'acquisition_id_visualizer', 'sample_label_visualizer'])
    test_loader = DataLoader(CustomDataset(X[test_idx], y[test_idx]), batch_size=args.batch_size, shuffle=False)#x, exclude_keys=['cell_type', 'region_id', 'status', 'acquisition_id_visualizer', 'sample_label_visualizer'])

    acc = []
    f1 = []
    for i in range(5):
        model = MLP(2*args.output_dim, 2*args.hidden_dim, len(torch.unique(y)), args.num_layers).to(args.device)
        best_acc, best_f1 = train(model, train_loader, val_loader, test_loader)
        acc.append(best_acc)
        f1.append(best_f1)
    acc = np.array(acc)
    f1 = np.array(f1)
    print("F1")
    print(f"Mean:{f1.mean()}, Std:{f1.std()}.")
    print("Acc")
    print(f"Mean:{acc.mean()}, Std:{acc.std()}.")
    # class_weights = torch.tensor([(1 - labels.float().mean()), labels.float().mean()]).to(args.device)
    # train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.2, random_state = args.random_state)#, stratify=labels.cpu().detach().numpy())
    # val_idx, test_idx = train_test_split(val_idx, test_size=0.5, stratify=labels[val_idx].cpu().detach().numpy())
    # train_idx = torch.LongTensor(train_idx).to(args.device)
    # val_idx = torch.LongTensor(val_idx).to(args.device)
    # test_idx = torch.LongTensor(test_idx).to(args.device)
    # print(f"Num samples: {len(labels)}")
    # train_loader = DataLoader_PyG([_high_level_graphs[i] for i in train_idx], batch_size=16, shuffle=True, exclude_keys=['cell_type', 'region_id', 'status', 'acquisition_id_visualizer', 'sample_label_visualizer'])
    # val_loader = DataLoader_PyG([_high_level_graphs[i] for i in val_idx], batch_size=16, shuffle=False, exclude_keys=['cell_type', 'region_id', 'status', 'acquisition_id_visualizer', 'sample_label_visualizer'])
    # test_loader = DataLoader_PyG([_high_level_graphs[i] for i in test_idx], batch_size=16, shuffle=False, exclude_keys=['cell_type', 'region_id', 'status', 'acquisition_id_visualizer', 'sample_label_visualizer'])

    # best_acc, best_aoc_roc = train(model, train_loader, val_loader, test_loader)
    # print(f"Best acc:{best_acc}, Best aoc roc:{best_aoc_roc}.")
    # args.low_level_pool = 'sum'
    # args.high_level_pool = 'sum'
    # args.best_acc = best_acc
    # args.best_aoc_roc = best_aoc_roc
    # args_dict = vars(args)

    # with open("data/{}/{}_best.json".format(args.data_name, args.label_name), "r") as json_file:
    #     best_args = json.load(json_file)
    # if best_args['best_aoc_roc'] < args_dict['best_aoc_roc']:
    #     print("Better accuracy.")
    #     args_json = json.dumps(args_dict, indent=4)
    #     with open("data/space-gm/{}_{}_best.json".format(args.data_name, args.label_name), "w") as json_file:
    #         json_file.write(args_json)

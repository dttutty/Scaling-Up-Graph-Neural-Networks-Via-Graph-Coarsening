import argparse
import torch
import torch.nn.functional as F
from networks import Net
from torch import tensor
from torch.optim import Adam
from utils import load_data, coarsening
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--coarsening_ratio', type=float, default=0.5)
parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')
args = parser.parse_args()
path = "params/"
if not os.path.isdir(path):
    os.mkdir(path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.num_features, args.num_classes, candidate, C_list, Gc_list = coarsening(args.dataset, 1-args.coarsening_ratio, args.coarsening_method)
model = Net(args).to(device)



def train(args, path, model, test_acces, data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, optimizer, best_val_loss, val_losses):
    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        model.train()
        optimizer.zero_grad()
        out = model(coarsen_features, coarsen_edge)
        loss = F.nll_loss(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        pred = model(coarsen_features, coarsen_edge)
        val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')

        val_losses.append(val_loss)
        if args.early_stopping > 0 and epoch > args.epochs // 2:
            tmp = tensor(val_losses[-(args.early_stopping + 1):-1])
            if val_loss > tmp.mean().item():
                break

    model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
    model.eval()
    pred = model(data.x, data.edge_index).max(1)[1]
    test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
    print(test_acc)
    test_acces.append(test_acc)

for _ in range(args.runs):

    data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge = \
        map(lambda x: x.to(device), load_data(args.dataset, candidate, C_list, Gc_list, args.experiment))

    if args.normalize_features:
        coarsen_features = F.normalize(coarsen_features, p=1)
        data.x = F.normalize(data.x, p=1)

    model.reset_parameters()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float('inf')
    val_losses = []
    test_acces = []
    train(args, path, model, test_acces, data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge, optimizer, best_val_loss, val_losses)
    
    #plot all_acc 
    plt.figure()
    plt.plot(test_acces)
    plt.savefig('all_acc.png')


print('ave_acc: {:.4f}'.format(np.mean(test_acces)), '+/- {:.4f}'.format(np.std(test_acces)))


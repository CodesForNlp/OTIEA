import os
import argparse
import itertools

import apex
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelbackup import FTDEA
from DataProcess import DataProcess
from loss_l import L1_Loss
from utils import add_inverse_rels, get_train_batch, get_hits, get_hits_stable

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--data", default="data/DBP15K")
    parser.add_argument("--lang", default="zh_en")
    parser.add_argument("--rate", type=float, default=0.3)

    parser.add_argument("--r_hidden", type=int, default=100)
    parser.add_argument("--t_hidden", type=int, default=100)

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=3)  # margin based loss

    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--neg_epoch", type=int, default=10)
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--stable_test", action="store_true", default=False)
    args = parser.parse_args()
    return args
def init_data(args, device):
    data = DataProcess(args.data, args.lang, rate=args.rate)[0]
    data.x1 = F.normalize(data.x1, dim=1, p=2).to(device).requires_grad_()  # 实体glove的embedding
    data.x2 = F.normalize(data.x2, dim=1, p=2).to(device).requires_grad_()
    data.edge_index_all1, data.rel_all1 = add_inverse_rels(data.edge_index1,
                                                           data.rel1)  # edge_index表示h和t组合，rel表示所有不重复关系
    data.edge_index_all2, data.rel_all2 = add_inverse_rels(data.edge_index2, data.rel2)
    data.rel_size1=torch.arange(data.rel1.size(0)).view(1,-1)
    data.rel_size2 = torch.arange(data.rel2.size(0)).view(1,-1)
    return data  # 包含属性：两个KG的embedding表示；两个KG的所有关系索引；两个KG的所有边索引
def get_emb(model, data):
    model.eval()
    with torch.no_grad():
        x1, r1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_size1)
        x2, r2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_size2)
    return x1, x2
def train(model, criterion, optimizer, data, train_batch):
    model.train()
    x1, r1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_size1)
    x2, r2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_size2)
    loss = criterion(x1, x2, data.train_set, train_batch)
    optimizer.zero_grad()
    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
    return loss
def test(model, data, stable=False):
    x1, x2 = get_emb(model, data)
    print('-' * 16 + 'Train_set' + '-' * 16)
    get_hits(x1, x2, data.train_set)
    print('-' * 16 + 'Test_set' + '-' * 17)
    get_hits(x1, x2, data.test_set)
    if stable:
        get_hits_stable(x1, x2, data.test_set)
    print()
    return x1, x2
def main(args):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    data = init_data(args, device).to(device)  # 包含属性：两个KG的embedding表示；两个KG的所有关系索引；两个KG的所有边索引
    model = FTDEA(data.x1.size(1), args.r_hidden, args.t_hidden).to(device)  # 输入embedding维度大小300
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), iter([data.x1, data.x2])))
    # model, optimizer = apex.amp.initialize(model, optimizer)
    # criterion = L1_Loss(args.gamma)
    # pairs = data.pair_set
    # pair_set = pairs
    # pair_set = pair_set[:, torch.randperm(pair_set.size(1))]
    # train_set = pair_set[:, :int(args.rate * pair_set.size(1))]
    # data.train_set = train_set.t()
    # test_set = pair_set[:, int(args.rate * pair_set.size(1)):]
    # data.test_set = test_set.t()
    model, optimizer = apex.amp.initialize(model, optimizer)
    criterion = L1_Loss(args.gamma)
    for epoch in range(args.epoch):
        if epoch % args.neg_epoch == 0:
            x1, x2 = get_emb(model, data)
            train_batch, train_batch1, train_batch2 = get_train_batch(x1, x2, data.train_set, data.edge_index1, data.edge_index2, data.trans_index1, data.trans_index2, args.k)
        loss = train(model, criterion, optimizer, data, train_batch)
        print('Epoch:', epoch + 1, '/', args.epoch, '\tLoss: %.3f' % loss, '\r', end='')
        if (epoch + 1) % args.test_epoch == 0:
            print()
            test(model, data, args.stable_test)
if __name__ == '__main__':
    args = parse_args()
    if args.lang == 'zh_en':
        data1 = DataProcess(args.data, args.lang, rate=args.rate)[0]
    main(args)

import os
import json
import torch
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import sort_edge_index
from utils_train import add_inverse_rels

class DataProcess(InMemoryDataset):
    def __init__(self, root, pair, KG_num=1, rate=0.3, seed=1):   #seed不要变
        self.pair = pair
        self.KG_num = KG_num
        self.rate = rate
        self.seed = seed
        torch.manual_seed(seed)
        super(DataProcess, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):  # 存放原始数据的路径root/
        return ['zh_en', 'fr_en', 'ja_en']

    @property
    def processed_file_names(self):  # 存放处理后的数据，一般是pt格式
        return '%s_%d_%.2f_%d.pt' % (self.pair, self.KG_num, self.rate, self.seed)

    def process(self):
        x1_path = os.path.join(self.root, self.pair, 'ent_ids_1')
        x2_path = os.path.join(self.root, self.pair, 'ent_ids_2')
        g1_path = os.path.join(self.root, self.pair, 'triples_1')
        g2_path = os.path.join(self.root, self.pair, 'triples_2')
        emb_path = os.path.join(self.root, self.pair, 'vectorList.json')
        x1, edge_index1, rel1, assoc1 = self.process_graph(g1_path, x1_path, emb_path)
        x2, edge_index2, rel2, assoc2 = self.process_graph(g2_path, x2_path, emb_path)

        pair_path = os.path.join(self.root, self.pair, 'ref_ent_ids')
        pair_set1 = self.process_pair(pair_path, assoc1, assoc2)
        pair_set = pair_set1[:, torch.randperm(pair_set1.size(1))]
        train_set = pair_set[:, :int(self.rate * pair_set.size(1))]
        test_set = pair_set[:, int(self.rate * pair_set.size(1)):]
        edge_index_all1, rel_all1 = add_inverse_rels(edge_index1,
                                                               rel1)  # edge_index表示h和t组合，rel表示所有不重复关系
        edge_index_all2, rel_all2 = add_inverse_rels(edge_index2, rel2)
        trans_index1 = torch.randperm(rel1.size(0))[:int(self.rate * rel1.size(0))]
        trans_index2 = torch.randperm(rel2.size(0))[:int(self.rate * rel2.size(0))]
        rel_size1 = torch.arange(rel1.size(0)).view(1, -1)
        rel_size2 = torch.arange(rel2.size(0)).view(1, -1)
        if self.KG_num == 1:
            data = Data(x1=x1, edge_index1=edge_index1, rel1=rel1,
                        x2=x2, edge_index2=edge_index2, rel2=rel2,
                        rel_size1=rel_size1, rel_size2=rel_size2,
                        edge_index_all1=edge_index_all1, edge_index_all2=edge_index_all2,
                        rel_all1=rel_all1, rel_all2=rel_all2,
                        trans_index1=trans_index1, trans_index2=trans_index2,
                        pair_set=pair_set1,
                        train_set=train_set.t(), test_set=test_set.t())
        else:
            x = torch.cat([x1, x2], dim=0)
            edge_index = torch.cat([edge_index1, edge_index2 + x1.size(0)], dim=1)
            rel = torch.cat([rel1, rel2 + rel1.max() + 1], dim=0)
            data = Data(x=x, edge_index=edge_index, rel=rel, train_set=train_set.t(), test_set=test_set.t())
        torch.save(self.collate([data]), self.processed_paths[0])

    def process_graph(self, triple_path, ent_path, emb_path):
        g = read_txt_array(triple_path, sep='\t', dtype=torch.long)
        subj, rel, obj = g.t()

        assoc = torch.full((rel.max().item() + 1,), -1, dtype=torch.long)
        assoc[rel.unique()] = torch.arange(rel.unique().size(0))
        rel = assoc[rel]

        idx = []
        with open(ent_path, 'r') as f:
            for line in f:
                info = line.strip().split('\t')
                idx.append(int(info[0]))
        idx = torch.tensor(idx)
        with open(emb_path, 'r', encoding='utf-8') as f:
            embedding_list = torch.tensor(json.load(f))
        x = embedding_list[idx]

        assoc = torch.full((idx.max().item() + 1,), -1, dtype=torch.long)
        assoc[idx] = torch.arange(idx.size(0))
        subj, obj = assoc[subj], assoc[obj]
        edge_index = torch.stack([subj, obj], dim=0)  # 所有三元组的边
        edge_index, rel = sort_edge_index(edge_index, rel)
        return x, edge_index, rel, assoc  # x表示所有实体对应的embedding,edge_index由h和t组成的二维数组，rel是三元组中所有的关系，assoc表示arrange(实体个数)

    def process_pair(self, path, assoc1, assoc2):
        e1, e2 = read_txt_array(path, sep='\t', dtype=torch.long).t()
        return torch.stack([assoc1[e1], assoc2[e2]], dim=0)

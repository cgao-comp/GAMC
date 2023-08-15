import os.path as osp
import pickle
from collections import namedtuple, Counter
import numpy as np

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected, degree
from torch_geometric.io import read_txt_array
from torch_sparse import coalesce
import scipy.sparse as sp

from ogb.nodeproppred import PygNodePropPredDataset

from sklearn.preprocessing import StandardScaler


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_dataset(dataset_name):
    if dataset_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root="./data")
        graph = dataset[0]
        num_nodes = graph.x.shape[0]
        graph.edge_index = to_undirected(graph.edge_index)
        graph.edge_index = remove_self_loops(graph.edge_index)[0]
        graph.edge_index = add_self_loops(graph.edge_index)[0]
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)
        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.train_mask, graph.val_mask, graph.test_mask = train_mask, val_mask, test_mask
        graph.y = graph.y.view(-1)
        graph.x = scale_feats(graph.x)
    else:
        dataset = Planetoid("", dataset_name, transform=T.NormalizeFeatures())
        graph = dataset[0]
        graph.edge_index = remove_self_loops(graph.edge_index)[0]
        graph.edge_index = add_self_loops(graph.edge_index)[0]

    num_features = dataset.num_features
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)


def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()   # 小写字母转为大写字母
    dataset = TUDataset(root="./data", name=dataset_name)
    dataset = list(dataset)
    graph = dataset[0]


    if graph.x == None:
        if graph.y and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g in dataset:
                feature_dim = max(feature_dim, int(g.y.max().item()))
            
            feature_dim += 1
            for i, g in enumerate(dataset):
                node_label = g.y.view(-1)
                feat = F.one_hot(node_label, num_classes=int(feature_dim)).float()
                dataset[i].x = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g in dataset:
                feature_dim = max(feature_dim, degree(g.edge_index[0]).max().item())
                degrees.extend(degree(g.edge_index[0]).tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for i, g in enumerate(dataset):
                degrees = degree(g.edge_index[0])
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                degrees = torch.Tensor([int(x) for x in degrees.numpy().tolist()])
                feat = F.one_hot(degrees.to(torch.long), num_classes=int(feature_dim)).float()
                g.x = feat
                dataset[i] = g

    else:
        print("******** Use `attr` as node features ********")
    feature_dim = int(graph.num_features)

    labels = torch.tensor([x.y for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    for i, g in enumerate(dataset):
        dataset[i].edge_index = remove_self_loops(dataset[i].edge_index)[0]
        dataset[i].edge_index = add_self_loops(dataset[i].edge_index)[0]
    #dataset = [(g, g.y) for g in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")
    return dataset, (feature_dim, num_classes)


def read_file(folder, name, dtype=None):
    path = osp.join(folder, '{}.txt'.format(name))
    return read_txt_array(path, sep=',', dtype=dtype)


def split(data, batch):
    """
    PyG util code to create graph batches
    """

    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.news_node is not None:
        slices['news_node'] = node_slice

    return data, slices

def read_graph_data(folder, feature):
    """
    PyG util code to create PyG data instance from raw graph data
    """

    node_attributes = sp.load_npz(folder + f'new_{feature}_feature.npz')
    edge_index = read_file(folder, 'A', torch.long).t()
    node_graph_id = np.load(folder + 'node_graph_id.npy')
    graph_labels = np.load(folder + 'graph_labels.npy')


    edge_attr = None
    x = torch.from_numpy(node_attributes.todense()).to(torch.float)
    node_graph_id = torch.from_numpy(node_graph_id).to(torch.long)
    i = 0
    news_node = []
    for id in node_graph_id:
        if i == id:
            i = i + 1
            news_node.append(1)
        else:
            news_node.append(0)
    news_node = torch.from_numpy(np.array(news_node)).to(torch.long)
    y = torch.from_numpy(graph_labels).to(torch.long)
    _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, news_node=news_node)
    data, slices = split(data, node_graph_id)

    return data, slices


class FNNDataset(InMemoryDataset):
    r"""
        The Graph datasets built upon FakeNewsNet data

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root, name, feature='spacy', empty=False, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.root = root
        self.feature = feature
        super(FNNDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if not empty:
            self.data, self.slices, self.train_idx, self.val_idx, self.test_idx = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        name = 'raw/'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed/'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1)

    @property
    def raw_file_names(self):
        names = ['node_graph_id', 'graph_labels']
        return ['{}.npy'.format(name) for name in names]

    @property
    def processed_file_names(self):
        if self.pre_filter is None:
            return f'{self.name[:3]}_data_{self.feature}.pt'
        else:
            return f'{self.name[:3]}_data_{self.feature}_prefiler.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. No download allowed')

    def process(self):

        self.data, self.slices = read_graph_data(self.raw_dir, self.feature)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        # The fixed data split for benchmarking evaluation
        # train-val-test split is 20%-10%-70%
        self.train_idx = torch.from_numpy(np.load(self.raw_dir + 'train_idx.npy')).to(torch.long)
        self.val_idx = torch.from_numpy(np.load(self.raw_dir + 'val_idx.npy')).to(torch.long)
        self.test_idx = torch.from_numpy(np.load(self.raw_dir + 'test_idx.npy')).to(torch.long)

        torch.save((self.data, self.slices, self.train_idx, self.val_idx, self.test_idx), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


def load_fake_news_graph_dataset(dataset_name, feature, deg4feat=False):
    dataset = FNNDataset(root='./data', feature=feature, empty=False, name=dataset_name)
    # dataset = list(dataset)
    graph = dataset[0]
    feature_dim = int(graph.num_features)
    labels = torch.tensor([x.y for x in dataset])

    num_classes = torch.max(labels).item() + 1
    # for i, g in enumerate(dataset):
    #     dataset[i].edge_index = remove_self_loops(dataset[i].edge_index)[0]
    #     dataset[i].edge_index = add_self_loops(dataset[i].edge_index)[0]
    # dataset = [(g, g.y) for g in dataset]
    # node = np.load(osp.join('./data', dataset_name, 'raw/node_graph_id.npy'))
    # id = pickle.load(open(osp.join('./data', dataset_name, 'pol_id_twitter_mapping.pkl'), 'rb'))

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")
    return dataset, (feature_dim, num_classes)
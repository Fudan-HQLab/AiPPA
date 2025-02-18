from typing import Any
import os
import pickle
from torch.utils.data import Dataset
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import (RandomFlip,
                                        RandomRotate,
                                        RandomTranslate,
                                        Compose,
                                        Center)


class PairData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class GNNDataset(Dataset):
    def __init__(self, root, phase):
        """Initialization"""
        total = sorted(os.listdir(root))
        self.root = root
        self.data = total
        self.phase = phase
        np.random.seed(16)

        total_size = len(total)
        permu = np.random.permutation(total_size)
        if phase == 'train':
            self.list_IDs = permu[:int(total_size * 0.9)]
        elif phase == 'val':
            self.list_IDs = permu[int(total_size * 0.9):]
        elif phase == 'test':
            self.list_IDs = permu
        else:
            raise ValueError('wrong phase!')
        self.transform = Compose([RandomFlip(0),
                                  RandomFlip(1),
                                  RandomFlip(2),
                                  RandomRotate(360, 0),
                                  RandomRotate(360, 1),
                                  RandomRotate(360, 2)])
                                  # Center()])
        self.trans = Center()

    def __len__(self):
        """Denotes the total number of samples"""
        return self.list_IDs.shape[0]

    def __getitem__(self, index):
        """Generates one sample of data"""
        index = self.list_IDs[index]
        path = self.data[index]
        pdb = path.split('.')[0]
        path = os.path.join(self.root, path)
        with open(path, 'rb') as f:
            x_s, edge_features_s, edge_index_s, \
            x_t, edge_features_t, edge_index_t, y = pickle.load(f)

        data_s_pos = Data(pos=x_s[:, 2:5])
        data_t_pos = Data(pos=x_t[:, 2:5])
        if self.phase == 'train':
            # random flip&rotate
            self.transform(data_s_pos)
            self.transform(data_t_pos)

        
        self.trans(data_s_pos)
        self.trans(data_t_pos)

        data = PairData(x_s=x_s,
                        edge_features_s=edge_features_s,
                        edge_index_s=edge_index_s,
                        pos_s=data_s_pos.pos,
                        x_t=x_t,
                        edge_features_t=edge_features_t,
                        edge_index_t=edge_index_t,
                        pos_t=data_t_pos.pos,
                        y=y,
                        name=pdb.split('.')[0])

        return data


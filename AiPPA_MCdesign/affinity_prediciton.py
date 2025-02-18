import argparse
import warnings
from typing import Any
from torch_geometric.data import Data
from AiPPA import BA
from torch_geometric.transforms import Center
from Bio.PDB import SASA, PDBParser
import numpy as np
from scipy.spatial.distance import cdist
import torch
import pickle


class PairData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class Features:
    def __init__(self, struct):
        self.struct = struct

    def get_res_features(self, res):
        code_map = {'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'GLU': 'E',
                    'GLN': 'Q', 'ASP': 'D', 'ASN': 'N', 'HIS': 'H',
                    'TRP': 'W', 'PHE': 'F', 'TYR': 'Y', 'ARG': 'R',
                    'LYS': 'K', 'SER': 'S', 'THR': 'T', 'MET': 'M',
                    'ALA': 'A', 'GLY': 'G', 'PRO': 'P', 'CYS': 'C'}

        res_name = code_map[res.get_resname()]
        aa_type_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        aa_type = aa_type_list.index(res_name)

        return [aa_type]

    def structure2graph(self, edge_threshold=8):
        pos = []
        res_tag = []
        x = []
        edge_attr = []
        edge_index = []
        # sasa
        sr = SASA.ShrakeRupley()
        sr.compute(self.struct, level='R')

        for resi in self.struct.get_residues():
            feature = self.get_res_features(resi)
            resi_pos = resi['CA'].get_coord().tolist()
            pos.append(resi_pos)
            res_tag.append(resi)
            sasa = round(resi.sasa, 3)
            x.append(feature + [sasa] + resi_pos)
        x = torch.tensor(x, dtype=torch.float)

        dis_matrix = cdist(pos, pos)
        inter_mol_idx = np.where(dis_matrix < edge_threshold)
        # print(inter_mol_idx)
        idx = [(i, j) for i, j in zip(inter_mol_idx[0], inter_mol_idx[1])
               if i < j]
        for i, j in idx:
            edge_token = 0
            if res_tag[i].get_parent() != res_tag[j].get_parent():
                edge_token = 1
            edge_feature = [edge_token] + [dis_matrix[i, j]]
            edge_attr.append(edge_feature)
            edge_index.append((i, j))
            edge_attr.append(edge_feature)
            edge_index.append((j, i))
        edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

        return x, edge_attr, edge_index


class CalBA:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = BA(self.device, layers=2)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load('data/model_trained.pth'))

    def affinity(self, receptor_file, ligand_file):
        # time1 = time.time()
        warnings.filterwarnings('ignore')
        parser = PDBParser()
        # rep = parser.get_structure('', receptor_file)
        lig = parser.get_structure('', ligand_file)

        # calculate target protein
        with open(receptor_file, 'rb') as f:
            x_s, edge_features_s, edge_index_s = pickle.load(f)
        # x_s, edge_features_s, edge_index_s \
        #     = Features(rep).structure2graph(edge_threshold=8)

        x_t, edge_features_t, edge_index_t \
            = Features(lig).structure2graph(edge_threshold=8)

        edge_features_s_batch = torch.tensor([0] * edge_features_s.shape[0],
                                             dtype=torch.int64)
        edge_features_t_batch = torch.tensor([0] * edge_features_t.shape[0],
                                             dtype=torch.int64)
        data_s_pos = Data(pos=x_s[:, 2:5])
        data_t_pos = Data(pos=x_t[:, 2:5])
        # print(x_s[0])
        Center()(data_s_pos)
        Center()(data_t_pos)
        data = PairData(x_s=x_s,
                        edge_features_s=edge_features_s,
                        edge_index_s=edge_index_s,
                        x_t=x_t,
                        edge_features_t=edge_features_t,
                        edge_index_t=edge_index_t,
                        pos_s=data_s_pos.pos,
                        pos_t=data_t_pos.pos,
                        edge_features_s_batch=edge_features_s_batch,
                        edge_features_t_batch=edge_features_t_batch)
        data = Center()(data)
        data = data.to(self.device)

        self.model.eval()
        with torch.no_grad():
            prediction_ba = self.model(data)
        ba = prediction_ba.detach().item()
        # time2 = time.time()
        # print(time2 - time1)
        return round(ba, 3)


if __name__ == '__main__':
    a = CalBA().affinity('TL1A.pkl', 'designedNb.pdb')
    print(a)

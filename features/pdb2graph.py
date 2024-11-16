from Bio.PDB import Structure, SASA, PDBParser
import numpy as np
from Bio import PDB
from scipy.spatial.distance import cdist
import torch


def get_res_features(res: PDB.Residue.Residue):
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


def structure2graph(struct: Structure.Structure, edge_threshold):
    pos = []
    res_tag = []
    x = []
    edge_attr = []
    edge_index = []
    # sasa
    sr = SASA.ShrakeRupley()
    sr.compute(struct, level='R')

    for resi in struct.get_residues():
        feature = get_res_features(resi)
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


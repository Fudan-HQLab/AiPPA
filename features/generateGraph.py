import argparse
import warnings
import torch
import pickle
from Bio.PDB import *
from pdb2graph import structure2graph
import os

edge_threshold = [8.0]


def features(data_dir, target_path, target_file):
    n_train = 0
    m_train = 0
    pdb_ba = {}
    parser = PDBParser()
    can_not_process_complex = []
    with open(target_file, 'r') as f:
        for line in f:
            if line != '' and line[0] != '#':
                info = line.strip().split()
                pdb_ba[info[0]] = float(info[1])

    for pdb, ba in pdb_ba.items():
        n_train += 1
        try:
            rep_path = os.path.join(data_dir, pdb,
                                    f'{pdb}_proteinA.pdb')
            rep = parser.get_structure('', rep_path)
        except Exception as e:
            can_not_process_complex.append(pdb)
            m_train += 1
            print(f'{pdb}_proteinA error:{e}')
            continue
        if rep is None:
            can_not_process_complex.append(pdb)
            m_train += 1
            print(f'{pdb}_proteinA error: proteinA is None')
            continue

        try:
            lig_path = os.path.join(data_dir, pdb,
                                    f'{pdb}_proteinB.pdb')
            lig = parser.get_structure('', lig_path)
        except Exception as e:
            can_not_process_complex.append(pdb)
            m_train += 1
            print(f'{pdb}_proteinB error:{e}')
            continue
        if lig is None:
            can_not_process_complex.append(pdb)
            m_train += 1
            print(f'{pdb}_proteinB error: proteinB is None')
            continue
        # outer

        if not os.path.exists(target_path):
            os.mkdir(target_path)

        for thr in edge_threshold:
            inner_target_path = os.path.join(target_path,
                                             f'edge_thr_{int(thr)}')
            # inner
            if not os.path.exists(inner_target_path):
                os.mkdir(inner_target_path)
            try:
                x_s, edge_features_s, edge_index_s \
                    = structure2graph(struct=rep,
                                      edge_threshold=thr)
                x_t, edge_features_t, edge_index_t \
                    = structure2graph(struct=lig,
                                      edge_threshold=thr)
                y = torch.tensor(float(ba), dtype=torch.float)

                with open(os.path.join(inner_target_path,
                                       f'{pdb}.pkl'), 'wb') as f:
                    pickle.dump((x_s, edge_features_s, edge_index_s,
                                 x_t, edge_features_t, edge_index_t,
                                 y), f)
            except Exception as e:
                can_not_process_complex.append(pdb)
                m_train += 1
                print(f'{pdb}_features error: {e}')
                continue
        # print('all data is ', m_train, '/', n_train)
    print(can_not_process_complex)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help=f'data_dir')
    parser.add_argument('--target_path', type=str, required=True,
                        help=f'target_path')
    parser.add_argument('--target_file', type=str, required=True,
                        help=f'target_file')
    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = args_parse()
    features(args.data_dir,
             args.target_path,
             args.target_file)

# running
# python generateGraph.py --data_dir ../data/pdbs \
#                         --target_path ./pkls \
#                         --target_file ../data/affinity.txt

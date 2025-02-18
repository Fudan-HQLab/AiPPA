import os
import pickle
import time
import warnings

from Bio import SeqIO

from affinity_prediciton import Features, CalBA
import numpy as np
# from abnumber import Chain
import abnumber
from Bio.PDB import *
from igfold.refine.pyrosetta_ref import init_pyrosetta
from mc_simulation import MCsimulation


class MCMC:
    def __init__(self, ini_struct, target, ini_fa=None):
        self.nb = ini_struct
        if ini_fa:
            records = str(list(SeqIO.parse(ini_fa, 'fasta'))[0].seq)
            self.ini_fa = records
        else:
            self.ini_fa = self.struct2fasta(self.nb)
        self.target = target

    def struct2fasta(self, struct):
        parser = PDBParser()
        seq = ''
        aa_codes = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
                    'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                    'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W'
                    }
        for resi in parser.get_structure('', struct).get_residues():
            seq += aa_codes[resi.get_resname()]
        return seq

    def cdr_region(self):
        chain = abnumber.Chain(self.ini_fa, scheme='chothia')
        fr1 = chain.fr1_seq
        fr2 = chain.fr2_seq
        fr3 = chain.fr3_seq
        fr4 = chain.fr4_seq
        cdr1 = chain.cdr1_seq
        cdr2 = chain.cdr2_seq
        cdr3 = chain.cdr3_seq
        cdr_list = [cdr1, cdr2, cdr3]
        fr_list = [fr1, fr2, fr3, fr4]
        return cdr_list, fr_list

    def run(self, seed, step, log_out, pdbs_pos,
            out_seq, T_factor, terminal,
            cdr3_only,
            mut_point,
            mut_fr, sol_thresh,
            transfer_prob_file=None,
            proportion_file=None,
            ):
        warnings.filterwarnings('ignore')
        np.random.seed(seed)
        time1 = time.time()
        init_pyrosetta()

        if os.path.exists('out'):
            os.system(f'rm -r out')
        os.system(f'mkdir out')
        if os.path.exists(os.path.join('out', pdbs_pos)):
            os.system(f'rm -r out/{pdbs_pos}')
        os.system(f'mkdir out/{pdbs_pos}')
        # if os.path.exists(log_out):
        #     os.system(f'rm {log_out}')
        # if os.path.exists(out_seq):
        #     os.system(f'rm {out_seq}')

        parser = PDBParser()
        rep = parser.get_structure('', self.target)
        x_s, edge_features_s, edge_index_s \
            = Features(rep).structure2graph(edge_threshold=8)
        with open('TL1A.pkl', 'wb') as f:
            pickle.dump((x_s, edge_features_s, edge_index_s), f)

        # print(records)
        # print(self.cdr_region())
        cdr_list, fr_list = self.cdr_region()
        initial_ba = CalBA().affinity('TL1A.pkl', self.nb)
        start_info = {'ba': initial_ba,
                      'rep': 'TL1A.pkl',
                      'cdr_list': cdr_list,
                      'fr_list': fr_list,
                      'init_struct': self.nb}
        sim = MCsimulation(T_factor, start_info)
        sim.Simulation(terminal_ba=terminal,
                       cycle=step,
                       log=os.path.join('out', log_out),
                       outseq=os.path.join('out', out_seq),
                       pos=os.path.join('out', pdbs_pos),
                       cdr3_only=cdr3_only,
                       mut_point=mut_point,
                       mut_fr=mut_fr,
                       sol_thresh=sol_thresh,
                       transfer_prob_file=transfer_prob_file,
                       proportion_file=proportion_file,
                       )
        time2 = time.time()
        # print(time2 - time1)



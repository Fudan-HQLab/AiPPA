import os.path
import pickle
import random

import numpy as np
from solubility_prediction import cal_sol
from structure_prediction import pred_struct
from affinity_prediciton import CalBA


class FRMutation:
    def __init__(self):
        # property
        fr1_mut_prob = np.load('data/FRMut/fr1_mut.npy')
        fr2_mut_prob = np.load('data/FRMut/fr2_mut.npy')
        fr3_mut_prob = np.load('data/FRMut/fr3_mut.npy')
        fr4_mut_prob = np.load('data/FRMut/fr4_mut.npy')
        self.mut_site_prob = [fr1_mut_prob, fr2_mut_prob,
                              fr3_mut_prob, fr4_mut_prob]

        with open('data/FRMut/fr_mut_site.pkl', 'rb') as f:
            mut = pickle.load(f)
            # [[0, 15],
            #  [0, 2, 4, 11, 12, 14, 16, 17, 18],
            #  [1, 2, 3, 7, 17, 19, 20, 21, 30, 39, 40],
            #  [0, 1, 5]]
        mut_site_length = [len(mut['fr1']), len(mut['fr2']),
                           len(mut['fr3']), len(mut['fr4'])]
        self.mut_fr_site = [mut['fr1'], mut['fr2'], mut['fr3'], mut['fr4']]
        mut_fr_prob = [round(i / sum(mut_site_length), 3)
                       for i in mut_site_length]
        # sum == 1
        self.mut_fr_prob = [mut_fr_prob[0], mut_fr_prob[1],
                            mut_fr_prob[2],
                            1 - mut_fr_prob[0] - mut_fr_prob[1] - mut_fr_prob[2]]
        self.aa_type_list = ['A', 'C', 'D', 'E', 'F',
                             'G', 'H', 'I', 'K', 'L',
                             'M', 'N', 'P', 'Q', 'R',
                             'S', 'T', 'V', 'W', 'Y',
                             ]

    def fr_mut(self, fr_list):
        mut_fr = np.random.choice([0, 1, 2, 3],
                                  p=self.mut_fr_prob)
        mut_site = np.random.choice(self.mut_fr_site[mut_fr])
        mut_aa = np.random.choice(self.aa_type_list,
                                  p=self.mut_site_prob[mut_fr][:, mut_site])
        mut_list = fr_list.copy()
        mut_list[mut_fr] = mut_list[mut_fr][:mut_site] + mut_aa + \
                           mut_list[mut_fr][mut_site + 1:]
        mut_fr_map = dict(zip([0, 1, 2, 3], ['fr1', 'fr2', 'fr3', 'fr4']))
        print(f'{mut_fr_map[mut_fr]}_{mut_site} '
              f'mut: {fr_list[mut_fr][mut_site]} --> {mut_aa}')
        return mut_list


class CDRMutation:
    def __init__(self, cdr3_only,
                 transfer_prob_file=None, proportion_file=None,
                 ):
        self.cdr3_only = cdr3_only

        if transfer_prob_file:
            print('Using individual CDR sequence length transfer probability')
            transfer_prob = transfer_prob_file
        else:
            print('Using CDR sequence length transfer probability calculated with INDI database')
            transfer_prob = 'data/cdr_length_transfer_prob.pkl'
        with open(transfer_prob, 'rb') as f:
            self.cdr3_length_transfer_prob = pickle.load(f)['cdr3']

        if proportion_file:
            print('Using individual a.a proportion')
            proportion = proportion_file
        else:
            print('Using a.a. proportion calculated with INDI database and '
                  'proportion in McMahon library')
            proportion = 'data/nb_cdr_revised_proportion_no_gap.npy'
        self.aa_proportion = np.load(proportion)

    def cdr_format(self, seq, tag):
        if tag == 'cdr2' and len(seq) == 5:
            return seq[:1] + '-' + seq[1:]
        else:
            return seq

    def replace_gap(self, seq):
        return seq.replace('-', '')

    def random_mutation(self, cdr_seq, tag):
        aa_type_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                        '-']
        format_seq = self.cdr_format(cdr_seq, tag=tag)

        if tag == 'cdr3':
            # CDR sequence length transfer
            transfer_prob = self.cdr3_length_transfer_prob[len(cdr_seq)]
            mut_type = list(transfer_prob.keys())
            length_change = np.random.choice(mut_type,
                                             p=[transfer_prob[i] for i in mut_type])

            # Site for insertion or deletion
            if len(cdr_seq) <= 8:
                mut_site = len(cdr_seq) - 2
            else:
                mut_site = np.random.choice(range(6, len(format_seq) - 2))

            if length_change == 'insert':
                insert_aa = np.random.choice(aa_type_list,
                                             p=self.aa_proportion[:, 11 + len(cdr_seq)])
                print(f'insert on {tag}_{mut_site}: {insert_aa}')
                return format_seq[:mut_site] + insert_aa + format_seq[mut_site:]
            
            if length_change == 'delete':
                print(f'delete on {tag}_{mut_site}: {format_seq[mut_site]}')
                return format_seq[:mut_site] + format_seq[mut_site + 1:]
        # Site for mutation 
        mut_site = np.random.choice(range(len(format_seq)))
        cdr_portion = {'cdr1': self.aa_proportion[:, :7],
                       'cdr2': self.aa_proportion[:, 7:13],
                       'cdr3': np.concatenate((self.aa_proportion[:, 13: 11 + len(format_seq)],
                                               self.aa_proportion[:, -2:]),
                                              axis=1)}
        mut_aa = np.random.choice(aa_type_list,
                                  p=cdr_portion[tag][:, mut_site])
        print(f'{tag}_{mut_site} mut: {format_seq[mut_site]} --> {mut_aa}')
        return format_seq[:mut_site] + mut_aa + format_seq[mut_site + 1:]

    def seq_mutation(self, cdr_list):
        pos_encod = {'cdr1': 0,
                     'cdr2': 1,
                     'cdr3': 2}
        # mut_cdr_list = cdr_list.copy()
        mut_cdr_list = cdr_list.copy()
        if self.cdr3_only:
            cdr_tag = 'cdr3'
        else:
            # increase the possibility of mutating cdr3
            # probability determined on the sequence length 
            cdr_len = [len(i) for i in cdr_list]
            # print(cdr_len)
            prob = [round(i / sum(cdr_len), 3) for i in cdr_len]

            p = [prob[0], prob[1], 1 - prob[0] - prob[1]]
            cdr_tag = np.random.choice(['cdr1', 'cdr2', 'cdr3'],
                                       p=p)
        # print(cdr_tag)
        cdr_pos = pos_encod[cdr_tag]
        mut_cdr_list[cdr_pos] = self.random_mutation(mut_cdr_list[cdr_pos],
                                                     tag=cdr_tag)
        cdr_list = [self.replace_gap(i) for i in mut_cdr_list]
        return cdr_list


class MCsimulation:
    def __init__(self, T_factor, start_info):
        self.T_factor = T_factor
        self.start_info = start_info
        self.cdr_list = start_info['cdr_list']
        self.fr_list = start_info['fr_list']

    def Boltzmann(self, ddG): #ddG: kcal/mol
        # k_B = 1.380649 * 1E-23
        R = 8.314  # J/(mol*K)
        T = 298.15 * self.T_factor  # K
        JperKcal = 4185.85  # J/kcal
        p = np.exp(- ddG * JperKcal / (R * T))  # np.exp(- ddG / 0.6 / T_factor)
        return p

    def LD(self, a, b):
        seq1 = ''.join(a)
        seq2 = ''.join(b)
        ld_matrix = np.zeros((len(seq1) + 1, len(seq2) + 1))
        # initialize
        ld_matrix[0] = np.array(list(range(len(seq2) + 1)))
        ld_matrix[:, 0] = np.array(list(range(len(seq1) + 1)))
        # print(ld_matrix)
        # calculate LD
        for i in range(1, len(seq1) + 1):
            for j in range(1, len(seq2) + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    ld_pos = 0
                else:
                    ld_pos = 1
                ld_matrix[i, j] = min(ld_matrix[i - 1, j - 1] + ld_pos,
                                      ld_matrix[i - 1, j] + 1,
                                      ld_matrix[i, j - 1] + 1)

        # print(ld_matrix)
        return int(ld_matrix[-1, -1])

    def assemble_FrCdr(self, fr, cdr):
        full = []
        for i, j in zip(fr[:-1], cdr):
            full.append(i)
            full.append(j)
        return ''.join(full + [fr[-1]])

    def Simulation(self,
                   terminal_ba,
                   cycle,
                   log,
                   outseq,
                   pos,
                   cdr3_only,
                   mut_point,
                   mut_fr,
                   sol_thresh,
                   transfer_prob_file=None,
                   proportion_file=None,
                   ):
        wt_ba = self.start_info['ba']
        rep_pkl = self.start_info['rep']

        step = 1
        cdr_list = self.cdr_list
        fr_list = self.fr_list
        cdr_mutation = CDRMutation(cdr3_only=cdr3_only,
                                   transfer_prob_file=transfer_prob_file,
                                   proportion_file=proportion_file)
        fr_mutation = FRMutation()

        #  mutation
        with open(log, 'a') as f:
            f.write(f'  step     mut_ba   now_ba  accept\n')
            f.write(f'step{str(0).zfill(len(str(cycle)))}'
                    f'  {str(wt_ba).rjust(7)}\n')
        while step <= cycle:
            step_name = f'step{str(step).zfill(len(str(cycle)))}'
            print('***********************************************')
            print(f'{step_name} begins')
            sol = 0
            # one MCMC mutation
            while sol < sol_thresh:
                fr_mut_list = fr_list.copy()
                cdr_mut_list = cdr_list.copy()
                while self.LD(fr_mut_list, fr_list) + \
                        self.LD(cdr_mut_list, cdr_list) < mut_point:
                    # for _ in range(mut_point):

                    if mut_fr == 'true':
                        fr_cdr_select = np.random.choice(['fr', 'cdr'],
                                                         p=[0.3, 0.7])
                        # print(fr_cdr_select)
                    else:
                        fr_cdr_select = 'cdr'
                    if fr_cdr_select == 'fr':
                        fr_mut_list = fr_mutation.fr_mut(fr_mut_list)
                        cdr_mut_list = cdr_mut_list
                    else:
                        fr_mut_list = fr_mut_list
                        cdr_mut_list = cdr_mutation.seq_mutation(cdr_mut_list)

                full_seq = self.assemble_FrCdr(fr=fr_mut_list,
                                               cdr=cdr_mut_list)
                sol = cal_sol(full_seq)
                print(f'mut_seq: {full_seq}\n')
                print(f'seq solubility: {sol}\n')
            # Predicting structure using IgFold

            pred_struct(seq=full_seq,
                        outname=f'{pos}/{step_name}.pdb',
                        init_struct=self.start_info['init_struct'])
            # calculate binding affinity
            mut_ba = CalBA().affinity(receptor_file=rep_pkl,
                                    ligand_file=os.path.join(pos, f'{step_name}.pdb'))
            accept_tag = 0
            # terminate if ddG < terminal_ba
            if mut_ba <= terminal_ba:
                wt_ba = mut_ba
                accept_tag = 1
                # min_ba_step = step
                with open(log, 'a') as f:
                    f.write(f'{step_name}  {str(mut_ba).rjust(7)}  '
                            f'{str(wt_ba).rjust(7)}  {accept_tag}\n')
                with open(outseq, 'a') as f:
                    f.write(f'{full_seq}\n')
                break
            # MC accept
            # f.write(f'{step_name}  {str(wt_ba).rjust(6)}  {str(mut_ba).rjust(6)}\n')
            if mut_ba <= wt_ba or np.random.random() <= self.Boltzmann(mut_ba - wt_ba):
                wt_ba = mut_ba
                cdr_list = cdr_mut_list.copy()
                fr_list = fr_mut_list.copy()
                accept_tag = 1
            with open(outseq, 'a') as f:
                f.write(f'>{step_name}|accept:{accept_tag}\n')
                f.write(f'{full_seq}\n')
            with open(log, 'a') as f:
                f.write(f'{step_name}  {str(mut_ba).rjust(7)}  '
                        f'{str(wt_ba).rjust(7)}  {accept_tag}\n')
            step += 1
        return


if __name__ == '__main__':
    random.seed(42)
    # time1 = time.time()
    # log_out = 'MC.log'
    # init_pyrosetta()
    # if os.path.exists('pdbs'):
    #     os.system(f'rm -r pdbs')
    # os.system(f'mkdir pdbs')
    # if os.path.exists(log_out):
    #     os.system(f'rm {log_out}')
    # min_ba, min_step = MCSimulation(terminal_ba=-15,
    #                                 cycle=3000,
    #                                 log=log_out)
    # print(min_ba, min_step)
    # time2 = time.time()
    # print(time2 - time1)

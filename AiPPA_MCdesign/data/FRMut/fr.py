import pickle

import numpy as np
from abnumber import Chain
from Bio import SeqIO

fr1 = []
fr2 = []
fr3 = []
fr4 = []
for seq in SeqIO.parse('drugs_seq_only.fasta', 'fasta'):
    # print(seq)
    chain = Chain(str(seq.seq), scheme='chothia')
    fr1.append(chain.fr1_seq)
    fr2.append(chain.fr2_seq)
    fr3.append(chain.fr3_seq)
    fr4.append(chain.fr4_seq)

a = [fr1, fr2, fr3, fr4]
result = [[], [], [], []]
for fr_idx, i in enumerate(a):
    for idx, j in enumerate(zip(*i)):
        if len(set(j)) != 1:
            # print(idx)
            result[fr_idx].append(idx)

mut_site = {'fr1': result[0],
            'fr2': result[1],
            'fr3': result[2],
            'fr4': result[3],
            }
with open('fr_mut_site.pkl', 'wb') as f:
    pickle.dump(mut_site, f)
#     print('*********')
# print(result)
# [[0, 15],
#  [0, 2, 4, 11, 12, 14, 16, 17, 18],
#  [1, 2, 3, 7, 17, 19, 20, 21, 30, 39, 40],
#  [0, 1, 5]]

aa_type_list = ['A', 'C', 'D', 'E', 'F',
                'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'P', 'Q', 'R',
                'S', 'T', 'V', 'W', 'Y',
                ]


def generate_fr_mut(fr, npy):
    pos = np.zeros((20, len(fr[0])))
    print(pos.shape)
    for idx, j in enumerate(zip(*fr)):
        prob = [j.count(aa)/4 for aa in aa_type_list]
        pos[:, idx] = prob
    np.save(npy, pos)
    return pos


data1 = generate_fr_mut(fr1, 'fr1_mut.npy')
data2 = generate_fr_mut(fr2, 'fr2_mut.npy')
data3 = generate_fr_mut(fr3, 'fr3_mut.npy')
data4 = generate_fr_mut(fr4, 'fr4_mut.npy')

# check
# for i in range(data.shape[1]):
#     for idx, j in enumerate(data[:, i]):
#         if j != 0 and j != 1:
#             print(i, j, aa_type_list[idx])

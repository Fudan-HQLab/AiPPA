import os


def cal_sol(seq):
    os.chdir('data/protein-sol')
    with open('predicted_seq.fasta', 'w') as f:
        f.write(f'>pred\n')
        f.write(f'{seq}\n')
    os.system('sh multiple_prediction_wrapper_export.sh predicted_seq.fasta')
    with open('seq_prediction.txt', 'r') as f2:
        for line in f2:
            if 'SEQUENCE PREDICTIONS' in line:
                sol = float(line.split(',')[3])
    os.chdir('../..')
    return sol


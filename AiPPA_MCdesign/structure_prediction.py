import os
# from igfold import IgFoldRunner
from silent_IgFoldRunner import IgFoldRunner
from pymol import cmd


def pred_struct(seq, outname, init_struct):
    sequences = {"H": seq}
    igfold = IgFoldRunner()
    igfold.fold(
        outname,  # File name of output PDB file
        sequences=sequences,  # Antibody sequences
        do_refine=True,  # Refine the antibody structure with PyRosetta
        do_renum=False,  # Renumber the residues of the predicted antibody structure according to the Chothia scheme
    )
    cmd.load(init_struct, 'refer_nb')
    cmd.load(outname, 'igfold_nb')
    cmd.align('igfold_nb', 'refer_nb')
    cmd.save(outname, 'igfold_nb')
    cmd.delete('all')


if __name__ == '__main__':
    vhh1 = 'EVQLVESGGGLVQAGGSLSLSCSASGENLSRYHMGWFRQAPGKERELLGAI' \
           'SWSGIQIYYKDSVKGRFTISRDDAKNTIYLQMNRLKPEDTAVYYCAASLLP' \
           'LSDDPGNETYWGQGTQVTVS'
    # pred_struct(vhh1, 'vhh1_pred.pdb')

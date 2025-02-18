import argparse
import time
from mcmc_runner import MCMC
import warnings


parser = argparse.ArgumentParser()
initial_group = parser.add_argument_group(title='initial parameters')
initial_group.add_argument('--target', type=str, required=True,
                           help=f'Target protein structure in PDB format.\n')
initial_group.add_argument('--nb', type=str, required=True,
                           help=f'Initial Nb structure (PDB) for maturation process.\n')
initial_group.add_argument('--seq', type=str,
                           help=f'Initial Nb sequence in FASTA format (optional).\n')

output_group = parser.add_argument_group(title='output parameters')
output_group.add_argument('--outseq', type=str, default='MC_seq_mut.txt',
                          help=f'Output filename for mutated sequences (default: MC_seq_mut.txt)\n'
                               f'Under "out" folder\n')
output_group.add_argument('--log', type=str, default='MC.log',
                          help=f'Log filename for recording MCMC process (default: MC.log).\n'
                               f'Under "out" folder\n')
output_group.add_argument('--outpos', type=str, default='pdbs',
                          help=f'Output directory for MCMC generated PDB files (default: pdbs)\n'
                               f'Under "out" folder\n')

mcmc_group = parser.add_argument_group(title='MCMC running parameters')
mcmc_group.add_argument('--T_factor', type=float, default=0.2,
                        help=f'Temperature factor controlling Monte Carlo acceptance probability (default: 0.2)\n')
mcmc_group.add_argument('--terminal', type=float, default=-18.0,
                        help=f'Termination threshold for binding affinity (ddG in kcal/mol) (default: -18.0)\n')
mcmc_group.add_argument('--step', type=int, default=30000,
                        help=f'Maximum number of MCMC steps to perform (default: 30000)\n')
mcmc_group.add_argument('--seed', type=int, default=16,
                        help=f'Random seed for reproducibility (default: 16)\n')
mcmc_group.add_argument('--mut_point', type=int, default=1,
                        help=f'Number of simultaneous mutations per step (default: 1)\n')
mcmc_group.add_argument('--cdr3-only', action='store',
                        help=f'Restrict mutations to CDR3 region only\n')
mcmc_group.add_argument('--mut_fr', default='false', choices=['false', 'true'],
                        help=f'Enable framework region mutations (default: False)\n')
mcmc_group.add_argument('--sol_thresh', type=float, default=0.45,
                        help=f'Solubility threshold for mutant screening (default: 0.45)\n')

# parameters to tune
mcmc_group.add_argument('--length_transfer_file', default=None,
                        help=f'File defining custom CDR length distributions (optional)\n')
mcmc_group.add_argument('--aa_proportion_file', default=None,
                        help=f'CSV file specifying amino acid proportion (optional)\n')
args = parser.parse_args()


warnings.filterwarnings('ignore')
time1 = time.time()
mcmc = MCMC(ini_struct=args.nb,
            target=args.target,
            ini_fa=args.seq)

mcmc.run(seed=args.seed,
         step=args.step,
         log_out=args.log,
         pdbs_pos=args.outpos,
         out_seq=args.outseq,
         T_factor=args.T_factor,
         terminal=args.terminal,
         cdr3_only=args.cdr3_only,
         mut_point=args.mut_point,
         mut_fr=args.mut_fr,
         sol_thresh=args.sol_thresh,
         transfer_prob_file=None,
         proportion_file=None,
         )
time2 = time.time()
print(time2 - time1)

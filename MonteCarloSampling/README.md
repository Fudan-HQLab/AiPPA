# MonteCarloSampling

MonteCarloSampling contains the code for de novo nanobody design that integrates AiPPA with the thermodynamic Monte Carlo sampling method, as described in our submitted manuscript: "De novo nanobody design using graph neural networks and thermodynamic Monte Carlo sampling."

## Dependencies
We used [IgFold](https://www.nature.com/articles/s41467-023-38063-x) to predict the 3D structure of a nanobody. Please follow the [instructions](https://github.com/Graylab/IgFold) for installation.

## Quick start

Use the default settings as described in our manuscript. To run the code, simply execute `run_command.sh`:
```
sh run_command.sh
```
The outputs will be saved in the `out` folder, organized as follows:
```
.
├── MC.log           
├── MC_seq_mut.txt   
└── pdbs           
    ├── step00001.fasta
    └── step00001.pdb
```
where the `MC.log` file records the binding free energy for each round, while `MC_seq_mut.txt` stores all the mutated sequences.

To read the descriptions of all available parameters, simply run `python runfile.py --help`, and modify them according to your preferences.

## Customize the mutation sites and define the probability for the FRs
We used three clinically approved nanobody-based drugs—Envafolimab, Ozoralizumab, and Caplacizumab—to define the mutation sites and probabilities for the FRs as defaults (note that these were not used in our manucript).

You can define mutation sites and their probabilities according to your needs. The probabilities should be saved as a `ndarrays` with  dimensions `20 × seqLength`, structured as follows:
```
[[0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.   0.   0.   0.   0.   0.   0.   0.   1.   1.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.   0.   0.   0.   0.   0.   0.   1.   0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.75 0.   0.   0.   0.   1.   0.   0.   0.   0.   0.   0.   0.   0.  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   1.   1.   1.   0.   0.   0.   0.  1.   0.75 0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.   0.   0.   1.   0.   0.   0.   0.   0.   0.   1.   0.   0.   0.  0.   0.   0.   1.   0.   1.   0.   0.   0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.   0.25 0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   1.  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.25 0.   1.   0.   0.   0.   0.   0.   0.   0.   0.   0.   1.   0.  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.   0.   0.   0.   1.   0.   0.   0.   0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   1.   0.   0.   0.   0.   0.   0.   0.  0.   0.   1.   0.   0.   0.   1.   0.   0.   0.   1.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.   1.   0.   0.   1.   0.   0.   0.   0.   0.   0.   1.   0.   0.  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]
 [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]]
```
> Note: The amino-acids are ranked according to their one-letter identifiers: ACDEFGHIKLMNPQRSTVWY. 

## Personalized design for CDRs
The definition of mutation amino-acid probabilities at specific CDR positions and the CDR length transition probabilities are described in Methods of our manuscript.

Also, you can customize them freely.

The amino-acid mutation probabilities for CDRs should be organised in the same format as those for FRs mentioned earlier.

For length transition probabilities, you only need to define the probability for each CDR length. You can then generate the corresponding matrix using the script `generate_length_transfer_file.py` located in the `data` folder.

## Authors and article to be cited
* Lei Wang, Xiaoming He, Gaoxing Guo, Xinzhou Qian, and Qiang Huang*. De novo nanobody design using graph neural networks and thermodynamic Monte Carlo sampling. (Submitted)

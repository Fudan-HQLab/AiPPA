# AiPPA
# Description
AiPPA is a GNN model for protein-protein affinity prediction without protein complex structures

## Dependencies
AiPPA was built and tested on Python 3.9 with PyG 2.3. To install all dependencies, directly run:
```
cd AiPPA-main
conda env create -f environment.yml
conda activate AiPPA
```

## Affinity Prediction
### Quick start
Use `AffinityPrediction.py` to predict the affinity between any two proteins.
```
python AffinityPrediction.py -pA proteinA.pdb -pB proteinB.pdb
```

### Prediction on a dataset

#### 1. Data Preparation
The example data files are located in the `data` folder.
Please prepare the necessary data for training or testing. 
For convenience, the `data` folder can be organised as follows.
```
.
├── affinity.txt
└── pdbs
    ├── ComplexName1
    │   ├── ComplexName1_proteinA.pdb
    │   └── ComplexName1_proteinB.pdb
    └── ComplexName2
        ├── ComplexName2_proteinA.pdb
        └── ComplexName2_proteinB.pdb
```

#### 2. pdb2graph
Navigate to the `features` folder, 
and run `generateGraph.py` to convert protein structure files into graphs.
The resulting pickle files will be saved in `features/pkls`:
``` 
python generateGraph --data_dir ../data/pdbs \
                     --target_path ./pkls \
                     --target_file ../data/affinity.txt
```

#### 3. Train or Test
Run `Runfile.py` to train.

Run `BenchmarkTest.py` to test on your own dataset.

# AiPPA_MCdesign
## Description
AiPPA_MCdesign contains the code for de novo nanobody design that integrates AiPPA with the thermodynamic Monte Carlo sampling method, as described in our submitted manuscript: "De novo nanobody design using graph neural networks and thermodynamic Monte Carlo sampling."

All related code is organized within the `AiPPA_MCdesign` folder.

## Dependencies
We used [IgFold](https://www.nature.com/articles/s41467-023-38063-x) to predict the 3D structure of a nanobody. Please follow the [instructions](https://github.com/Graylab/IgFold) for installation.

## Quick start
Navigate to the `AiPPA_MCdesign` folder where the de novo nanobody design code is located:

```
cd AiPPA_MCdesign
```

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
where the `MC.log` file records the binding affinities for each round, while `MC_seq_mut.txt` stores all the mutated sequences.

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

# Authors
* Lei Wang, Xiaoming He, Gaoxing Guo, Xinzhou Qian, and Qiang Huang*. De novo nanobody design using graph neural networks and thermodynamic Monte Carlo sampling. (Submitted)

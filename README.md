# AiPPA

## Description

AiPPA is a GNN model for protein-protein binding free energy prediction without the need for the protein complex structures

## Dependencies
AiPPA was built and tested on Python 3.9 with PyG 2.3. To install all dependencies, directly run:
```
cd    AiPPA-main
conda env create -f environment.yml
conda activate AiPPA
```

## Affinity Prediction
### Quick start
Use `AffinityPrediction.py` to predict the binding free energy between any two proteins.
```
python AffinityPrediction.py -pA proteinA.pdb -pB proteinB.pdb
```

### Prediction on a dataset

#### 1. Data Preparation
The example data files are located in the `data` folder.
Please prepare the necessary data for training or testing. 
For convenience, the `data` folder can be organized as follows.
```
.
├── affinity.txt
└── pdbs
    ├── ComplexName1
    │   ├── ComplexName1_proteinA.pdb
    │   └── ComplexName1_proteinB.pdb
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


## MonteCarloSampling

MonteCarloSampling contains the code for de novo nanobody design that integrates AiPPA with thermodynamic Monte Carlo sampling, as described in our submitted manuscript: "De novo nanobody design using graph neural networks and thermodynamic Monte Carlo sampling."


## Authors and article to be cited 
* Lei Wang, Xiaoming He, Gaoxing Guo, Xinzhou Qian, and Qiang Huang*. De novo nanobody design using graph neural networks and thermodynamic Monte Carlo sampling. (Submitted)


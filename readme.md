# AiPPA
A GNN Model for Protein-Protein Affinity Prediction without Protein Complex
## Dependencies
AiPPA was build and tested on Python 3.9 with PyG 2.3. To install all dependencies, directly run:
```
cd AiPPA-main
conda env create -f environment.yml
conda activate AiPPA
```

## Affinity Prediction
### Quick to use
Use `AffinityPrediction.py` to predict the affinity between any two proteins.
```
python AffinityPrediction.py -pA proteinA.pdb -pB proteinB.pdb
```

### Prediction on a dataset

#### 1. Data Preparation
The example data files are under the `data` folder.
Please prepare the data needed to be trained or tested. 
By the way, for convenience, the `data` folder could be organised like follows.
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
and run `generateGraph.py` to trans protein structure 
files to graphs, the result pickle files will be saved in `features/pkls`;
``` 
python generateGraph --data_dir ../data/pdbs \
                     --target_path ./pkls \
                     --target_file ../data/affinity.txt
```

#### 3. Train or Test
Run `Runfile.py` to train.

Run `BenchmarkTest.py` to test on your own dataset.

# Probabilistic Graph Rewiring via Virtual Nodes

<img src="https://github.com/chendiqian/PR-MPNN/blob/main/main-figure.png" alt="drawing" width="800"/>
<p align="center">
</p>

Reference implementation of our rewiring method as proposed in 

[Probabilistic Graph Rewiring via Virtual Nodes](https://arxiv.org/abs/2405.17311)  
Chendi Qian*, Andrei Manolache*, Christopher Morris<sup>†</sup>, Mathias Niepert<sup>†</sup>

*These authors contributed equally.  
<sup>†</sup>Co-senior authorship.

## Environment setup
```
conda create -n NAME python=3.10
conda activate NAME

conda install pytorch==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install openbabel fsspec rdkit -c conda-forge
pip install torch_geometric==2.4.0
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install multimethod wandb
pip install matplotlib seaborn ogb
pip install gdown
```

## To replicate experiments
We provide yaml files under `configs`, run e.g. 
`python run.py with PATH_TO_CONFIG`

In case of issues or other questions, please contact [chendi.qian@log.rwth-aachen.de](mailto:chendi.qian@log.rwth-aachen.de)
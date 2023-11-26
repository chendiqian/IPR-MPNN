# Environment preparation

```
conda create -n NAME python=3.10
conda activate NAME

conda install pytorch==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install multimethod ml-collections wandb
```

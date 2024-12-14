#!/usr/bin/env bash
export HGCN_HOME=$(pwd)
export LOG_DIR="$HGCN_HOME/logs"
export PYTHONPATH="$HGCN_HOME:$PYTHONPATH"
export DATAPATH="$HGCN_HOME/data"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
# source activate hgcn  # replace with source hgcn/bin/activate if you used a virtualenv

python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 1,1 --space_dim_list 7,1 --num-layers 3 --seed 1234 --dim 10 --epochs 100
python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 1,1 --space_dim_list 7,1 --num-layers 3 --seed 1235
python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 1,1 --space_dim_list 7,1 --num-layers 3 --seed 1236

python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 1,1 --space_dim_list 6,2 --num-layers 3 --seed 1234
python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 1,1 --space_dim_list 6,2 --num-layers 3 --seed 1235
python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 1,1 --space_dim_list 6,2 --num-layers 3 --seed 1236

python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 2,1 --space_dim_list 5,2 --num-layers 3 --seed 1234 --epochs 100
python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 2,1 --space_dim_list 5,2 --num-layers 3 --seed 1235 --epochs 100
python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 2,1 --space_dim_list 5,2 --num-layers 3 --seed 1236 --epochs 100

python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 2,1 --space_dim_list 4,3 --num-layers 3 --seed 1234
python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 2,1 --space_dim_list 4,3 --num-layers 3 --seed 1235
python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 2,1 --space_dim_list 4,3 --num-layers 3 --seed 1236

python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 2,2 --space_dim_list 3,3 --num-layers 3 --seed 1234
python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 2,2 --space_dim_list 3,3 --num-layers 3 --seed 1235
python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 2,2 --space_dim_list 3,3 --num-layers 3 --seed 1236

python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 1,1 --space_dim_list 4,4 --num-layers 3 --seed 1234
python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 1,1 --space_dim_list 4,4 --num-layers 3 --seed 1235
python3 -W ignore train.py --dataset cora --task nc --model PGCN  --manifold ProductManifold --manifold_list  PseudoHyperboloid,PseudoHyperboloid --time_dim_list 1,1 --space_dim_list 4,4 --num-layers 3 --seed 1236

python3 -W ignore train.py --dataset cora --task nc --model HGCN  --manifold PseudoHyperboloid --time_dim 1 --space_dim 9 --num-layers 3 --seed 1234 --dim 10 --min-epochs 50
python3 -W ignore train.py --dataset cora --task nc --model HGCN  --manifold PseudoHyperboloid --time_dim 1 --space_dim 9 --num-layers 3 --seed 1235 --dim 10 --min-epochs 50
python3 -W ignore train.py --dataset cora --task nc --model HGCN  --manifold PseudoHyperboloid --time_dim 1 --space_dim 9 --num-layers 3 --seed 1236 --dim 10 --min-epochs 50

python3 -W ignore train.py --dataset cora --task nc --model HGCN  --manifold PseudoHyperboloid --time_dim 2 --space_dim 8 --num-layers 3 --seed 1234 --dim 10 --epochs 100 
python3 -W ignore train.py --dataset cora --task nc --model HGCN  --manifold PseudoHyperboloid --time_dim 1 --space_dim 9 --num-layers 3 --seed 1235 --dim 10 --min-epochs 50
python3 -W ignore train.py --dataset cora --task nc --model HGCN  --manifold PseudoHyperboloid --time_dim 1 --space_dim 9 --num-layers 3 --seed 1236 --dim 10 --min-epochs 50

python3 -W ignore train.py --dataset cora --task nc --model HGCN  --manifold Hyperboloid --dim 10 --num-layers 3 --seed 1234 --epochs 100 --space_dim 10 --time_dim 0
python3 -W ignore train.py --dataset cora --task nc --model HGCN  --manifold Hyperboloid --dim 10 --num-layers 3 --seed 1235 --epochs 100
python3 -W ignore train.py --dataset cora --task nc --model HGCN  --manifold Hyperboloid --dim 10 --num-layers 3 --seed 1236 --epochs 100
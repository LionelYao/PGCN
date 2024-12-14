# Example Usage
# Assume manifolds are initialized (e.g., Euclidean(), Hyperboloid(), Sphere(), etc.)
from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time
from manifolds.euclidean import Euclidean
from manifolds.hyperboloid import Hyperboloid
from manifolds.sphere import Sphere
from manifolds.pseudohyperboloid import PseudoHyperboloid
from layers.product_layers import ProductManifoldLayer


import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel, MDModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics
import torch.nn.functional as F
import torch
from productmanifold import ProductManifold
from product_layers import ProductManifoldAggregation, ProductManifoldConvolution
from manifolds.euclidean import Euclidean
from manifolds.hyperboloid import Hyperboloid
from manifolds.sphere import Sphere

# Define individual manifolds
manifolds = [
    PseudoHyperboloid()
    Euclidean(dim=5),
    Hyperboloid(dim=3, curvature=1.0),
    Sphere(dim=4, curvature=0.5)
]

# Combine manifolds into a ProductManifold
product_manifold = ProductManifold(manifolds)

# Define layer parameters
in_features = [10, 8, 6]  # Input dimensions for each manifold
out_features = [5, 4, 3]  # Output dimensions for each manifold
curvatures = [None, 1.0, 0.5]  # Curvatures for each manifold
dropout = 0.5
act = torch.relu
use_bias = True
use_attention = True

# Define input tensors for each manifold
inputs = [torch.randn(32, in_dim) for in_dim in in_features]  # Batch size of 32
adj_matrix = torch.randn(32, 32)  # Adjacency matrix

# Test ProductManifoldAggregation
agg_layer = ProductManifoldAggregation(
    manifolds=manifolds,
    dimensions=out_features,
    curvatures=curvatures,
    dropout=dropout,
    use_attention=use_attention
)

# Perform forward pass for aggregation
print("Testing ProductManifoldAggregation...")
aggregated_output = agg_layer(inputs, adj_matrix)
print(f"Aggregated Output Shape: {aggregated_output.shape}")

# Test ProductManifoldConvolution
conv_layer = ProductManifoldConvolution(
    manifolds=manifolds,
    in_features=in_features,
    out_features=out_features,
    curvatures=curvatures,
    dropout=dropout,
    act=act,
    use_bias=use_bias,
    use_attention=use_attention
)

# Perform forward pass for convolution
print("Testing ProductManifoldConvolution...")
convolved_output = conv_layer(inputs, adj_matrix)
print(f"Convolved Output Shape: {convolved_output.shape}")
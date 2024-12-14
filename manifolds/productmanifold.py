"""Product manifold."""

import torch

from manifolds.base import Manifold
from utils.math_utils import arcosh, cosh, sinh 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math


class ProductManifold(Manifold):
    """
    Product of multiple manifolds.
    """

    def __init__(self, manifolds = []):
        """
        Initialize a product manifold with a list of individual manifolds.
        
        Args:
            manifolds (list of Manifold): The component manifolds.
        """
        super(ProductManifold, self).__init__()
        self.manifolds = manifolds
        self.name = "ProductManifold"
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-10
        self.max_norm = 1e4
        self.dim = sum(m.dim for m in self.manifolds)
        self.time_dims = [m.time_dim for m in self.manifolds]
        self.dims = [m.dim for m in self.manifolds]

    def expmap(self, u, p, c, time_dim = None):
        """
        Exponential map for the product manifold.
        
        Args:
            u (list of torch.Tensor): Tangent vectors for each manifold.
            p (list of torch.Tensor): Points on each manifold.
            c (list): Curvature for each manifold.
        
        Returns:
            list of torch.Tensor: Exponential map result for each manifold.
        """
        time_dim = self.time_dims if time_dim==None else time_dim
        return [m.expmap(u[i], p[i], c[i], time_dim[i]) for i, m in enumerate(self.manifolds)]

    def expmap0(self, u, c, time_dim = None):
        """
        Exponential map for the product manifold.
        
        Args:
            u (list of torch.Tensor): Tangent vectors for each manifold.
            p (list of torch.Tensor): Points on each manifold.
            c (list): Curvature for each manifold.
        
        Returns:
            list of torch.Tensor: Exponential map result for each manifold.
        """
        time_dim = self.time_dims if time_dim==None else time_dim
        return [m.expmap0(u[i], c[i], time_dim[i]) for i, m in enumerate(self.manifolds)]

    def logmap(self, u, p, c, time_dim = None):
        """
        Logarithmic map for the product manifold.
        
        Args:
            p1 (list of torch.Tensor): Points on each manifold.
            p2 (list of torch.Tensor): Points on each manifold.
            c (list): Curvature for each manifold.
        
        Returns:
            list of torch.Tensor: Logarithmic map result for each manifold.
        """
        time_dim = self.time_dims if time_dim==None else time_dim
        return [m.logmap(p1[i], p2[i], c[i], time_dim[i]) for i, m in enumerate(self.manifolds)]

    def logmap0(self, u, c, time_dim = None):
        """
        Logarithmic map for the product manifold.
        
        Args:
            p1 (list of torch.Tensor): Points on each manifold.
            p2 (list of torch.Tensor): Points on each manifold.
            c (list): Curvature for each manifold.
        
        Returns:
            list of torch.Tensor: Logarithmic map result for each manifold.
        """
        time_dim = self.time_dims if time_dim==None else time_dim
        #for i, m in enumerate(self.manifolds):
        #    print("Before operation logmap0: min=", u[i].min(), "max=", u[i].max(), "any NaNs?", torch.isnan(u[i]).any())
        return [m.logmap0(u[i], c[i], time_dim[i]) for i, m in enumerate(self.manifolds)]

    def proj(self, p, c, time_dim=None):
        """
        Project a point onto the product manifold.
        
        Args:
            p (list of torch.Tensor): Points for each manifold.
            c (list): Curvature for each manifold.
        
        Returns:
            list of torch.Tensor: Projection result for each manifold.
        """
        time_dim = self.time_dims if time_dim==None else time_dim
        return [m.proj(p[i], c[i], time_dim[i]) for i, m in enumerate(self.manifolds)]

    def proj_tan(self, u, p, c, time_dim=None):
        """
        Project a vector onto the tangent space of the product manifold.
        
        Args:
            u (list of torch.Tensor): Tangent vectors for each manifold.
            p (list of torch.Tensor): Points on each manifold.
            c (list): Curvature for each manifold.
        
        Returns:
            list of torch.Tensor: Tangent projection result for each manifold.
        """
        time_dim = self.time_dims if time_dim==None else time_dim
        return [m.proj_tan(u[i], p[i], c[i], time_dim[i]) for i, m in enumerate(self.manifolds)]

    def proj_tan0(self, u, c, time_dim=None):
        """
        Project a vector onto the tangent space of the product manifold.
        
        Args:
            u (list of torch.Tensor): Tangent vectors for each manifold.
            p (list of torch.Tensor): Points on each manifold.
            c (list): Curvature for each manifold.
        
        Returns:
            list of torch.Tensor: Tangent projection result for each manifold.
        """
        time_dim = self.time_dims if time_dim==None else time_dim
        #print("shape",u.shape)
        return [m.proj_tan0(u[i], c[i], time_dim[i]) for i, m in enumerate(self.manifolds)]

    def inner(self, p, c, u, v=None):
        """
        Compute the inner product of tangent vectors at a point on the product manifold.
        
        Args:
            p (list of torch.Tensor): Points on each manifold.
            c (list): Curvature for each manifold.
            u (list of torch.Tensor): Tangent vectors for each manifold.
            v (list of torch.Tensor, optional): Tangent vectors for each manifold.
        
        Returns:
            torch.Tensor: Inner product result.
        """
        if v is None:
            v = u
        return sum(m.inner(p[i], c[i], u[i], v[i]) for i, m in enumerate(self.manifolds))

    def mobius_add(self, x, y, c, time_dim=None):
        """
        Möbius addition for the product manifold.
    
        Args:
            x (list of torch.Tensor): Points on each manifold.
            y (list of torch.Tensor): Points on each manifold.
            c (list): Curvature for each manifold.
    
        Returns:
            list of torch.Tensor: Result of Möbius addition for each manifold.
        """
        time_dim = self.time_dims if time_dim==None else time_dim
        return [m.mobius_add(x[i], y[i], c[i], time_dim[i]) for i, m in enumerate(self.manifolds)]

    def mobius_matvec(self, m, x, c, time_dim = None, out_splits = None):
        """
        Möbius matrix-vector multiplication for the product manifold.

        Args:
            m (torch.Tensor): A single matrix to be applied after concatenation.
                          Shape: (sum of individual dimensions, sum of individual dimensions)
            x (list of torch.Tensor): Points on each manifold.
            c (list): Curvature for each manifold.
            time_dim: time dimension for each manifold.

        Returns:
            list of torch.Tensor: Result of Möbius matrix-vector multiplication for each manifold.
        """
        time_dim = self.time_dims if time_dim==None else time_dim
        # Step 1: Log map for each manifold
        #print(x[0].shape, c[0], time_dim[0])
        #print(x[0].shape, c[0], time_dim[0]) #x is not a list here
        logs = [manifold.logmap0(x[i], c[i], time_dim[i]) for i, manifold in enumerate(self.manifolds)]
        # Step 2: Concatenate log maps
        log_concat = torch.cat(logs, dim=-1)  # Shape: (batch_size, sum of dimensions)
        # Step 3: Apply the matrix
        
        transformed = log_concat @ m.transpose(-1, -2)  # Shape: (batch_size, sum of dimensions)
        # Step 4: Split the transformed result into corresponding dimensions

        transformed_splits = torch.split(transformed, out_splits, dim=-1)
        # Step 5: Exp map for each manifold
        mx = [manifold.expmap0(transformed_splits[i], c[i], time_dim[i]) for i, manifold in enumerate(self.manifolds)]

        return mx


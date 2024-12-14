import torch
import unittest
from product_layers import PNNLayer, ProLinear, ProAct, ProAgg
from productmanifold import ProductManifold

class MockManifold:
    """A mock manifold for testing."""
    def __init__(self, dim, time_dim=0):
        self.dim = dim
        self.time_dim = time_dim

    def expmap(self, u, p, c, time_dim=None):
        return u + p

    def expmap0(self, u, c, time_dim=None):
        return u

    def logmap(self, p1, p2, c, time_dim=None):
        return p2 - p1

    def logmap0(self, u, c, time_dim=None):
        return u

    def proj(self, p, c):
        return p

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def inner(self, p, c, u, v=None):
        if v is None:
            v = u
        return torch.sum(u * v)

    def mobius_add(self, x, y, c):
        return x + y

    def mobius_matvec(self, m, x, c, time_dim=None):
        return torch.matmul(m, torch.cat(x, dim=-1))

class TestProductLayers(unittest.TestCase):
    def setUp(self):
        """Initialize common test setups."""
        self.manifolds = [MockManifold(dim=3), MockManifold(dim=2)]
        self.product_manifold = ProductManifold(self.manifolds)

    def test_pro_linear(self):
        """Test ProLinear layer."""
        layer = ProLinear(self.product_manifold, 5, 7, [1.0, 1.0], use_bias=True)
        x = torch.randn(10, 5)  # Batch of 10, input size 5
        output = layer.forward(x)
        self.assertEqual(output.shape, (10, 7))

    def test_pro_act(self):
        """Test ProAct layer."""
        layer = ProAct(self.product_manifold, c_in=[1.0, 1.0], c_out=[1.0, 1.0], act=torch.relu)
        x = torch.randn(10, 5)  # Batch of 10, input size 5
        output = layer.forward(x)
        self.assertEqual(output.shape, (10, 5))

    def test_pnn_layer(self):
        """Test PNNLayer."""
        layer = PNNLayer(self.product_manifold, in_features=5, out_features=7, c=[1.0, 1.0], act=torch.relu, use_bias=True)
        x = torch.randn(10, 5)  # Batch of 10, input size 5
        output = layer.forward(x)
        self.assertEqual(output.shape, (10, 7))

    def test_product_manifold(self):
        """Test basic ProductManifold operations."""
        p = [torch.randn(10, 3), torch.randn(10, 2)]
        u = [torch.randn(10, 3), torch.randn(10, 2)]
        c = [1.0, 1.0]

        exp_result = self.product_manifold.expmap0(u, c)
        self.assertEqual(len(exp_result), 2)
        self.assertEqual(exp_result[0].shape, (10, 3))
        self.assertEqual(exp_result[1].shape, (10, 2))

        proj_result = self.product_manifold.proj(p, c)
        self.assertEqual(len(proj_result), 2)
        self.assertEqual(proj_result[0].shape, (10, 3))
        self.assertEqual(proj_result[1].shape, (10, 2))

if __name__ == '__main__':
    unittest.main()

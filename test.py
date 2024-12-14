import torch
import unittest
from manifolds.productmanifold import ProductManifold
from manifolds.pseudohyperboloid_sr import PseudoHyperboloid
from layers.product_layers import ProLinear, ProAct, ProAgg, PNNLayer

class TestProductManifold(unittest.TestCase):
    def setUp(self):
        """Initialize common test setups."""
        self.manifolds = [PseudoHyperboloid(space_dim=6, time_dim=1), PseudoHyperboloid(space_dim=5, time_dim=3)]
        self.product_manifold = ProductManifold(self.manifolds)

    def test_product_manifold(self):
        """Test basic ProductManifold operations."""
        p = [torch.randn(10, 7), torch.randn(10, 8)]
        u = [torch.randn(10, 7), torch.randn(10, 8)]
        c = [torch.tensor(-1, dtype=torch.float),torch.tensor(-1, dtype=torch.float)]
        #print(u)
        print(self.product_manifold.manifolds)
        exp_result = self.product_manifold.expmap0(u, c)
        self.assertEqual(len(exp_result), 2)
        self.assertEqual(exp_result[0].shape, (10, 7))
        self.assertEqual(exp_result[1].shape, (10, 8))

        proj_result = self.product_manifold.proj(p, c)
        self.assertEqual(len(proj_result), 2)
        self.assertEqual(proj_result[0].shape, (10, 7))
        self.assertEqual(proj_result[1].shape, (10, 8))

class TestProductLayers(unittest.TestCase):
    def setUp(self):
        """Initialize common test setups."""
        self.manifolds = [PseudoHyperboloid(space_dim=6, time_dim=1), PseudoHyperboloid(space_dim=5, time_dim=3)]
        self.product_manifold = ProductManifold(self.manifolds)

    def test_pro_linear(self):
        """Test ProLinear layer."""
        layer = ProLinear(self.product_manifold, [7,8], [4,5], [torch.tensor(-1, dtype=torch.float),torch.tensor(-1, dtype=torch.float)], use_bias=True)
        x = torch.randn(10, 15)  # Batch of 10, input size 5
        output = layer.forward(x)
        self.assertEqual(output.shape, (10, 9))

    def test_pro_act(self):
        """Test ProAct layer."""
        layer = ProAct(self.product_manifold, c_in=[torch.tensor(-1, dtype=torch.float),torch.tensor(-1, dtype=torch.float)], c_out=[torch.tensor(-1, dtype=torch.float),torch.tensor(-1, dtype=torch.float)], out_features = [4,5], act=torch.relu)
        x = torch.randn(10, 9)  # Batch of 10, input size 5
        output = layer.forward(x)
        self.assertEqual(output.shape, (10, 9))

    def test_pnn_layer(self):
        """Test PNNLayer."""
        layer = PNNLayer(self.product_manifold, in_features=[7,8], out_features=[4,5], c=[torch.tensor(-1, dtype=torch.float),torch.tensor(-1, dtype=torch.float)], act=torch.relu, use_bias=True)
        x = torch.randn(10, 15)  # Batch of 10, input size 5
        output = layer.forward(x)
        self.assertEqual(output.shape, (10, 9))

    def test_pro_agg(self):
        """Test ProAgg aggregation layer."""
        agg_layer = ProAgg(self.product_manifold, c=[torch.tensor(-1, dtype=torch.float),torch.tensor(-1, dtype=torch.float)], in_features=[7,8], local_agg=torch.spmm)
        x = torch.randn(10, 15)  # Input feature matrix (10 nodes, 5 features)
        adj = torch.eye(10)  # Adjacency matrix (identity for simplicity)

        output = agg_layer.forward(x, adj)

        # Assert the output has the same number of rows as input
        self.assertEqual(output.shape[0], x.shape[0])
        # Assert the output dimension matches the sum of manifold dimensions
        self.assertEqual(output.shape[1], sum(self.dims))


if __name__ == '__main__':
    unittest.main()
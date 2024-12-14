import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dim_split = almost_equal_split(args.feat_dim,len(args.dim_list))
    dims = [dim_split] + ([args.dim_list] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        for m,i in enumerate(args.manifold_list):
            if m in ['Hyperboloid', 'PoincareBall']:
                curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
            else:
                curvatures = [nn.Parameter(torch.Tensor([-1.]))  for _ in range(n_curvatures)]
    else: # fixed curvature
        curvatures = [torch.tensor([args.c], requires_grad=True) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class PNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(PNNLayer, self).__init__()
        #print(in_features, out_features)
        self.linear = ProLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.pro_act = ProAct(manifold, c, c, act, out_features)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.pro_act.forward(h)
        return h


class ProLinear(nn.Module):
    """
    Product Manifold linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):#in_features, out_features, curvatures are lists
        super(ProLinear, self).__init__()
        self.manifold = manifold
        self.time_dims = [m.time_dim for m in self.manifold.manifolds]
        self.dims = [m.dim for m in self.manifold.manifolds]
        self.in_features = in_features
        self.out_features = out_features
        self.curvatures = [c]*len(self.manifold.manifolds)
        self.dropout = dropout
        self.use_bias = use_bias
        print(in_features, out_features)
        if isinstance(in_features, int):
            if isinstance(out_features, int):
                self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            else:
                self.weight = nn.Parameter(torch.Tensor(sum(out_features), in_features))
        else:
            if isinstance(out_features, int):
                self.weight = nn.Parameter(torch.Tensor(out_features, sum(in_features)))
            else:
                self.weight = nn.Parameter(torch.Tensor(sum(out_features), sum(in_features)))
        if isinstance(out_features, int):
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias = nn.Parameter(torch.Tensor(sum(out_features)))
        self.reset_parameters()
        

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0.0001)

    def forward(self, x):#x is a tensor
        #print("layer dims", self.in_features,self.out_features)
        for k in x:
            assert not torch.isnan(k).any()
            #print(k.shape)
        #step 1: split x in terms of dimensions of the input dimension
        #print("ProLin forward dim", x.shape)
        #print("in_features, out_features", self.in_features,self.out_features)
        x_splits = torch.split(x, self.in_features, dim=-1)
        #print("splited dim",[x.shape for x in x_splits])
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        #print(drop_weight.shape,self.curvatures,self.time_dims,self.out_features)
        res = self.manifold.mobius_matvec(drop_weight, x_splits, self.curvatures, time_dim = self.time_dims, out_splits = self.out_features)
        res = self.manifold.proj(res, self.curvatures)
        for r in res:
            assert not torch.isnan(r).any()
        if self.use_bias:
            #print(self.bias.view(1, -1))
            bias_list = torch.split(self.bias.view(1, -1), self.out_features, dim=-1)
            bias = self.manifold.proj_tan0(bias_list, self.curvatures)
            hyp_bias = self.manifold.expmap0(bias, self.curvatures)
            hyp_bias = self.manifold.proj(hyp_bias, self.curvatures)
            # assert not torch.isnan(hyp_bias).any()
            res = self.manifold.mobius_add(res, hyp_bias, self.curvatures)
            res = self.manifold.proj(res, self.curvatures)
        # assert self.manifold._check_point_on_manifold(res,self.c)
        for r in res:
            assert not torch.isnan(r).any()
        return torch.cat(res, dim=-1)

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.curvatures
        )

class ProAct(nn.Module): #to write
    """
    Product manifold activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act, out_features):
        super(ProAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act
        self.out_features = out_features

    def forward(self, x):
        x_splits = torch.split(x, self.out_features , dim=-1)
        #print(self.manifold.logmap0(x_splits, self.c_in))
        logs = self.manifold.logmap0(x_splits, [self.c_in]*len(x_splits))
        xt = [torch.clamp(self.act(k),max=self.manifold.max_norm) for k in logs] #clamp issue
        xt = self.manifold.proj_tan0(xt, [self.c_out]*len(x_splits))
        output = self.manifold.expmap0(xt, [self.c_out]*len(x_splits))
        
        # assert self.manifold._check_point_on_manifold(output, self.c_out)
        # output = self.manifold.perform_rescaling_beta(output, self.c_out)
        return torch.cat(output, dim=-1)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
    

class ProAgg(nn.Module): #to write

    """
    Product manifold aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, local_agg):
        super(ProAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.curvatures = [c]*len(self.manifold.manifolds)
        self.dropout = dropout
        self.in_features = in_features
        self.local_agg = local_agg

    def forward(self, x, adj):
        assert not torch.isnan(x).any()
        #print(x.shape)
        x_splits = torch.split(x, self.manifold.dims, dim=-1)
        x_tangent = self.manifold.logmap0(x_splits, self.curvatures)
        x_tangent = torch.cat(x_tangent, dim = -1)
        support_t = torch.clamp(torch.spmm(adj, x_tangent), max=self.manifold.max_norm)

        assert not torch.isnan(x_tangent).any()
        assert not torch.isnan(support_t).any()
        support_t = torch.split(support_t, self.manifold.dims, dim=-1)
        res = self.manifold.proj_tan0(support_t,self.curvatures)
        res = self.manifold.expmap0(res, self.curvatures)
        output = self.manifold.proj(res,self.curvatures)
        # assert self.manifold._check_point_on_manifold(output,self.c)
        # output = self.manifold.perform_rescaling_beta(output, self.c)
        return torch.cat(output, dim=-1)

    def extra_repr(self):
        return 'c={}'.format(self.c)

class ProGraphConv(nn.Module):
    """
    Product Manifold graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout ,act, use_bias, local_agg):
        super(ProGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.linear = ProLinear(manifold, in_features, out_features, self.c_in, dropout,use_bias)
        self.agg = ProAgg(manifold, self.c_in, out_features, dropout,local_agg)
        self.pro_act = ProAct(manifold, self.c_in, self.c_out, act, out_features)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.pro_act.forward(h)
        output = h, adj
        return output

def almost_equal_split(n, k):
    """Splits an integer n into k almost equal parts."""

    base = n // k  # Integer division to get the base value
    remainder = n % k  # Get the remainder

    result = [base] * k  # Initialize a list with k base values

    # Distribute the remainder across the first few elements
    for i in range(remainder):
        result[i] += 1

    return result
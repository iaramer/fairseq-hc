import torch
from torch import nn
import numpy as np


PERFECT_CUBES = {x**3: x for x in range(2,33)}


class HyperCubeLayer(nn.Module):
    __constants__ = ['weight_tensor_shape']
    in_features: int
    out_last_dim_features: int
    weight: torch.Tensor

    def __init__(self, weight_tensor_shape: tuple(int), bias: bool = True,
                 equation='bnijk,ijkl->bnjkl', device=None, dtype=None) -> None:
        """
        The weight tensor must be of shape (i,j,k,l)

        Example of equation:
        'bnijk,ijkl->bnjkl'
        b - batch size
        n - sequence length
        i, j, k - input feature tensor
        j, k, l - output feature tensor
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        assert len(weight_tensor_shape) != 4, "Input weight tensor shape must be of size 4"
        self.weight_tensor_shape = weight_tensor_shape
        self.weight = nn.Parameter(torch.empty(weight_tensor_shape, **factory_kwargs))
        self.equation = equation
        if bias:
            self.bias = nn.Parameter(torch.empty((weight_tensor_shape[-1],), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return 'weight_tensor_shape={}, bias={}'.format(
            self.weight_tensor_shape, self.bias is not None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum(self.equation, x, self.weight) + self.bias
        return x


class HyperCubeBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super(HyperCubeBlock, self).__init__()
        if in_features not in PERFECT_CUBES or out_features not in PERFECT_CUBES:
            raise NotImplementedError(f"Both `in_features` and `out_features` must be perfect cubes, however the following params were given: {in_features} and {out_features}")
        self.in_features_edge_dim = in_edge = PERFECT_CUBES[in_features]
        self.out_features_edge_dim = out_edge = PERFECT_CUBES[out_features]
        self.hc_layers_1 = HyperCubeLayer(
            (in_edge, in_edge, in_edge, out_edge),
            bias=bias,
            equation='bnijk,ijkl->bnilj',
            device=device, 
            dtype=dtype
        )
        self.hc_layers_2 = HyperCubeLayer(
            (in_edge, out_edge, in_edge, out_edge),
            bias=bias,
            equation='bnilj,iljp->bnpli',
            device=device, 
            dtype=dtype
        )
        self.hc_layers_3 = HyperCubeLayer(
            (out_edge, out_edge, in_edge, out_edge),
            bias=bias,
            equation='bnpli,pliq->bnplq',
            device=device, 
            dtype=dtype
        )
        # self.relu = nn.ReLU()
            
    def forward(self, x):
        sh = x.shape
        in_edge = self.in_features_edge_dim
        x = x.view((*sh[:-1], in_edge, in_edge, in_edge))
        x = self.hc_layers_1(x)
        # x = self.relu(x)  # TODO: Experiment with intermediate activation
        x = self.hc_layers_2(x)
        # x = self.relu(x)  # TODO: Experiment with intermediate activation
        x = self.hc_layers_3(x)
        x = x.view(sh)
        return x

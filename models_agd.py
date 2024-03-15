import torch, math

import torch.nn.functional as F

def gelu(x): return F.gelu(x) * math.sqrt(2)
def relu(x): return F.relu(x) * math.sqrt(2)

def spectral_norm(p, u, num_steps=1):
    for _ in range(num_steps):
        u /= u.norm(dim=0, keepdim=True)
        v = torch.einsum('ab..., b... -> a...', p, u)
        u = torch.einsum('a..., ab... -> b...', v, p)
    return u.norm(dim=0, keepdim=True).sqrt(), u


# class Linear(torch.nn.Module):

#     def __init__(self, in_features, out_features, init_scale):
#         super().__init__()
#         self.scale = math.sqrt(out_features / in_features)*init_scale
#         self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
#         torch.nn.init.orthogonal_(self.weight)
#         self.weight.data *= self.scale
#         self.register_buffer("momentum", torch.zeros_like(self.weight))
#         self.register_buffer("u",        torch.randn_like(self.weight[0]))

#         self.bias = torch.nn.Parameter(torch.empty((out_features, )))
#         torch.nn.init.zeros_(self.bias)
#         self.register_buffer("momentum_bias", torch.zeros_like(self.bias))

#     def forward(self, input):
#         return F.linear(input, self.weight)+self.bias

#     @torch.no_grad()
#     def update(self, lr, beta, wd):
#         self.momentum += (1-beta) * (self.weight.grad - self.momentum)
#         spec_norm, self.u = spectral_norm(self.momentum, self.u)
#         self.weight -= lr * torch.nan_to_num(self.momentum / spec_norm, 0, 0, 0) * self.scale
#         self.weight *= 1 - lr * wd

#         self.momentum_bias += (1-beta) * (self.bias.grad - self.momentum_bias)
#         self.bias -= lr * torch.nan_to_num(self.momentum_bias / spec_norm, 0, 0, 0) * self.scale
#         self.bias *= 1 - lr * wd


class Linear(torch.nn.Module):

    def __init__(self, in_features, out_features, init_scale=1):
        super().__init__()
        self.scale = math.sqrt(out_features / in_features)*init_scale
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        torch.nn.init.orthogonal_(self.weight)
        self.weight.data *= self.scale
        self.register_buffer("momentum", torch.zeros_like(self.weight))
        self.register_buffer("u",        torch.randn_like(self.weight[0]))

    def forward(self, input):
        return F.linear(input, self.weight)

    @torch.no_grad()
    def update(self, lr, beta, wd):
        self.momentum += (1-beta) * (self.weight.grad - self.momentum)
        spec_norm, self.u = spectral_norm(self.momentum, self.u)
        self.weight -= lr * torch.nan_to_num(self.momentum / spec_norm, 0, 0, 0) * self.scale
        self.weight *= 1 - lr * wd


class MLP(torch.nn.Module):
    def __init__(self, depth, width, input_dim, output_dim, init_scale):
        super(MLP, self).__init__()
        self.depth   = depth
        self.initial = Linear(input_dim, width, init_scale)
        self.layers  = torch.nn.ModuleList([Linear(width, width, init_scale) for _ in range(depth-2)])
        self.exit    = Linear(width, output_dim, init_scale)

    def forward(self, x):
        x = self.initial(x)
        x = relu(x)

        for layer in self.layers:
            x = layer(x)
            x = relu(x)

        return self.exit(x)

    @torch.no_grad()
    def update(self, lr, beta, wd):
        self.initial.update(lr / self.depth, beta, wd)

        for layer in self.layers:
            layer.update(lr / self.depth, beta, wd)
        
        self.exit.update(lr / self.depth, beta, wd)


import math
import torch

class ResMLP(torch.nn.Module):
    def __init__(self, num_blocks, block_depth, width, input_dim, output_dim):
        super(ResMLP, self).__init__()

        self.num_blocks = num_blocks
        self.width = width
        self.initial = Linear(input_dim, width)
        self.layers  = torch.nn.ModuleList([MLP(block_depth, width, width, width) for _ in range(num_blocks-2)])
        self.exit    = Linear(width, output_dim)

    def forward(self, x):
        x = self.initial(x)

        for layer in self.layers:
            y = torch.nn.functional.layer_norm(x, (self.width,))
            x = x + layer(y) / math.sqrt(self.num_blocks)

        return self.exit(x)

    @torch.no_grad()
    def update(self, lr, beta, wd):
        self.initial.update(lr / self.num_blocks, beta, wd)

        for layer in self.layers:
            layer.update(lr / math.sqrt(self.num_blocks), beta, wd)
        
        self.exit.update(lr / self.num_blocks, beta, wd)

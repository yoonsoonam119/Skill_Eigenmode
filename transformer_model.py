import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, Embedding
from torch.nn.functional import relu, relu



class CustomEmb(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1 = (x==0).int()
        x2 = (x==1).int()
        x = torch.stack((x1,x2), dim=2)
        pos_enc = torch.eye(x.shape[1]).unsqueeze(0).repeat(x.shape[0],1,1)
        device = 'cpu' if x.get_device()==-1 else 'cuda'
        x = torch.cat((x, pos_enc.to(device)), dim=2)
        return x

# emb = CustomEmb()
# x = torch.randint(0,2, size=(100, 50))
# emb(x)


class MLP(torch.nn.Module):
    def __init__(self, depth, width, input_dim, output_dim):
        super().__init__()

        self.width   = width
        self.initial = Linear(input_dim, width)
        self.layers  = torch.nn.ModuleList([Linear(width, width) for _ in range(depth-2)])
        self.exit    = Linear(width, output_dim)

    def forward(self, x):
        x = self.initial(x)
        x = relu(x)

        for layer in self.layers:
            x = layer(x)
            x = relu(x)

        return self.exit(x)

class SelfAttention(nn.Module):

    def __init__(self, n_head, n_embd, block_size, causal):
        super().__init__()
        assert n_embd % n_head == 0
        self.Q    = Linear(n_embd, n_embd, bias=True)
        self.K    = Linear(n_embd, n_embd, bias=True)
        self.V    = Linear(n_embd, n_embd, bias=True)
        self.exit = Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.causal = causal
        if self.causal:
            bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
            self.register_buffer("bias", bias)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q = self.Q(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.K(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.V(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        if self.causal:
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        y = F.softmax(att, dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.exit(y)

class Block(nn.Module):

    def __init__(self, n_head, n_embd, block_size, causal, out=None):
        super().__init__()
        print('AAAAAA',n_head, n_embd)
        if out is None:
            out = n_embd
        self.ln_1 = nn.LayerNorm(n_embd, elementwise_affine=True)
        self.attn = SelfAttention(n_head, n_embd, block_size, causal)
        self.ln_2 = nn.LayerNorm(out, elementwise_affine=True)
        self.mlp = MLP(depth=2, width=4*n_embd, input_dim=n_embd, output_dim=out)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, n_head, n_embd, n_layer, context_length, vocab_size, n_skills, ex):
        super().__init__()

        self.context_length = context_length

        self.lin_layer = nn.Linear(in_features=n_embd, out_features=ex)

        self.transformer = nn.ModuleDict(dict(
            wt_embedding = CustomEmb(), #Embedding(vocab_size, n_embd),
            wp_embedding = CustomEmb(), #Embedding(context_length, n_embd),

            h = nn.ModuleList(
                [Block(n_head, ex, context_length, causal=True, out=ex) for _ in range(n_layer)]
                +
                [Block(n_head, ex, context_length, causal=True) for _ in range(n_layer-1)]
                ),
            ln_f = nn.LayerNorm(ex, elementwise_affine=True),
        ))
        self.lm_head = Linear(ex*context_length, vocab_size)

        num_blocks = sum(1 for name, p in self.named_parameters() if 'exit' in name)

        for name, p in self.named_parameters():
            if 'exit' in name:
                p.data /= math.sqrt(num_blocks)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.context_length, f"Cannot forward sequence of length {t}, block size is only {self.context_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wt_embedding(idx) # token embeddings of shape (b, t, n_embd)
        # pos_emb = self.transformer.wp_embedding(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb # + pos_emb

        # MLP

        x = self.lin_layer(x)

        x = relu(x)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = x.reshape(x.shape[0], -1)

        return self.lm_head(x)

class ViT(nn.Module):

    def __init__(self, n_head, n_embd, n_layer, num_patches, img_size, num_channels, num_classes):
        super().__init__()

        self.block_size = num_patches**2 + 1
        self.patch_size = img_size // num_patches
        pixels_per_patch = self.patch_size**2 * num_channels

        self.transformer = nn.ModuleDict(dict(
            patch_embedding = Linear(pixels_per_patch, n_embd),
            position_embedding = Embedding(self.block_size, n_embd),
            h = nn.ModuleList([Block(n_head, n_embd, self.block_size, causal=False) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd, elementwise_affine=True),
        ))
        self.cls_token = nn.Parameter(torch.randn(n_embd))
        self.lm_head = Linear(n_embd, num_classes)

        num_blocks = sum(1 for name, p in self.named_parameters() if 'exit' in name)

        for name, p in self.named_parameters():
            if 'exit' in name:
                p.data /= math.sqrt(num_blocks)

    def forward(self, x):
        n_patch = x.shape[2] // self.patch_size * x.shape[3] // self.patch_size
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        patches = patches.reshape(x.size(0), n_patch, -1)

        pos = torch.arange(0, self.block_size, dtype=torch.long, device=patches.device)
        pos_emb   = self.transformer.position_embedding(pos)
        patch_emb = self.transformer.patch_embedding(patches)
        patch_emb = torch.cat([self.cls_token.view(1,1,self.cls_token.shape[0]).repeat(patch_emb.size(0),1,1), patch_emb],dim=1)

        x = patch_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        return self.lm_head(x[:,0])

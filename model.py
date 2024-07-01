import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for the query
    n_kv_heads: Optional[int] = None # Number of the heads for the key and value
    vocab_size: int = -1 # Will set this while loading the model
    multiple_of: int  = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Params for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precompute_theta_pos_frequency(head_dim:int, seq_len:int ,device:str, theta:float = 10000.0):

    # As per the paper the head_dim should be an even number
    assert head_dim % 2 == 0, "Dimension must be an even number"
    
    # Calculating the theta = 1000 ^ -(2 * (i - 1))/dim, where i->0:dim/2. As per paper
    theta_numerator = torch.arange(0, head_dim, 2).float() / head_dim
    theta = 1.0 / (theta ** (theta_numerator)).to(device)

    # Create the position wise (m) wise m*theta
    m = torch.arange(seq_len, device = device)

    # Now, multiply each theta with each with all the m's
    # Shape m:(seq_len), theta: (head_dim / 2) --> freqs: (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()

    # Now, once we have all the m*theta we can calculate the re^(i * m * theta)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):

    # (B, Seq_len, H, head_dim) -> (B, Seq_len, H, head_dim / 2, 2) as we are pairing consequtive embeds.
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Now we have to multiply x_complex and complex freqs element wise as per paper
    # Shape : (1, Seq_len, 1, head_dim / 2)
    freqs_complex =freqs_complex.unsqueeze(0).unsqueeze(2)

    # Shape (B, Seq_len, H, head_dim / 2, 2)
    x_rotated = x_complex * freqs_complex

    # Converting the obtained complex numbers to tensors.
    # Shape (B, Seq_len, H, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)

    # Then we flatten x_out and bring it back to the original tensor shape
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim))
    def _norm(self, x: torch.Tensor):
        # (B, T, C)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x: torch.Tensor):
        # (C) * (B, T, C) -> (B, T, C)
        return self.weight * self._norm(x.float()).type_as(x)

class FeedForward(nn.Module):
    
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = int(config.ffn_dim_multiplier * config.dim) if config.ffn_dim_multiplier is not None else 4 * config.dim
        
        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, dim) -> (B, T, hidden_dim)
        swish = F.silu(self.w1(x))
        # (B, T, dim) -> (B, T, hidden_dim)
        x_V = self.w3(x)
        # (B, T, hidden_dim) * (B, T, hidden_dim) -> (B, T, hidden_dim)
        x = swish * x_V
        # (B, T, hidden_dim) -> (B, T, dim)
        x = self.w2(x)
        return x

class SelfAttention(nn.Module):

    def __init__(self, config: ModelArgs) -> None:
        super().__init__()

        self.kv_heads = config.n_kv_heads if config.n_kv_heads is not None else self.n_heads
        # Number of dimensions of the embedding
        self.dim = config.dim
        # Number of heads
        self.n_heads = config.n_heads
        # head dim for each head.
        self.head_dim = self.dim // self.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.cache_k = torch.zeros((config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim))
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len , _  = x.shape
        
        assert seq_len == 1, "Only one token can be processed at a time"

        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim) # (B, T, dim) -> (B, T, n_heads * head_dim)-> (B, T, n_heads, head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.kv_heads, self.head_dim) # (B, T, dim) -> (B, T, n_kv_heads * head_dim) -> (B, T, n_kv_heads, head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.kv_heads, self.head_dim) # (B, T, dim) -> (B, T, n_kv_heads * head_dim) -> (B, T, n_kv_heads, head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex, device = x.device) # (B, T, n_heads, head_dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device = x.device) # (B, T, n_kv_heads, head_dim)

        # accessing and modifying the kv cache for the last token
        self.cache_k[ :batch_size, : start_pos: start_pos + seq_len] = xk
        self.cache_v[ :batch_size, : start_pos: start_pos + seq_len] = xv

        k = self.cache_k[:batch_size, start_pos: start_pos + seq_len]
        v = self.cache_v[:batch_size, start_pos: start_pos + seq_len]

        # repeat the queries for the kv pairs
        k = repeat_kv(k, self.n_rep) # (B, T, n_kv_heads, head_dim) -> (B, T, n_heads, head_dim)
        v = repeat_kv(v, self.n_rep) # (B, T, n_kv_heads, head_dim) -> (B, T, n_heads, head_dim)

        xq = xq.transpose(1 , 2) # (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)

        keys = k.transpose(1, 2) # (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        values = v.transpose(1, 2) # (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)

        attn = (xq @ keys.transpose(-2, -1)) * (self.head_dim ** -0.5) # (B, n_heads, T, T)
        attn = F.softmax(attn.float(), dim = -1).type_as(xq) # (B, n_heads, T, T)

        out = attn @ values # (B, n_heads, T, T) @ (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).view(batch_size, seq_len, -1) # (B, n_heads, T, head_dim) -> (B, T, n_heads * head_dim)
        out = self.wo(out)
        return out
class EncoderBlock(nn.Module):
    
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = self.dim // self.n_heads
        self.attention = SelfAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(self.dim, eps=config.norm_eps)
        self.ff_norm = RMSNorm(self.dim, eps=config.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        # (B, Seq_Len, dim) + (B, Seq_Len, dim) -> (B, Seq_Len, dim)
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex)
        
        # (B, Seq_Len, dim) + (B, Seq_Len, dim) -> (B, Seq_Len, dim)
        out = h + self.feed_forward.forward(self.ff_norm(h))

        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        
        assert args.vocab_size != -1 # make sure to set vocab size

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias = False)

        # Rotary KV frequencies
        self.freqs_complex = precompute_theta_pos_frequency(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)


    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (Batch, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token can be processed at a time"

        # (B, T) -> (B, T, dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions (start_pos, start_pos + seq_len)
        freqs_complex = self.freqs_complex[start_pos, start_pos + seq_len]

        # Pass over all encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output



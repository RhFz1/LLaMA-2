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

def precompute_theta_pos_frequencies(head_dim:int, device:str, theta:float = 10000.0):

    # As per the paper the head_dim should be an even number
    assert head_dim % 2 == 0, "Dimension must be an even number"
    

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



import torch
import tqdm
import time
import json
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from typing import Optional, List
from model import Transformer, ModelArgs


class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
    
    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()

        if load_model:
            checkpoints = list(Path(checkpoints_dir).glob('*.pth'))
            assert len(checkpoints) > 0, f"No checkpoints found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading model from {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            print(f'Model loaded in {time.time() - prev_time:.2f} seconds')
            prev_time = time.time()
        with open(Path(checkpoints_dir) + 'params.json', 'r') as file:
            params = json.load(file.read())
        
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            device=device,
            max_batch_size=max_batch_size,
            **params
        
        )
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == 'cuda':
            torch.set_default_tensor_type(torch.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        
        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f'Loaded model statedict in {time.time() - prev_time:.2f}')
        
        return LLaMA(Transformer, tokenizer, model_args)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

prompts = ['Hey There LLaMA, how are you?'] 

model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )
import torch
from sentencepiece import SentencePieceProcessor
from model import Transformer, ModelArgs

tokenizer_path='tokenizer.model'

model_args = ModelArgs()


tokenizer = SentencePieceProcessor()
tokenizer.load(tokenizer_path)
model_args.vocab_size = tokenizer.vocab_size()
model = Transformer(model_args)


x = torch.randint(0, 50257, (1, 1024))
x = tokenizer.encode(x, out_type=int, add_bos=True, add_eos=True)

print(x.shape)

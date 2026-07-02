from transformers import AutoTokenizer
import os

token = "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"

print("Downloading SPLADE tokenizer locally...")
splade_tok = AutoTokenizer.from_pretrained("naver/splade-v3", token=token)
splade_tok.save_pretrained("models/splade_tokenizer")

print("Downloading BGE-M3 tokenizer locally...")
bge_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3", token=token)
bge_tok.save_pretrained("models/bge_tokenizer")

print("Tokenizers downloaded to local directories!")

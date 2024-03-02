import os
import time
from minbpe import BasicTokenizer, RegexTokenizer
"""
Trains minbpe tokenizer on resulting patch embeddings from Patchnizer for image patches of the first video frame.
The whole training takes ~30 seconds on my M2 Macbook Air.
Run patchnizer.py before running this code.
"""

text = ""
embeddings_dir = "patch_embeddings"

first_frame = next(os.path.join(embeddings_dir, name) for name in os.listdir(embeddings_dir) if os.path.isdir(os.path.join(embeddings_dir, name)))

text = ""
for embedding_name in os.listdir(first_frame):
    if embedding_name.endswith('.txt'):
        embedding_path = os.path.join(first_frame, embedding_name)
        with open(embedding_path, 'r') as f:
            text += f.read()

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)
    
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")

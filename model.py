
from tokenizers import Tokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class KeyboardDataset(Dataset):
    def __init__(self, txt_file, tokenizer_path, max_length=32):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length

        # Load all text and tokenize it into one giant list of IDs
        with open(txt_file, 'r', encoding='utf-8') as f:
            full_text = " ".join([line.strip() for line in f if len(line.strip()) > 2])

        self.tokens = self.tokenizer.encode(full_text).ids

    def __len__(self):
        # Number of sequences we can create
        return (len(self.tokens) - 1) // self.max_length

    def __getitem__(self, idx):
        # Change: Jump by max_length so we don't repeat data 32 times
        start_ptr = idx * self.max_length
        end_ptr = start_ptr + self.max_length

        x = torch.tensor(self.tokens[start_ptr: end_ptr])
        y = torch.tensor(self.tokens[start_ptr + 1: end_ptr + 1])
        return x,




# 1. Setup Device for M4
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. Hyperparameters
vocab_size = 10000  # Your chosen size
n_embd = 256        # Size of the "vector" for each word
n_head = 8          # Number of attention "heads"
n_layer = 4         # Number of transformer blocks
block_size = 32     # Your context length

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class KeyboardTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# Instantiate the model and move to M4 GPU
model = KeyboardTransformer().to(device)
print(f"Model initialized on {device}")
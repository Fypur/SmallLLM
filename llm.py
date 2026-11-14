from llmbase import *

# Create Multiheaded attention
# Add a feedforward layer

# Add residual path
# Add layer norm
# Add dropouts

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.head_size = head_size
        self.keys = nn.Linear(n_embd, head_size, bias=False)
        self.queries = nn.Linear(n_embd, head_size, bias=False)
        self.values = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, input):
        # get keys, queries, values
        # input is of shape (B, T, C) where T is block_size and C is n_embd
        # H is head_size

        B, T, C = input.shape

        k = self.keys(input) # batch matrix mul (B, T, C) * (C, H) -> (B, T, H)
        q: torch.Tensor = self.queries(input)
        v = self.values(input)

        wei: torch.Tensor = q @ k.transpose(dim0=-2, dim1=-1) / self.head_size ** 0.5 # (B, T, H) * (B, H, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # type: ignore # can't see in the future
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v # (T, T) * (B, T, H) -> (B, T, H)
        return out

class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for i in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        out = torch.cat([head(input) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        return self.net(input)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.multi_head = MultiHead(4, n_embd // 4)
        self.ffwd = FeedForward()

    def forward(self, x):
        x = x + self.multi_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MicroLLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(*[Block() for _ in range(num_blocks)])
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.final_linear = nn.Linear(n_embd, vocab_size)

    def forward(self, input: torch.Tensor, targets=None):
        # input is (B, T)
        B, T = input.shape
        # we want (B, T, C) with probs for next one at every point

        embd = self.token_embedding(input) # (B, T) -> (B, T, C) giving embeddings for each num
        pos_embd = self.positional_embedding(torch.arange(T, device=input.device))
        x = embd + pos_embd

        x = self.blocks(x)
        x = self.final_layer_norm(x)

        logits = self.final_linear(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_view = logits.view(B*T, C)
            targ_view = targets.view(B*T)
            loss = F.cross_entropy(logits_view, targ_view)

        return logits, loss

    def generate(self, idx, max_tokens=1000):
            for _ in range(max_tokens):
                #idx has shape (B, T)
                idx_cond = idx[:, -block_size:] # cut everything before block size (if contect too big)
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] # we only care about the prediction for the next token
                probs = F.softmax(logits, dim=-1)
                preds = torch.multinomial(probs, 1, replacement=True)
                idx = torch.cat((idx, preds), dim=1)
    
            return idx

mic = MicroLLM()
mic.to(device)
optimizer = torch.optim.AdamW(mic.parameters(), lr=1e-3)

for i in range(5000):
    xs, ys = load_batch(training_set)
    logits, loss = mic(xs, ys)

    mic.zero_grad(set_to_none=True)
    loss.backward()

    optimizer.step()

    if i % 100 == 0:
        losses = estimate_loss(mic)
        print(f"step {i}, training : {losses["training_set"]:.4f}, test : {losses["test_set"]:.4f}")

# loss of 2.14 :O
print(decode(mic.generate(torch.zeros((1, 1), dtype=torch.long, device=device))[0].tolist()))
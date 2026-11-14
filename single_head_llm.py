from llmbase import *

# Create a head of attention

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.head_size = head_size
        self.keys = nn.Linear(n_embd, head_size)
        self.queries = nn.Linear(n_embd, head_size)
        self.values = nn.Linear(n_embd, head_size)

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

        out = wei @ v # (T, T) * (B, T, H) -> (B, T, H)
        return out

class MicroLLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.head = Head(vocab_size)

    def forward(self, input: torch.Tensor, targets=None):
        # input is (B, T)
        # we want (B, T, C) with probs for next one at every point
        embd: torch.Tensor = self.embedding(input) # (B, T) -> (B, T, C) giving probabilities for each num
        logits = self.head(embd)

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
optimizer = torch.optim.AdamW(mic.parameters(), lr=1e-3)

for i in range(10000):
    xs, ys = load_batch(training_set)
    logits, loss = mic(xs, ys)

    mic.zero_grad(set_to_none=True)
    loss.backward()

    optimizer.step()

    if i % 1000 == 0:
        print(estimate_loss(mic))

# loss of 2.46
print(decode(mic.generate(torch.zeros((1, 1), dtype=torch.long))[0].tolist()))

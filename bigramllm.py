from llmbase import *
# Make a bigram model for a baseline

class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input: torch.Tensor, targets=None):
        # input is (B, T)
        # we want (B, T, C) with probs for next one at every point
        logits: torch.Tensor = self.embedding(input) # (B, T) -> (B, T, C) giving probabilities for each num

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



big = BigramModel()
optimizer = torch.optim.AdamW(big.parameters(), lr=1e-3)

print(decode(big.generate(torch.zeros((1, 1), dtype=torch.long))[0].tolist()))

for i in range(10000):
    xs, ys = load_batch(training_set)
    logits, loss = big(xs, ys)

    big.zero_grad(set_to_none=True)
    loss.backward()

    optimizer.step()

    if i % 1000 == 0:
        print(estimate_loss(big))

# baseline loss : 2.47
print(decode(big.generate(torch.zeros((1, 1), dtype=torch.long))[0].tolist()))

import torch
import torch.nn.functional as F
# Load dataset and count characters
# Create embeddings
# Create dataset (xs, ys)
# Create W logits matrix
# calc negative log likelihood
# Backwards pass
# Sample a name

# Opt : Create N matrix with counts and see if they match up

text = open("names.txt").read()
words = text.splitlines()

chars = ["."] + sorted(list(set("".join(words))))

stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}

xs = []
ys = []
for name in words[:5]:
    w = ["."] + list(name) + ["."]
    for ch1, ch2 in zip(w, w[1:]):
        i1 = stoi[ch1]
        i2 = stoi[ch2]
        xs.append(i1)
        ys.append(i2)
        # print(ch1, ch2)

xs = torch.tensor(xs)
ys = torch.tensor(ys) # (5)

torch.manual_seed(2147483647)

W = torch.randn((27, 27), requires_grad=True, dtype=torch.float32)



"""for i in range(len(xs)):
    x = xs[i].item()
    y = ys[i].item()
    ch1 = itos[x]
    ch2 = itos[y]
    print(f"Bigram {ch1}{ch2}")
    print(f"Probability {probs[i, ys[i]]}")
    print(f"Negative log likelihood {-torch.log(probs[i, ys[i]])}")"""



for k in range(100):
    #forward pass
    logits = W[xs]
    # e = logits.exp()
    # probs = e / e.sum(1, keepdim=True) # (len(xs), 27) * (27, 27) -> (len(xs), 27)

    # loss = -probs[torch.arange(len(xs)), ys].log().mean() + 0.01 * (W**2).mean()
    loss = F.cross_entropy(logits, torch.nn.functional.one_hot(ys).float()) + (W**2).mean()
    print(loss.item())

    W.grad = None
    loss.backward()

    W.data -= 50 * W.grad

torch.manual_seed(2147483647)

for n in range(5):
    ix = 0
    for i in range(100):
        x = torch.tensor([ix])

        logits = W[x]
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)

        ix = torch.multinomial(probs[0], 1, replacement=True).item()

        if ix == 0:
            break
        else:
            print(itos[ix], end="")
        
    print(end="\n")
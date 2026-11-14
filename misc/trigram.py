import os
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

training_len = int(0.8 * len(words))
dev_len = int(0.9 * len(words))
training_set = words[:training_len]
dev_set = words[training_len:dev_len]
test_set = words[dev_len:]

chars = ["."] + sorted(list(set("".join(words))))

stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}

def get_encoding(used_set):
    x1s = []
    x2s = []
    ys = []
    for name in used_set:
        w = ["."] + list(name) + ["."]
        for ch1, ch2, ch3 in zip(w, w[1:], w[2:]):
            i1 = stoi[ch1]
            i2 = stoi[ch2]
            i3 = stoi[ch3]
            x1s.append(i1)
            x2s.append(i2 + 27)
            ys.append(i3)
            # print(ch1, ch2)

    x1s = torch.tensor(x1s)
    x2s = torch.tensor(x2s)
    xenc = F.one_hot(x1s, num_classes=27*2).float() # (5, 54)
    xenc[torch.arange(len(x1s)), x2s] = 1
    ys = torch.tensor(ys) # (5)

    return xenc, ys

torch.manual_seed(2147483646)

W = torch.randn((54, 27), requires_grad=True, dtype=torch.float32)

def get_loss(xenc, ys, regularization=0.01):
    logits = xenc @ W
    e = logits.exp()
    probs = e / e.sum(1, keepdim=True) # (len(xs), 27) * (27, 27) -> (len(xs), 27)

    return -probs[torch.arange(xenc.shape[0]), ys].log().mean() + regularization * (W**2).mean()

# test loss
def test_model(used_set):
    xenc, ys = get_encoding(used_set)
    train_loss = get_loss(xenc, ys, 0)    
    print(f"test loss : {train_loss.item()}")
    return train_loss.item()

def train_model(regularization):
    global W
    xenc, ys = get_encoding(training_set)
        
    torch.manual_seed(2147483646)
    W = torch.randn((54, 27), requires_grad=True, dtype=torch.float32)

    for k in range(30):
        #forward pass
        loss = get_loss(xenc, ys, regularization)    
        # print(f"train loss : {loss.item()}")

        W.grad = None
        print(loss)
        loss.backward()


        W.data -= 50 * W.grad
        
        if k % 10 == 0: # Print less often so your console isn't flooded
            print(f"Epoch {k}: Loss {loss.item():.4f}")

test_result_losses = []
for i in range(10):
    train_model(i * 0.005)
    test_result_losses.append(test_model(dev_set))

final_reg = 0.005 * torch.argmin(torch.tensor(test_result_losses))

print(f"final reg {final_reg}")
test_model(test_set)


for n in range(5):
    next = 0
    next2 = torch.randint(low=1, high=27, size=(1,)).item()
    print(itos[next2], end="")
    for i in range(100):
        x = torch.tensor([next])
        one = F.one_hot(x, num_classes=54).float()
        one[0, next2 + 27] = 1

        logits = one @ W
        exped = logits.exp()
        probs = exped / exped.sum(1, keepdim=True)
        next = next2
        next2 = torch.multinomial(probs[0], 1, replacement=True).item()
        if next2 == 0:
            break
        else:
            print(itos[next2], end="")
        
    print(end="\n")
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparams
block_size = 8
batch_size = 16
eval_iter = 100
n_embd = 32
num_blocks = 1
dropout = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(1234)
# load dataset
text = open("data/input.txt").read()
# text = text[:1000] # COMMENT THIS AFTERWARDS

# encode and decode functions
chars = sorted(list(set(text)))
vocab_size = len(chars)
encode_dict = {c:i for i,c in enumerate(chars)}
decode_dict = {i:c for i,c in enumerate(chars)}

encode = lambda s: [encode_dict[c] for c in s]
decode = lambda s: "".join([decode_dict[c] for c in s])

# separate training and test set

n_train = int(len(text) * 0.9)
training_set = torch.tensor(encode(list(text[:n_train])))
test_set = torch.tensor(encode(list(text[n_train:])))

# Make a data loader (random ix, then fetches block_size)
def load_batch(set: torch.Tensor):
    idx = torch.randint(high=set.shape[0] - block_size, size=(batch_size,))
    xs = torch.stack([set[i: i+block_size] for i in idx])
    ys = torch.stack([set[i+1: i+1+block_size] for i in idx])
    xs, ys = xs.to(device), ys.to(device)
    return xs, ys

@torch.no_grad()
def estimate_loss(model):
    # get multiple batchs from training set, get mean loss
    # do the same for the test set
    model.eval()

    out = {}
    for set, set_name in zip([training_set, test_set], ["training_set", "test_set"]):
        losses = torch.zeros(eval_iter)
        for i in range(eval_iter):
            xs, ys = load_batch(set)
            _, loss = model(xs, ys)
            losses[i] = loss

        out[set_name] = losses.mean().item()
        
    model.train()
    return out
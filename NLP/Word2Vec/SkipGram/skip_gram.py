import torch
import torch.nn as nn
import torch.optim as optim

class SkipGram(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(SkipGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out = self.out(x)
        return out
        


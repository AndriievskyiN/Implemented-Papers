import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = torch.mean(self.embedding(x), dim=1) # (batch_size, d_model)
        x = self.fc(x) # (batch_size, vocab_size)
        return x
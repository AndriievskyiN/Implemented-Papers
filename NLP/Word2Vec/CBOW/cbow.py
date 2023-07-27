import torch
import torch.nn as nn

class Cbow(nn.Module):
    def __init__(self):
        super().__init__()
        nn.Embedding(10, 300)
        
    def forward(self):
    	pass
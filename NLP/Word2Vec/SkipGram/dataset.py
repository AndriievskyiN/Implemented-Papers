import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, pairs, word_to_idx, unk_token):
        self.pairs = pairs
        self.word_to_idx = word_to_idx
        self.unk_token = unk_token

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        target = self.pairs[index][0]
        context = self.pairs[index][1]

        # Tokenize the text
        target = torch.tensor([self.word_to_idx.get(target, self.unk_token)])
        context = torch.tensor([self.word_to_idx.get(context, self.unk_token)])

        return {
            "target": target,
            "context": context
        }
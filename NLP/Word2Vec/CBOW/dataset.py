import torch
import pandas as pd
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, windows, labels, word_to_idx, unk_token):
        self.windows = windows
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.unk_token = unk_token

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        label = self.labels[index]

        # Tokenize the text
        tokenized_window = torch.tensor([self.word_to_idx.get(word, self.unk_token) for word in window.split()])
        tokenized_label = torch.tensor([self.word_to_idx.get(label, self.unk_token)])


        return {
            "windows": tokenized_window,
            "labels": tokenized_label
        }
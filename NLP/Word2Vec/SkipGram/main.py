import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils import *
from skip_gram import SkipGram
from dataset import TextDataset

def main():
    # Set Hyperparameters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(script_dir, "..", "data", "Articles")
    BATCH_SIZE = 64
    CONTEXT_SIZE = 6
    D_MODEL = 512
    N_EPOCHS = 10
    LEARNING_RATE = 0.0001
    TRAIN = False
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load the Data
    data = get_data(DATA_PATH)

    # Create pairs
    pairs = create_pairs(data, CONTEXT_SIZE)  

    # Load the tokenizer and word2idx and idx2word mappings
    _, word_to_idx, idx_to_word, vocabulary = create_tokenizer(data)
    VOCAB_SIZE = len(vocabulary)

    # Create a dataset
    unk_token = word_to_idx["UNK"]
    train_ds = TextDataset(pairs, word_to_idx, unk_token)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = SkipGram(VOCAB_SIZE, D_MODEL).to(DEVICE)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(), lr=LEARNING_RATE)

    if TRAIN:
        # Train the model
        train_skip_gram(N_EPOCHS, model, train_loader, criterion, optimizer, VOCAB_SIZE, idx_to_word, DEVICE)
    
    # Test the embeddings
    words_to_test = ["man", "woman", "young", "amazing", "strong"]

    for word in words_to_test:
        print(f"WORDS THAT ARE SIMILAR TO: {word}")
        get_most_similar_words(word, model, word_to_idx, idx_to_word, unk_token, n=3)
        print("------------------------------\n")

if __name__ == "__main__":
    main()



import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

import os

def get_data(data_path):
    data = {'File Name': [], 'Text': []}

    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r', encoding='ISO-8859-1') as file:
                        content = file.read()
                        data['File Name'].append(file_name)
                        data['Text'].append(content)

    df = pd.DataFrame(data)

    # Remove the first line (title) from the 'Text' column
    df['Text'] = df['Text'].str.split('\n', 1).str[1]

    # Remove extra spaces and symbols
    df['Text'] = df['Text'].str.replace(r'\n', ' ', regex=True)
    df['Text'] = df['Text'].str.replace(r'[^\w\s]', ' ', regex=True)  # Remove non-alphanumeric characters
    df['Text'] = df['Text'].str.replace(r'\s+', ' ', regex=True)  # Remove extra spaces

    # Remove numbers
    df['Text'] = df['Text'].str.replace(r'\d+', '', regex=True)

    # Lowercase all words
    df['Text'] = df['Text'].str.lower()

    return pd.DataFrame(df["Text"])

def create_tokenizer(data):
    # Create a CountVectorizer instance
    tokenizer = CountVectorizer(lowercase=True)

    # Fit the vectorizer on your text data
    text_data = data['Text'].tolist()
    tokenizer.fit(text_data)

    # Get the vocabulary (list of words) and its corresponding indices
    vocabulary = tokenizer.get_feature_names_out()
    vocabulary = np.append(vocabulary, "UNK")

    # Create word-to-index and index-to-word mappings
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
    idx_to_word = {idx: word for idx, word in enumerate(vocabulary)}

    return tokenizer, word_to_idx, idx_to_word, vocabulary

def create_windows(data, context_size):
    all_windows_str = [] 
    all_labels = []

    for _, row in data.iterrows():
        input_sequence = row["Text"].split()

        num_windows = len(input_sequence) - 2 * context_size

        for i in range(num_windows):
            window = input_sequence[i: i + context_size] + input_sequence[i + context_size + 1: i + 2 * context_size + 1]
            window_str = " ".join(window)  # Convert the window list to a string
            label = input_sequence[i + context_size]
            all_windows_str.append(window_str)  # Append the window string
            all_labels.append(label)

    # Create a pandas DataFrame from the lists
    windows_df = pd.DataFrame({
        'windows': all_windows_str,  
        'labels': all_labels
    })

    return windows_df

def create_pairs(data, context_size):
    pairs = []
    for _, row in data.iterrows():
        input_sequence = row["Text"].split()

        num_windows = len(input_sequence) - 2 * context_size

        for i in range(num_windows):
            window = input_sequence[i: i + context_size] + input_sequence[i + context_size + 1: i + 2 * context_size + 1]
            target = input_sequence[i + context_size]
            
            for word in window:
                pairs.append([target, word])

    return pairs

def train_cbow(n_epochs, model, train_loader, criterion, optimizer, device):
    for epoch in range(n_epochs):
        total_loss = 0

        model.train()
        for batch in tqdm(train_loader, desc=f"EPOCH: {epoch+1} / {n_epochs}", leave=False):
            # Get data from the loader and put it on GPU if available.
            windows = batch["windows"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(windows)

            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")

def train_skip_gram(n_epochs, model, train_loader, criterion, optimizer, vocab_size, idx_to_word, device):
    for epoch in range(n_epochs):
        total_loss = 0

        model.train()
        for batch in tqdm(train_loader, desc=f"EPOCH: {epoch+1} / {n_epochs}", leave=False):
            # Get data from the loader and put it on GPU if available.
            target = batch["target"].to(device)
            context = batch["context"].to(device)

            optimizer.zero_grad()
            outputs = model(target)

            loss = criterion(outputs.view(-1, vocab_size), context.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")
        rand_idx = torch.randint(0, len(batch), (1, )).item()
        target = idx_to_word.get(target[rand_idx].item())
        context = idx_to_word.get(context[rand_idx].item())

        pred_probs = torch.softmax(outputs[rand_idx], dim=1)
        pred_token = torch.argmax(pred_probs)
        predicted_context = idx_to_word.get(pred_token.item())

        print(f"TARGET: {target} | CONTEXT: {context} | PREDICTED: {predicted_context}")

def get_analogy(word1, word2, word3, model, word_to_idx, idx_to_word, unk_token, n=5):
    word1_index = word_to_idx.get(word1, unk_token)
    word2_index = word_to_idx.get(word2, unk_token)
    word3_index = word_to_idx.get(word3, unk_token)

    # Access the embedding layer of your model
    embedding_layer = model.embedding  # Replace with the actual name of your embedding layer

    # Get the embedding vector for the word
    word1_emb = embedding_layer.weight[word1_index]
    word2_emb = embedding_layer.weight[word2_index]
    word3_emb = embedding_layer.weight[word3_index]

    analogy_vector = word1_emb - word2_emb + word3_emb
    analogy_vector_cpu = analogy_vector.cpu().detach()
    word_embeddings_cpu = embedding_layer.weight.cpu().detach()

    # Calculate cosine similarity between the analogy vector and all word embeddings
    similarity_scores = cosine_similarity(analogy_vector_cpu.reshape(1, -1), word_embeddings_cpu)
    
    # Find the indices of the n most similar words
    most_similar_indices = np.argsort(similarity_scores[0])[-n:][::-1]

    # Get the words associated with the most similar indices and their similarity scores
    similar_words = [idx_to_word[idx] for idx in most_similar_indices]
    similar_scores = [similarity_scores[0][idx] for idx in most_similar_indices]

    for word, score in zip(similar_words, similar_scores):
        print(f"Word: {word}, Cosine Similarity: {score:.4f}")

def check_similarity(word1, word2, model, word_to_idx, idx_to_word, unk_token):
    word1_index = word_to_idx.get(word1, unk_token)
    word2_index = word_to_idx.get(word2, unk_token)

    # Access the embedding layer of your model
    embedding_layer = model.embedding

    # Get the embedding vectors for the words
    word1_emb = embedding_layer.weight[word1_index]
    word2_emb = embedding_layer.weight[word2_index]

    # Calculate cosine similarity between the embedding vectors
    similarity_score = cosine_similarity(word1_emb.cpu().detach().reshape(1, -1), word2_emb.cpu().detach().reshape(1, -1))

    print(f"Are '{word1}' and '{word2}' similar?")
    print(f"Cosine Similarity: {similarity_score[0][0]:.4f}")

def get_most_similar_words(input_word, model, word_to_idx, idx_to_word, unk_token, n=5):
    word_index = word_to_idx.get(input_word, unk_token)

    # Access the embedding layer of your model
    embedding_layer = model.embedding

    # Get the embedding vector for the input word
    input_word_emb = embedding_layer.weight[word_index]

    # Calculate cosine similarity between the embedding vector of the input word and all word embeddings
    similarity_scores = cosine_similarity(input_word_emb.cpu().detach().reshape(1, -1), embedding_layer.weight.cpu().detach())

    # Find the indices of the n most similar words
    most_similar_indices = np.argsort(similarity_scores[0])[-n:][::-1]

    # Get the words associated with the most similar indices and their similarity scores
    similar_words = [idx_to_word[idx] for idx in most_similar_indices]
    similar_scores = [similarity_scores[0][idx] for idx in most_similar_indices]

    print(f"Most Similar Words to '{input_word}':")
    for word, score in zip(similar_words, similar_scores):
        print(f"Word: {word}, Cosine Similarity: {score:.4f}")

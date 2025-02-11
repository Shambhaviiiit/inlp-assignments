import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import numpy as np
from tqdm import tqdm
import sys
import random
import re
import os

def create_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append((tokens[i:i+n-1], tokens[i+n-1]))
    return ngrams

def ngrams_to_indices(ngrams, vocab):
    indices = []
    for context, target in ngrams:
        context_indices = [vocab[word] for word in context]
        target_index = vocab[target]
        indices.append((context_indices, target_index))
    return indices

## Feed Forward Neural Network
class FFNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(FFNNLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.window_size = context_size
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((inputs.shape[0], -1))
        out = self.fc1(embeds)
        out = self.relu(out)
        out = self.fc2(out)
        return out

## RNN Language Model
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, inputs):
        # inputs shape: (batch_size, sequence_length)
        embeds = self.embedding(inputs)  # shape: (batch_size, seq_length, embedding_dim)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.rnn.num_layers, inputs.size(0), self.rnn.hidden_size).to(inputs.device)
        # Forward propagate RNN
        output, hn = self.rnn(embeds, h0)
        # Use the hidden state from the final time step to predict the next word.
        last_hidden = output[:, -1, :]  # shape: (batch_size, hidden_dim)
        logits = self.fc(last_hidden)   # shape: (batch_size, vocab_size)
        return logits
    
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, inputs):
        # inputs shape: (batch_size, sequence_length)
        embeds = self.embedding(inputs)  # shape: (batch_size, seq_length, embedding_dim)
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.lstm.num_layers, inputs.size(0), self.lstm.hidden_size).to(inputs.device)
        c0 = torch.zeros(self.lstm.num_layers, inputs.size(0), self.lstm.hidden_size).to(inputs.device)
        # Forward propagate LSTM
        lstm_out, (hn, cn) = self.lstm(embeds, (h0, c0))
        # Use the hidden state from the final time step
        last_hidden = lstm_out[:, -1, :]  # shape: (batch_size, hidden_dim)
        logits = self.fc(last_hidden)      # shape: (batch_size, vocab_size)
        return logits

## Train Model
def train_model(model, train_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        for context, target in tqdm(train_loader):
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

def predict_next_k_words(model, context, vocab, k):
    if isinstance(context, str):
        context_tokens = context.strip().split()
    else:
        context_tokens = context
    # If the model uses a fixed context window (e.g., FFNN), truncate the context tokens.
    if hasattr(model, "window_size"):
        context_tokens = context_tokens[-model.window_size:]
    unk_token = '<unk>'
    context_indices = torch.tensor(
        [vocab[word] if word in vocab else vocab[unk_token] for word in context_tokens],
        dtype=torch.long
    ).unsqueeze(0)
    with torch.no_grad():
        output = model(context_indices)
    probabilities = torch.softmax(output, dim=1)
    top_k_indices = torch.topk(probabilities, k).indices.squeeze(0).tolist()
    idx_to_word = {idx: word for word, idx in vocab.items()}
    return [idx_to_word[idx] for idx in top_k_indices]

# Define or modify the NGramDataset to convert outputs to torch.Tensor

from torch.utils.data import Dataset
import torch

class NGramDataset(Dataset):
    def __init__(self, ngrams):
        self.ngrams = ngrams

    def __len__(self):
        return len(self.ngrams)

    def __getitem__(self, idx):
        context, target = self.ngrams[idx]
        # Convert context (list) and target (single integer) into Tensors.
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# New helper function to clean each sentence.
def clean_sentence(sentence):
    # Lowercase the sentence.
    sentence = sentence.lower()
    # Remove URLs.
    sentence = re.sub(r'https?://\S+', '', sentence)
    # Remove punctuation.
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Remove extra whitespace.
    return sentence.strip()

if __name__ == "__main__":

    # input from arguments: language model type, corpus path, top k words
    language_model = sys.argv[1]
    corpus_path = sys.argv[2]
    top_k = int(sys.argv[3])

    # Read the corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    # Tokenize the corpus into sentences.
    # This splits on common sentence-ending punctuation. Adjust the regex as needed.
    sentences = re.split(r"[.?!]\s+", corpus)
    # Clean and tokenize each sentence into words.
    tokenized_sentences = [clean_sentence(s).split() for s in sentences if s.strip()]

    START = "<s>"
    END = "</s>"
    UNK = "<unk>"

    # Build vocabulary from all words in all sentences.
    all_tokens = [word for sentence in tokenized_sentences for word in sentence]

    # Count token frequencies.
    token_freq = Counter(all_tokens)
    # Define a frequency threshold (e.g., only keep words that appear 2 or more times).
    threshold = 2
    vocab_words = [word for word, count in token_freq.items() if count >= threshold]
    # Ensure special tokens are always included.
    vocab_words.extend([START, END, UNK])
    # Remove duplicates by converting to a set, then back to list.
    vocab_words = list(set(vocab_words))
    # Build word to index mapping.
    vocab = {word: idx for idx, word in enumerate(vocab_words)}
    vocab_size = len(vocab)

    print("Vocab Size:", vocab_size)

    # When building n-grams later, replace words not in the vocabulary with <unk>
    def map_unknowns(sentence):
        return [word if word in vocab else UNK for word in sentence]

    # Create padded tokens and n-grams for each sentence:
    # For 3-grams, add (3-1)=2 start tokens and 1 end token per sentence.
    # For 5-grams, add (5-1)=4 start tokens and 1 end token per sentence.
    ngrams_3 = []
    ngrams_5 = []
    for sentence in tokenized_sentences:
        # Replace rare or unseen words with <unk>
        sentence = map_unknowns(sentence)
        padded_3 = [START] * (3 - 1) + sentence + [END]
        padded_5 = [START] * (5 - 1) + sentence + [END]
        ngrams_3.extend(create_ngrams(padded_3, 3))
        ngrams_5.extend(create_ngrams(padded_5, 5))

    # print("Example 3-grams:", ngrams_3[:10])
    
    # Convert n-grams to indices
    indices_3 = ngrams_to_indices(ngrams_3, vocab)
    indices_5 = ngrams_to_indices(ngrams_5, vocab)

    # Split into train and test sets (test size = 1000) using random shuffling
    
    random.shuffle(indices_3)
    random.shuffle(indices_5)
    test_size = 1000
    test_indices_3 = indices_3[:test_size]
    train_indices_3 = indices_3[test_size:]
    test_indices_5 = indices_5[:test_size]
    train_indices_5 = indices_5[test_size:]

    # Create DataLoaders for training and testing
    train_loader_3 = DataLoader(NGramDataset(train_indices_3), batch_size=64, shuffle=True)
    test_loader_3 = DataLoader(NGramDataset(test_indices_3), batch_size=64, shuffle=False)
    train_loader_5 = DataLoader(NGramDataset(train_indices_5), batch_size=64, shuffle=True)
    test_loader_5 = DataLoader(NGramDataset(test_indices_5), batch_size=64, shuffle=False)

    # Hyperparameters
    embedding_dim = 30
    hidden_dim = 80

    # Function to compute perplexity for a single sentence
    def compute_sentence_perplexity(model, sentence, n, vocab, device='cpu'):
        START = "<s>"
        END = "</s>"
        UNK = "<unk>"
        # Tokenize the sentence (adjust splitting if necessary)
        tokens = sentence.strip().split()
        # Pad tokens: add (n-1) start tokens at beginning and one end token at the end.
        padded_tokens = [START] * (n - 1) + tokens + [END]
        total_log_prob = 0.0
        count = 0
        model.eval()
        with torch.no_grad():
            # Loop over each position that produces an n-gram
            for i in range(n - 1, len(padded_tokens)):
                # context is previous n-1 tokens; target is current token
                context = padded_tokens[i - (n - 1): i]
                target = padded_tokens[i]
                # Use vocab.get with UNK fallback
                context_idxs = torch.tensor(
                    [vocab.get(w, vocab[UNK]) for w in context],
                    dtype=torch.long
                ).unsqueeze(0).to(device)
                logits = model(context_idxs)  # shape: (1, vocab_size)
                log_probs = torch.log_softmax(logits, dim=1)
                target_idx = vocab.get(target, vocab[UNK])
                total_log_prob += log_probs[0, target_idx].item()
                count += 1
        avg_neg_log_prob = -total_log_prob / count
        perplexity = torch.exp(torch.tensor(avg_neg_log_prob))
        return perplexity.item()

    # # Compute perplexity for each sentence in the training corpus.
    # # Here, we assume sentences are separated by periods.
    # raw_sentences = corpus.split('.')
    # # Remove empty sentences and strip extra whitespace.
    # sentences = [s.strip() for s in raw_sentences if s.strip()]

    # Calculate perplexity scores for each model
    def compute_dataset_perplexity(model, dataloader, vocab, device='cpu'):
        model.eval()
        total_log_prob = 0.0
        total_tokens = 0
        with torch.no_grad():
            for contexts, targets in dataloader:
                contexts = contexts.to(device)
                targets = targets.to(device)
                logits = model(contexts)  # shape: (batch_size, vocab_size)
                log_probs = torch.log_softmax(logits, dim=1)  # shape: (batch_size, vocab_size)
                # Gather the log probability corresponding to the correct target for each sample
                batch_log_prob = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                # Compute average loss for the current batch (negative log probability)
                batch_loss = - batch_log_prob.mean().item()
                print(f"Batch Loss: {batch_loss:.4f}")
                total_log_prob += batch_log_prob.sum().item()
                total_tokens += targets.size(0)
        avg_neg_log_prob = - total_log_prob / total_tokens
        perplexity = torch.exp(torch.tensor(avg_neg_log_prob))
        print(f"Overall Average Loss: {avg_neg_log_prob:.4f} | Overall Perplexity: {perplexity.item():.4f}")
        return perplexity.item()
    
    saved_file_3 = ""
    saved_file_5 = ""

    if language_model == "-f":
        # Models
        model_3 = FFNNLanguageModel(vocab_size, embedding_dim, 2, hidden_dim)
        model_5 = FFNNLanguageModel(vocab_size, embedding_dim, 4, hidden_dim)
        saved_file_3 = "ul_model_3_ffnn.pth"
        saved_file_5 = "ul_model_5_ffnn.pth"

    elif language_model == "-r":
        # Models
        model_3 = RNNLanguageModel(vocab_size, embedding_dim, hidden_dim)
        model_5 = RNNLanguageModel(vocab_size, embedding_dim, hidden_dim)
        saved_file_3 = "ul_model_3_rnn.pth"
        saved_file_5 = "ul_model_5_rnn.pth"

    elif language_model == "-l":
        # Models
        model_3 = LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim)
        model_5 = LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim)
        saved_file_3 = "ul_model_3_lstm.pth"
        saved_file_5 = "ul_model_5_lstm.pth"

    # Load saved models if available; otherwise, train and then save.
    if os.path.exists(saved_file_3) and os.path.exists(saved_file_5):
        model_3.load_state_dict(torch.load(saved_file_3, weights_only=True))
        model_5.load_state_dict(torch.load(saved_file_5, weights_only=True))
        print("Loaded saved models.")

        context = input("Enter context words: ")
        context_tokens = context.strip().split()
        predictions_3 = predict_next_k_words(model_3, context_tokens, vocab, top_k)
        predictions_5 = predict_next_k_words(model_5, context_tokens, vocab, top_k)
        print("3-gram LM, Top", top_k, "predicted words:", predictions_3)
        # print("5-gram LM, Top", top_k, "predicted words:", predictions_3)

    else:
        train_model(model_3, train_loader_3, epochs=10)
        train_model(model_5, train_loader_5, epochs=10)
        torch.save(model_3.state_dict(), saved_file_3)
        torch.save(model_5.state_dict(), saved_file_5)
        print("Trained and saved models.")

        # Compute average perplexity on the training dataset for each model.
        avg_pp_3_train = compute_dataset_perplexity(model_3, train_loader_3, vocab)
        avg_pp_5_train = compute_dataset_perplexity(model_5, train_loader_5, vocab)

        print("Average Perplexity for 3-gram LM on training dataset:", avg_pp_3_train)
        print("Average Perplexity for 5-gram LM on training dataset:", avg_pp_5_train)

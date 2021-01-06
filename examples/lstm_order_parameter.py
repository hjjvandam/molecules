from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# This code is adapted from: https://github.com/tiwarylab/LSTM-predict-MD


def running_mean(x, N):
    """Use convolution to do running average."""
    return np.convolve(x, np.ones((N,)) / N, mode="valid")


def find_nearest(key_arr, target):
    """key_arr: array-like, storing keys.
    target: the value which we want to be closest to."""
    idx = np.abs(key_arr - target).argmin()
    return idx


def rm_peaks_steps(traj, threshold: int = 20):
    """
    Remove sudden changes in the trajectory such as peaks and small steps.
    In this method, I used gradient to identify the changes. If two nonzero
    gradients are too close (< threshold), we shall take this range as noise.
    """
    traj = np.array(traj)
    grad_traj = np.gradient(traj)  # gradient of trajectory
    idx_grad = np.where(grad_traj != 0)[0]
    idx0 = idx_grad[0]
    for idx in idx_grad:
        window = idx - idx0
        if window <= 1:  # neighbor
            continue
        elif window > 1 and window <= threshold:
            traj[idx0 : idx0 + window // 2 + 1] = traj[idx0]
            traj[idx0 + window // 2 + 1 : idx + 1] = traj[idx + 1]
            idx0 = idx
        elif window > threshold:
            idx0 = idx
    return traj


def preprocess(data: np.ndarray, X: List[float], smoothing_window: int):
    text = running_mean(data, smoothing_window)  # smooothen data
    text = [find_nearest(X, x) for x in text]  # convert to bins
    text = rm_peaks_steps(text)  # remove peaks and short steps
    return text


class SeqData(Dataset):
    def __init__(self, traj, seq_len, shift):
        self.traj = traj
        self.seq_len = seq_len
        self.shift = shift

    def __len__(self):
        return self.traj[self.shift :].shape[0] // self.seq_len

    def __getitem__(self, idx):
        x = self.traj[: -self.shift][idx * self.seq_len : (idx + 1) * self.seq_len]
        y = self.traj[self.shift :][idx * self.seq_len : (idx + 1) * self.seq_len]
        return x, y


class NLP(nn.Module):
    def __init__(self, input_dim, embedding_dim, rnn_units):

        super(NLP, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, rnn_units, batch_first=True)
        self.linear = nn.Linear(rnn_units, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, hidden = self.lstm(x)
        y_pred = self.linear(x)
        return y_pred


def train(epochs, vocab_size, train_loader, model, optimizer, loss_fn):
    model.train()
    for epoch in range(epochs):

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}\tLoss: {loss.item()}")


def validate(vocab_size, valid_loader, model, loss_fn):
    model.eval()
    y_preds = []
    losses = []
    with torch.no_grad():
        for X, y in valid_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred.view(-1, vocab_size), y.view(-1))
            y_preds.append(y_pred.numpy())
            losses.append(loss.item())

    return y_preds, losses


# infile = "/Users/abrace/tmp/lstm_test/test.txt"
# input_x, _ = np.loadtxt(infile, unpack=True, max_rows=100000)
# print(input_x.shape)
# print(input_x)
# np.save("/Users/abrace/tmp/test.npy", input_x)


epochs = 20
smoothing_window = 20
device = torch.device("cpu")
# data_path = "/Users/abrace/tmp/lstm_test/train_data.npy"
# x-values of the metastable states in the 3-state model potential.
# X = [1.5, 0, -1.5]
data_path = "/Users/abrace/tmp/lstm_test/bba_rmsd.npy"
X = [2.5, 8.0]
# Length of the vocabulary in chars
vocab_size = len(X)
# The embedding dimension
embedding_dim = 8
# Number of RNN units
rnn_units = 32
# Batch size
batch_size = 64
learning_rate = 0.001
# Sequence length and shift in step between past (input) & future (output)
seq_len = 100
shift = 1

data = np.load(data_path)
text = preprocess(data, X, smoothing_window)

# text stores a sequence of integers representing the bins:
# Counter(text) -> Counter({1: 38933, 0: 36007, 2: 25041})

dataset = SeqData(text, seq_len, shift)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

model = NLP(vocab_size, embedding_dim, rnn_units).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(epochs, vocab_size, train_loader, model, optimizer, loss_fn)

y_preds, losses = validate(vocab_size, valid_loader, model, loss_fn)
np.save("/Users/abrace/tmp/lstm_test/bba_y_preds.npy", y_preds)
np.save("/Users/abrace/tmp/lstm_test/bba_losses.npy", losses)

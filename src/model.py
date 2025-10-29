from torch import nn
import torch

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, pad_token):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_token)
        self.lstm = nn.LSTM(input_size= embedding_dim,
                            hidden_size= hidden_dim,
                            num_layers= num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.linear = nn.Linear(2*hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, (h_n,_) = self.lstm(x)

        forward_last_hidden = h_n[-2]  # 正向的hidden
        backward_last_hidden = h_n[-1]  # 反向的hidden
        final_hidden = torch.cat((forward_last_hidden, backward_last_hidden), dim=1)

        output = self.linear(final_hidden).squeeze(1)

        return output

class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, pad_token):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_token)
        self.gru = nn.GRU(input_size= embedding_dim,
                            hidden_size= hidden_dim,
                            num_layers= num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.linear = nn.Linear(2*hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x,h_n  = self.gru(x)

        forward_last_hidden = h_n[-2]  # 正向的hidden
        backward_last_hidden = h_n[-1]  # 反向的hidden
        final_hidden = torch.cat((forward_last_hidden, backward_last_hidden), dim=1)

        output = self.linear(final_hidden).squeeze(1)

        return output
from torch import nn
import torch

def _select_last_hidden(h_n: torch.Tensor, bidirectional: bool) -> torch.Tensor:
    """
    h_n: [num_layers * num_directions, B, H]
    返回: [B, H]（单向）或 [B, 2H]（双向拼接）
    """
    if bidirectional:
        # 取最后一层的正向、反向 hidden（倒数第2和倒数第1）
        last_fwd = h_n[-2]  # [B, H]
        last_bwd = h_n[-1]  # [B, H]
        return torch.cat([last_fwd, last_bwd], dim=-1)  # [B, 2H]
    else:
        return h_n[-1]  # [B, H]


class LSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, pad_token, dropout: float = 0.0):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_token)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.linear = nn.Linear(out_dim, 1)

    def forward(self, x):
        x = self.embedding(x)                            # [B, T, E]
        _, (h_n, _) = self.lstm(x)                       # h_n: [L*num_dir, B, H]
        feat = _select_last_hidden(h_n, self.bidirectional)  # [B, H or 2H]
        logits = self.linear(feat).squeeze(-1)           # [B]
        return logits


class GRU(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, pad_token, dropout: float = 0.0):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_token)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.linear = nn.Linear(out_dim, 1)

    def forward(self, x):
        x = self.embedding(x)                # [B, T, E]
        _, h_n = self.gru(x)                 # h_n: [L*num_dir, B, H]
        feat = _select_last_hidden(h_n, self.bidirectional)  # [B, H or 2H]
        logits = self.linear(feat).squeeze(-1)  # [B]
        return logits


class RNN(nn.Module):
    """可选：简单 RNN 版本，接口一致，方便对比实验"""
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        bidirectional,
        pad_token,
        dropout: float = 0.0
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token
        )
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.linear = nn.Linear(out_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        if self.bidirectional:
            feat = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            feat = h_n[-1]
        logits = self.linear(feat).squeeze(-1)
        return logits

# ===== 工厂函数：在 train.py / predict.py 里用它来创建模型 =====
def build_model(
    arch: str,
    vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    num_layers: int,
    bidirectional: bool,
    pad_token: int,
    dropout: float = 0.0,
) -> nn.Module:
    """
    arch: "GRU" | "LSTM" | "RNN"
    """
    arch = arch.upper()
    if arch == "GRU":
        return GRU(vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, pad_token, dropout)
    elif arch == "LSTM":
        return LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, pad_token, dropout)
    elif arch == "RNN":
        return RNN(vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, pad_token, dropout)
    else:
        raise ValueError(f"Unknown arch: {arch}")
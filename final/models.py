"""Các mô hình: RNN (LSTM/GRU), CNN 1D, MLP.

Mọi mô hình nhận (x, side, mask):
- x: biểu diễn A → (B, 11) giá trị đặc trưng; biểu diễn B → (B, L, 13) các bước lịch sử.
- side: (B, 27) one-hot year/month/weekday, ghép vào trước lớp phân loại.
- mask: (B, L) True = bước thật (chỉ có ở biểu diễn B), None với biểu diễn A.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FeatureTokenizer(nn.Module):
    """Biểu diễn A: đặc trưng thứ i có giá trị x_i → vector x_i·W_i + b_i (W_i, b_i riêng cho từng đặc trưng)."""

    def __init__(self, n_feat: int, d: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_feat, d) / d ** 0.5)
        self.b = nn.Parameter(torch.zeros(n_feat, d))

    def forward(self, x):                    # (B, F) → (B, F, d)
        return x.unsqueeze(-1) * self.W + self.b


def make_embed(rep: str, in_dim: int, d: int) -> nn.Module:
    return FeatureTokenizer(in_dim, d) if rep == "A" else nn.Linear(in_dim, d)   # B: chiếu mỗi bước sang d chiều


def make_head(in_dim: int, hidden: int, dropout: float, K: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, K))


def masked_mean(h, mask):                    # h (B, T, C), mask (B, T)
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1)


class RNNNet(nn.Module):
    def __init__(self, rep, in_dim, side_dim, K, cell="lstm", d=16, hidden=64, layers=1, bidirectional=False,
                 dropout=0.2, pool="last", head=64):
        super().__init__()
        self.embed = make_embed(rep, in_dim, d)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(d, hidden, layers, batch_first=True, bidirectional=bidirectional,
                       dropout=dropout if layers > 1 else 0.0)              # dropout giữa các tầng RNN
        out = hidden * (2 if bidirectional else 1)
        self.pool, self.bidirectional = pool, bidirectional
        if pool == "attn":
            self.att = nn.Linear(out, 1)                                     # attention đơn giản: điểm cho từng bước
        self.head = make_head(out + side_dim, head, dropout, K)

    def forward(self, x, side, mask=None):
        h = self.embed(x)
        if mask is not None:                                                 # chuỗi dài ngắn khác nhau → pack
            lengths = mask.sum(1).cpu()
            out, hn = self.rnn(pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False))
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=h.shape[1])
        else:
            out, hn = self.rnn(h)
            mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        if isinstance(hn, tuple):                                            # LSTM trả về (h_n, c_n)
            hn = hn[0]
        if self.pool == "last":                                              # hidden state cuối của tầng trên cùng
            z = torch.cat([hn[-2], hn[-1]], 1) if self.bidirectional else hn[-1]
        elif self.pool == "mean":
            z = masked_mean(out, mask)
        else:
            a = self.att(out).squeeze(-1).masked_fill(~mask, float("-inf")).softmax(1)
            z = (a.unsqueeze(-1) * out).sum(1)
        return self.head(torch.cat([z, side], 1))


class CNNNet(nn.Module):
    def __init__(self, rep, in_dim, side_dim, K, seq_len, d=16, channels=(32, 64), kernel=3, bn=False,
                 dropout=0.0, pool="max", flatten="gap", head=64):
        super().__init__()
        self.embed = make_embed(rep, in_dim, d)
        P = nn.MaxPool1d if pool == "max" else nn.AvgPool1d
        self.blocks, T, c_in = nn.ModuleList(), seq_len, d
        for c in channels:
            layers = [nn.Conv1d(c_in, c, kernel, padding=kernel // 2)]
            layers += [nn.BatchNorm1d(c)] if bn else []
            layers += [nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
            do_pool = T >= 4                                                 # không pool khi chuỗi đã quá ngắn
            self.blocks.append(nn.ModuleDict({"conv": nn.Sequential(*layers),
                                              "pool": P(2, ceil_mode=True) if do_pool else nn.Identity()}))
            T, c_in = (T + 1) // 2 if do_pool else T, c
        self.flatten = flatten
        out = c_in * T if flatten == "flatten" else c_in
        self.head = make_head(out + side_dim, head, dropout, K)

    def forward(self, x, side, mask=None):
        h = self.embed(x).transpose(1, 2)                                    # (B, d, T): Conv1d trượt theo trục T
        m = mask.unsqueeze(1).float() if mask is not None else None
        for blk in self.blocks:
            h = blk["conv"](h if m is None else h * m)                       # bước đệm luôn = 0
            h = blk["pool"](h)
            if m is not None and not isinstance(blk["pool"], nn.Identity):
                m = nn.functional.max_pool1d(m, 2, ceil_mode=True)
        if self.flatten == "flatten":
            z = h.flatten(1)
        elif m is None:
            z = h.mean(2) if self.flatten == "gap" else h.amax(2)
        elif self.flatten == "gap":
            z = (h * m).sum(2) / m.sum(2).clamp(min=1)
        else:
            z = h.masked_fill(m == 0, float("-inf")).amax(2)
        return self.head(torch.cat([z, side], 1))


class MLPNet(nn.Module):
    """Baseline: các đặc trưng (dạng phẳng, không coi là chuỗi) + side → 2 lớp ẩn."""

    def __init__(self, rep, in_dim, side_dim, K, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        dims, mods = [in_dim + side_dim] + [hidden] * layers, []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*mods, nn.Linear(dims[-1], K))

    def forward(self, x, side, mask=None):
        if x.dim() == 3:                                                     # B: chỉ dùng bước hiện tại (bước cuối)
            x = x[torch.arange(len(x)), mask.sum(1) - 1]
        return self.net(torch.cat([x, side], 1))


def build(cfg: dict, in_dim: int, side_dim: int, K: int, seq_len: int) -> nn.Module:
    kw = {k: v for k, v in cfg.get("params", {}).items()}
    if cfg["model"] == "rnn":
        return RNNNet(cfg["rep"], in_dim, side_dim, K, **kw)
    if cfg["model"] == "cnn":
        return CNNNet(cfg["rep"], in_dim, side_dim, K, seq_len, **kw)
    return MLPNet(cfg["rep"], in_dim, side_dim, K, **kw)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""PyTorch twin of the Rust answer head (crates/vs1-core/src/head.rs).

Any change here must be mirrored there; `vs1 check-head` verifies parity
after every export.

    frames [B,T,D] (unit-norm), question [B,D], options [B,K,D], mask [B,K] → logits [B,K]

logits = a_zs · pooled_similarity(frames, option) + gamma · (z · option)
where z comes from a question-conditioned attention over frames. With
gamma = 0 this is exactly the zero-shot scorer, which is the initial state.
"""

import math

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

EMBED_DIM = 512
ZERO_SHOT_SCALE = 50.0
ZERO_SHOT_POOL_BETA = 10.0


class AnswerHead(nn.Module):
    def __init__(self, dim: int = EMBED_DIM, hidden: int = 256, mlp: int = 512):
        super().__init__()
        self.wq = nn.Linear(dim, hidden)
        self.wk = nn.Linear(dim, hidden)
        self.wv = nn.Linear(dim, hidden)
        self.ln = nn.LayerNorm(hidden, eps=1e-5)
        self.w1 = nn.Linear(hidden, mlp)
        self.w2 = nn.Linear(mlp, hidden)
        self.wz = nn.Linear(hidden, dim)
        self.a_zs = nn.Parameter(torch.tensor([ZERO_SHOT_SCALE]))
        self.gamma = nn.Parameter(torch.tensor([0.0]))
        self.pool_beta = nn.Parameter(torch.tensor([ZERO_SHOT_POOL_BETA]))

    @staticmethod
    def pooled_similarity(frames: torch.Tensor, options: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Log-mean-exp over frames of cosine similarity: [B,K]."""
        sims = beta * torch.einsum("btd,bkd->bkt", frames, options)
        return (torch.logsumexp(sims, dim=-1) - math.log(frames.shape[1])) / beta

    def forward(self, frames, question, options, mask):
        hidden = self.wq.out_features
        qh = self.wq(question)
        keys = (self.wk(frames) * qh[:, None, :]).sum(-1) / math.sqrt(hidden)
        attn = keys.softmax(dim=-1)
        h = self.ln(qh + (attn[..., None] * self.wv(frames)).sum(1))
        h = h + self.w2(F.gelu(self.w1(h), approximate="tanh"))
        z = self.wz(h)
        zero_shot = self.pooled_similarity(frames, options, self.pool_beta)
        logits = self.a_zs * zero_shot + self.gamma * torch.einsum("bd,bkd->bk", z, options)
        return logits.masked_fill(~mask, float("-inf"))

    def export(self, path, name: str) -> None:
        tensors = {
            "wq": self.wq.weight, "bq": self.wq.bias,
            "wk": self.wk.weight, "bk": self.wk.bias,
            "wv": self.wv.weight, "bv": self.wv.bias,
            "ln_g": self.ln.weight, "ln_b": self.ln.bias,
            "w1": self.w1.weight, "b1": self.w1.bias,
            "w2": self.w2.weight, "b2": self.w2.bias,
            "wz": self.wz.weight, "bz": self.wz.bias,
            "a_zs": self.a_zs, "gamma": self.gamma, "pool_beta": self.pool_beta,
        }
        save_file({k: v.detach().contiguous().float() for k, v in tensors.items()}, str(path), metadata={"name": name})

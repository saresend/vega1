"""PyTorch twin of the Rust answer head (crates/vs1-core/src/head.rs).

Any change here must be mirrored there; `vs1 check-head` verifies parity
after every export.

    frames [B,T,D] (unit-norm), question [B,D], options [B,K,D], mask [B,K] → logits [B,K]

The first `n_full` frame rows are full views, the rest crops (with crops=True).

logits = a_zs · s · pooled_similarity(full rows, option) + gamma · (z · option)
       [+ a_crop · (pooled_similarity(crop rows, option), mean(crop rows · option))]
where z comes from a question-conditioned attention over frames and the scale
s = 1 + 2·tanh(ws · x) lets the head trust, discount or (for negated statements)
flip the zero-shot evidence per question. x is the attention output h
(scale_from="hidden", tensors ws/bs) or the question embedding alone
(scale_from="question", tensors wsq/bsq). With gamma = 0 and ws = 0 this is
exactly the zero-shot scorer, which is the initial state.

Crops (tensors a_crop/crop_beta): the two crop terms start at zero; their
learned mix lets the head reward evidence localised in some crops (a small
dog) over evidence spread evenly (generic captions match crops uniformly).
With crop_attention (tensors type_full/type_crop) attention also runs over
the crop rows, each row tagged with a learned full/crop type vector;
without it, attention sees the full views only.
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
    def __init__(self, dim: int = EMBED_DIM, hidden: int = 256, mlp: int = 512, dropout: float = 0.0,
                 scale_from: str = "none", crops: bool = False, crop_attention: bool = True):
        super().__init__()
        self.crops = crops
        self.crop_attention = crops and crop_attention
        if self.crop_attention:
            self.type_full = nn.Parameter(torch.zeros(dim))
            self.type_crop = nn.Parameter(torch.zeros(dim))
        if crops:
            self.a_crop = nn.Parameter(torch.zeros(2))
            self.crop_beta = nn.Parameter(torch.tensor([ZERO_SHOT_POOL_BETA]))
        assert scale_from in ("hidden", "question", "none")
        self.scale_from = scale_from
        self.dropout = nn.Dropout(dropout)  # training only; identity at inference, so Rust needs no twin
        self.wq = nn.Linear(dim, hidden)
        self.wk = nn.Linear(dim, hidden)
        self.wv = nn.Linear(dim, hidden)
        self.ln = nn.LayerNorm(hidden, eps=1e-5)
        self.w1 = nn.Linear(hidden, mlp)
        self.w2 = nn.Linear(mlp, hidden)
        self.wz = nn.Linear(hidden, dim)
        self.ws = nn.Linear(hidden if scale_from == "hidden" else dim, 1)
        nn.init.zeros_(self.ws.weight)
        nn.init.zeros_(self.ws.bias)
        self.a_zs = nn.Parameter(torch.tensor([ZERO_SHOT_SCALE]))
        self.gamma = nn.Parameter(torch.tensor([0.0]))
        self.pool_beta = nn.Parameter(torch.tensor([ZERO_SHOT_POOL_BETA]))

    @staticmethod
    def pooled_similarity(frames: torch.Tensor, options: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Log-mean-exp over frames of cosine similarity: [B,K]."""
        sims = beta * torch.einsum("btd,bkd->bkt", frames, options)
        return (torch.logsumexp(sims, dim=-1) - math.log(frames.shape[1])) / beta

    def forward(self, frames, question, options, mask, n_full=None):
        n_full = frames.shape[1] if n_full is None else n_full
        if not self.crops:
            frames, n_full = frames[:, :n_full], n_full  # crop rows are ignored
        full, crop = frames[:, :n_full], frames[:, n_full:]
        rows = full
        if self.crop_attention:
            rows = torch.cat([full + self.type_full, crop + self.type_crop], dim=1)
        hidden = self.wq.out_features
        qh = self.wq(question)
        keys = (self.wk(rows) * qh[:, None, :]).sum(-1) / math.sqrt(hidden)
        attn = keys.softmax(dim=-1)
        h = self.ln(qh + (attn[..., None] * self.wv(rows)).sum(1))
        h = h + self.w2(F.gelu(self.w1(h), approximate="tanh"))
        h = self.dropout(h)
        z = self.wz(h)
        scale = self.last_scale = self.scale(question, h)  # [B,1]; kept for the training penalty
        zero_shot = self.pooled_similarity(full, options, self.pool_beta)
        logits = self.a_zs * scale * zero_shot + self.gamma * torch.einsum("bd,bkd->bk", z, options)
        if self.crops and crop.shape[1] > 0:
            peak = self.pooled_similarity(crop, options, self.crop_beta)
            mean = torch.einsum("btd,bkd->bk", crop, options) / crop.shape[1]
            logits = logits + self.a_crop[0] * peak + self.a_crop[1] * mean
        return logits.masked_fill(~mask, float("-inf"))

    def scale(self, question, h):
        if self.scale_from == "none":
            return torch.ones_like(question[:, :1])
        return 1.0 + 2.0 * torch.tanh(self.ws(h if self.scale_from == "hidden" else question))

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
        if self.crops:
            tensors.update(a_crop=self.a_crop, crop_beta=self.crop_beta)
        if self.crop_attention:
            tensors.update(type_full=self.type_full, type_crop=self.type_crop)
        if self.scale_from == "hidden":
            tensors.update(ws=self.ws.weight, bs=self.ws.bias)
        elif self.scale_from == "question":
            tensors.update(wsq=self.ws.weight, bsq=self.ws.bias)
        save_file({k: v.detach().contiguous().float() for k, v in tensors.items()}, str(path), metadata={"name": name})

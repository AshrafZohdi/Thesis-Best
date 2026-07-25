"""
Music Transformer (Huang et al. 2018) — PyTorch implementation.

Key innovation over standard Transformer: relative self-attention.
Instead of absolute positional encodings, the model learns to attend
to "how far away" a token is rather than "what position" it is at.
This naturally captures musical repetition and phrase structure.

Reference: "Music Transformer: Generating Music with Long-Term Structure"
           Huang et al., ICLR 2019 (arXiv 1809.04281)
           Relative attention based on Shaw et al. 2018 with efficient skewing.
"""

from __future__ import annotations
import math
import json
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MusicTransformerConfig:
    vocab_size:  int   = 284    # REMI default; set to 526 for EC-REMI
    d_model:     int   = 256    # embedding / attention dimension
    n_heads:     int   = 4      # attention heads  (d_model must be divisible)
    n_layers:    int   = 4      # decoder blocks
    d_ff:        int   = 1024   # feedforward inner dimension
    max_seq_len: int   = 512    # maximum sequence length (= CHUNK_SIZE)
    dropout:     float = 0.1


class RelativeAttention(nn.Module):
    """
    Causal relative multi-head self-attention.

    S_rel[i, j] = (Q[i] · K[j] + Q[i] · E[i-j]) / sqrt(d_k)

    where E[r] is a learned embedding for relative distance r.
    The upper triangle (j > i) is masked for autoregressive generation.

    Efficient O(L^2 d) computation uses the "skewing" trick:
      1. Compute Q @ E_rev^T  (E in reversed-distance order)
      2. Pad one zero column on the left
      3. Reshape and slice — the diagonal lines up with the correct distances
      4. Upper triangle contains garbage but is masked out anyway
    """

    def __init__(self, cfg: MusicTransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k       = cfg.d_model // cfg.n_heads
        self.n_heads   = cfg.n_heads
        self.max_seq_len = cfg.max_seq_len

        self.W_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # E[r] = embedding for relative distance r (0 = self, 1 = one step back, ...)
        self.E = nn.Embedding(cfg.max_seq_len, self.d_k)

    @staticmethod
    def _skew(qe: torch.Tensor) -> torch.Tensor:
        """
        Convert Q @ E_rev^T (shape B×H×L×L) into relative attention logits
        where output[b, h, i, j] = Q[b,h,i] · E[i-j].

        Upper triangle is garbage — apply causal mask after this.
        """
        B, H, L, _ = qe.shape
        qe = F.pad(qe, (1, 0))         # (B, H, L, L+1)  — pad one zero column left
        qe = qe.view(B, H, L + 1, L)  # reshape
        return qe[:, :, 1:]            # (B, H, L, L)     — drop first row

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        def split_heads(t):
            return t.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_q(x))   # (B, H, L, d_k)
        K = split_heads(self.W_k(x))
        V = split_heads(self.W_v(x))

        scale = math.sqrt(self.d_k)

        # ── Content attention: Q · K^T ────────────────────────────────────
        content = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, L, L)

        # ── Relative attention: Q · E^T with skewing ─────────────────────
        # Reversed positions so that after skewing, [i,j] holds distance i-j
        rev_pos = torch.arange(L - 1, -1, -1, device=x.device).clamp(max=self.max_seq_len - 1)
        E_rev   = self.E(rev_pos)                                # (L, d_k)
        qe      = torch.matmul(Q, E_rev.T) / scale              # (B, H, L, L)
        relative = self._skew(qe)                                # (B, H, L, L)

        # ── Combine, mask, softmax ────────────────────────────────────────
        logits = (content + relative).masked_fill(causal_mask, float('-inf'))
        attn   = self.drop(F.softmax(logits, dim=-1))

        out = torch.matmul(attn, V)                              # (B, H, L, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)   # (B, L, d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: MusicTransformerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MusicTransformerBlock(nn.Module):
    def __init__(self, cfg: MusicTransformerConfig):
        super().__init__()
        self.attn = RelativeAttention(cfg)
        self.ff   = FeedForward(cfg)
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.ln2  = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), causal_mask)   # pre-norm residual
        x = x + self.ff(self.ln2(x))
        return x


class MusicTransformer(nn.Module):
    """
    Full Music Transformer decoder.

    Usage
    -----
    >>> cfg   = MusicTransformerConfig(vocab_size=284)
    >>> model = MusicTransformer(cfg)
    >>> ids   = torch.randint(0, 284, (2, 512))
    >>> logits, loss = model(ids, targets=ids)
    """

    def __init__(self, cfg: MusicTransformerConfig):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop   = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([MusicTransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f   = nn.LayerNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: output projection shares embedding matrix
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1 and 'embed' not in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(
        self,
        ids:     torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, L = ids.shape
        device = ids.device

        # Build causal mask once per forward pass: True = masked
        causal_mask = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1
        ).unsqueeze(0).unsqueeze(0)   # (1, 1, L, L)

        x = self.drop(self.embed(ids))                    # (B, L, d_model)
        for block in self.blocks:
            x = block(x, causal_mask)
        x      = self.ln_f(x)
        logits = self.head(x)                             # (B, L, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt:         torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 0.92,
        top_p:          float = 0.92,
        rep_penalty:    float = 1.0,
        rep_window:     int   = 64,
    ) -> torch.Tensor:
        """Autoregressive generation with nucleus (top-p) sampling and repetition penalty."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len context window
            ctx    = prompt if prompt.size(1) <= self.cfg.max_seq_len else prompt[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature         # (B, vocab_size)

            # Repetition penalty — divide logits of recently seen tokens
            if rep_penalty > 1.0 and prompt.size(1) > 0:
                recent = prompt[0, -rep_window:].tolist()
                for tok_id in set(recent):
                    logits[0, tok_id] = logits[0, tok_id] / rep_penalty

            # Nucleus sampling
            probs  = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = (cumulative - sorted_probs) > top_p
            sorted_probs[remove] = 0.0
            sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True) + 1e-9)

            next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            prompt = torch.cat([prompt, next_token], dim=1)
        return prompt

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def resize_vocab(self, new_vocab_size: int):
        """Expand vocabulary (for EC-REMI fine-tuning on a REMI-pretrained model)."""
        old_size = self.cfg.vocab_size
        if new_vocab_size == old_size:
            return
        old_embed = self.embed.weight.data
        new_embed = nn.Embedding(new_vocab_size, self.cfg.d_model)
        nn.init.xavier_uniform_(new_embed.weight)
        new_embed.weight.data[:old_size] = old_embed
        self.embed = new_embed
        self.head  = nn.Linear(self.cfg.d_model, new_vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.cfg.vocab_size = new_vocab_size

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / 'model.pt')
        with open(path / 'config.json', 'w') as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=2)

    @classmethod
    def load(cls, path: str | Path, map_location: str = 'cpu') -> 'MusicTransformer':
        path = Path(path)
        with open(path / 'config.json') as f:
            cfg = MusicTransformerConfig(**json.load(f))
        model = cls(cfg)
        model.load_state_dict(torch.load(path / 'model.pt', map_location=map_location))
        return model

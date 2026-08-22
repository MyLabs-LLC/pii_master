"""The student: a dilated depthwise-separable CNN tagger.

Why not a transformer. The serving budget is 1 CPU core. A transformer body
costs ~12*L*d^2 MACs per token, so even ettin-17m (~5.5M MACs/token) can only
process ~11 tokens inside the 4 ms residual budget -- useless for a 2,000-token
document. A depthwise-separable conv layer costs ~d^2 + k*d MACs per token,
roughly 100x less, which is the only shape that can read a whole document in
budget. See docs/DISTILLATION_PLAN.md section 2 for the arithmetic.

Receptive field comes from dilation, not attention: with k=5 and dilations
1,2,4,8 the field is 61 tokens, which comfortably covers the cue->value
distances PII detection actually needs (the rules tier uses a 40-char window).

Every other design choice here was settled by profiling ONNX Runtime on one
core, not by taste (numbers in docs/DISTILLATION_PLAN.md section 3):

  * BatchNorm, not LayerNorm. BatchNorm folds into the preceding convolution
    at inference and costs nothing; LayerNorm plus its SkipLayerNorm fusion
    measured 30% of total runtime.
  * ReLU, not GELU. Erf alone measured 10.8% of runtime, and its Mul/Div
    tail added more.
  * Channel-first (B, D, T) end to end, with the head as a 1x1 convolution.
    The per-block transposes of a channel-last design measured 7.4%.

Together those took a 2,000-token forward pass from 6.69 ms to 2.76 ms on a
single 2.8 GHz Xeon core -- a 2.4x speedup with identical FLOPs.

The student reuses the TEACHER'S TOKENIZER. That is deliberate: identical
tokenization makes teacher and student logits align 1:1 per token, so
distillation needs no alignment logic at all. The cost is a large embedding
table, but an embedding is a lookup -- it costs memory, not MACs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


@dataclass
class StudentConfig:
    vocab_size: int = 50368        # ModernBERT/ettin tokenizer
    num_labels: int = 111          # 55 types x BI + O
    d_model: int = 96
    n_layers: int = 4
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1
    pad_id: int = 50283

    def macs_per_token(self) -> int:
        """Body cost per token; drives the latency budget."""
        per_layer = self.d_model * self.d_model + self.kernel_size * self.d_model
        return self.n_layers * per_layer + self.d_model * self.num_labels

    def to_dict(self) -> dict:
        return asdict(self)


class ConvBlock(nn.Module):
    """Depthwise separable conv + BatchNorm + ReLU, residual. Channel-first."""

    def __init__(self, d: int, k: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.depthwise = nn.Conv1d(d, d, k, padding=pad, dilation=dilation, groups=d)
        self.pointwise = nn.Conv1d(d, d, 1)
        self.norm = nn.BatchNorm1d(d)          # folds into pointwise at inference
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                      # x: (B, D, T) in and out
        h = self.pointwise(self.depthwise(x))
        return x + self.drop(self.act(self.norm(h)))


class StudentTagger(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.embed_norm = nn.BatchNorm1d(cfg.d_model)
        dil = cfg.dilations or tuple(2 ** i for i in range(cfg.n_layers))
        self.blocks = nn.ModuleList([
            ConvBlock(cfg.d_model, cfg.kernel_size, dil[i % len(dil)], cfg.dropout)
            for i in range(cfg.n_layers)
        ])
        self.head = nn.Conv1d(cfg.d_model, cfg.num_labels, 1)

    def forward(self, input_ids, attention_mask=None):
        # One transpose in, one out; everything between stays channel-first.
        x = self.embed_norm(self.embed(input_ids).transpose(1, 2))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(1).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        return self.head(x).transpose(1, 2)    # (B, T, num_labels)

    def num_parameters(self, body_only: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if body_only:
            total -= self.embed.weight.numel()
        return total


# The size ladder. Pick by measurement on the target CPU, not by preference.
LADDER: dict[str, StudentConfig] = {
    "xs": StudentConfig(d_model=64, n_layers=4),
    "s":  StudentConfig(d_model=96, n_layers=4),
    "m":  StudentConfig(d_model=128, n_layers=6, dilations=(1, 2, 4, 8, 16, 32)),
}


if __name__ == "__main__":
    for name, cfg in LADDER.items():
        model = StudentTagger(cfg)
        macs = cfg.macs_per_token()
        # 2,000 tokens ~= a 10 KB English document.
        for gmacs in (8, 15, 30):
            ms = 2000 * macs / (gmacs * 1e9) * 1000
            print(f"{name:>3} d={cfg.d_model:<4} L={cfg.n_layers}  "
                  f"params={model.num_parameters()/1e6:5.2f}M "
                  f"(body {model.num_parameters(True)/1e6:4.2f}M)  "
                  f"{macs/1000:6.1f}k MACs/tok  "
                  f"@{gmacs}G/s -> {ms:6.2f} ms/10KB")
        print()

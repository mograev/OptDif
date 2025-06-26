"""
Hierarchical Transformer prior for LatentVQVAE-2 style models.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR

from src.models.latent_models import LatentVQVAE2


class ARTransformer(nn.Module):
    """
    Standard GPT-style causal Transformer over a fixed-length token sequence.
    """
    def __init__(
        self,
        vocab_size,
        seq_len,
        d_model=512,
        n_layers=8,
        n_heads=8,
        dropout=0.0,
    ):
        """
        Initialize the Transformer with the given parameters.
        Args:
            vocab_size: Number of discrete codes (including any offset).
            seq_len: Maximum sequence length (fixed grid length in VQ-VAE latents).
            d_model: Embedding / hidden size.
            n_layers: Number of TransformerEncoder layers.
            n_heads: Attention heads in each layer.
            dropout: Dropout rate for the Transformer layers.
        """
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.tr = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # causal mask cached on first forward
        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1),
            persistent=False,
        )

    def forward(self, idx):
        """
        Forward pass through the Transformer.
        Args:
            idx: Token indices (B, L).
        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = idx.shape
        if L > self.seq_len:
            raise ValueError(f"sequence length {L} > configured {self.seq_len}")

        pos = torch.arange(L, device=idx.device, dtype=torch.long).expand(B, L)
        h = self.tok_emb(idx) + self.pos_emb(pos)
        h = self.tr(h, mask=self._causal_mask[:L, :L])
        return self.head(h)

    @torch.no_grad()
    def generate(self, prefix, temperature=1.0, top_k=None):
        """
        Sampling helper for autoregressive generation.
        Autoregressively complete a sequence until `self.seq_len`.
        Args:
            prefix: Initial sequence to start generation from (B, <=seq_len).
            temperature: Softmax temperature for sampling.
            top_k: If specified, only sample from the top-k logits.
        Returns:
            seq: Generated sequence of indices.
        """
        B, cur = prefix.shape
        device = prefix.device
        seq = torch.full(
            (B, self.seq_len), fill_value=0, dtype=torch.long, device=device
        )
        seq[:, :cur] = prefix

        for t in range(cur, self.seq_len):
            logits = self(seq[:, : t + 1])[:, -1] / temperature  # (B, vocab)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).squeeze(-1)
            seq[:, t] = next_tok

        return seq



class GridTransformer(nn.Module):
    """
    GPT-style causal Transformer that leverages 2D grid positional embeddings.
    Processes flattened grid tokens (H*W) but uses separate row/column embeddings.
    """
    def __init__(
        self,
        vocab_size: int,
        height: int,
        width: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        """
        Initialize the GridTransformer.
        Args:
            vocab_size: Number of discrete tokens (e.g. VQ codes).
            height: Height of the grid.
            width: Width of the grid.
            d_model: Embedding / hidden size.
            n_layers: Number of TransformerEncoder layers.
            n_heads: Number of attention heads in each layer.
            dropout: Dropout rate for the Transformer layers.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.seq_len = height * width
        # token and 2D positional embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.row_emb = nn.Embedding(height, d_model)
        self.col_emb = nn.Embedding(width, d_model)

        # transformer encoder stack
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.tr = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # causal mask over flattened sequence
        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.ones(self.seq_len, self.seq_len) * float("-inf"), diagonal=1),
            persistent=False,
        )

    def forward(self, idx: torch.LongTensor):
        """
        Forward pass.
        Args:
            idx: Flattened grid token indices, shape (B, seq_len).
        Returns:
            logits: Shape (B, seq_len, vocab_size).
        """
        B, L = idx.shape
        if L > self.seq_len:
            raise ValueError(f"sequence length {L} > configured {self.seq_len}")

        # embed tokens (flattened)
        tok = self.tok_emb(idx)  # (B, L, D)

        # compute 2D positional embeddings for first L positions
        positions = torch.arange(L, device=idx.device)
        rows = positions // self.width
        cols = positions % self.width
        pos_flat = self.row_emb(rows) + self.col_emb(cols)  # (L, D)

        # combine token + pos, apply transformer with causal mask
        h = tok + pos_flat.unsqueeze(0)  # (B, L, D)
        h = self.tr(h, mask=self._causal_mask[:L, :L])

        return self.head(h)

    @torch.no_grad()
    def generate(self, prefix: torch.LongTensor, temperature: float = 1.0, top_k: int | None = None):
        """
        Autoregressive sampling for GridTransformer.
        Args:
            prefix: Initial sequence of token indices, shape (B, cur_len).
            temperature: Softmax temperature.
            top_k: If specified, use top-k sampling.
        Returns:
            seq: Generated full sequence of shape (B, self.seq_len).
        """
        B, cur = prefix.shape
        device = prefix.device
        # Initialize sequence with zeros
        seq = torch.zeros((B, self.seq_len), dtype=torch.long, device=device)
        seq[:, :cur] = prefix
        # Autoregressively generate tokens
        for t in range(cur, self.seq_len):
            logits = self(seq[:, : t + 1])[:, -1] / temperature  # (B, vocab)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).squeeze(-1)
            seq[:, t] = next_tok
        return seq


class HierarchicalTransformerPrior(pl.LightningModule):
    """
    Two-stage Transformer prior (top-level + bottom-level) for LatentVQVAE-2.
    """
    def __init__(
        self,
        vqvae : LatentVQVAE2,
        d_model=512,
        n_layers=8,
        n_heads=8,
        lr=3e-4,
        weight_decay=0.0,
    ):
        """
        Initialize the Hierarchical Transformer prior.
        Args:
            vqvae: Pre-trained VQ-VAE-2 model (encoder/decoder).
            d_model: Embedding / hidden size for the Transformer.
            n_layers: Number of Transformer layers.
            n_heads: Number of attention heads in each layer.
            lr: Learning rate for the optimizer.
            weight_decay: Weight decay for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["vqvae"])

        # Freeze the trained VQ-VAE-2 encoder/decoder
        self.vqvae = vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False

        # Sizes & Offsets
        self.top_vocab = self.vqvae.quantize_top.n_e
        self.bot_vocab = self.vqvae.quantize_bottom.n_e
        self.bottom_offset = self.top_vocab  # shift bottom indices by this
        
        top_res = self.vqvae.latent_dim_top[0]
        bot_res = self.vqvae.latent_dim_bottom[0]

        self.top_len = top_res * top_res
        self.bot_len = bot_res * bot_res
        self.seq_len_bot = self.top_len + self.bot_len

        # Setup AR Transformers
        # Top-level Transformer over top-level codes (fixed length)
        # self.top_prior = ARTransformer(
        #     vocab_size=self.top_vocab,
        #     seq_len=self.top_len,
        #     d_model=d_model,
        #     n_layers=n_layers,
        #     n_heads=n_heads,
        # )
        self.top_prior = GridTransformer(
            vocab_size=self.top_vocab,
            height=top_res,
            width=top_res,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
        )

        # Bottom-level Transformer over full sequence (top + bottom codes)
        self.bot_prior = ARTransformer(
            vocab_size=self.top_vocab + self.bot_vocab,
            seq_len=self.seq_len_bot,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
        )

        self.lr = lr
        self.weight_decay = weight_decay

    @torch.no_grad()
    def encode_to_indices(self, x):
        """
        Helper to encode input `x` to top and bottom indices.
        Args:
            x: Input tensor (B, C, H, W) in pixel space.
        Returns:
            top_idx: Indices for top-level codes (B, top_len).
            bot_idx: Indices for bottom-level codes (B, bot_len).
        """
        _, _, _, flat_idx = self.vqvae.encode(x)          # (B, top+bot)
        # some VQ‑VAEs use -1 for "dead" codes – map them to 0
        flat_idx = torch.clamp_min(flat_idx, 0)

        # Split the flat indices into top and bottom parts
        top_idx = flat_idx[:, : self.top_len]
        bot_idx = flat_idx[:, self.top_len :]

        # sanity check: clamp to valid id range
        top_idx = top_idx.clamp_max(self.top_vocab - 1)
        bot_idx = bot_idx.clamp_max(self.bot_vocab - 1)

        return top_idx.long(), bot_idx.long()

    def forward(self, x):
        """
        Forward pass that computes the loss for the input `x`.
        Args:
            x: Input tensor (B, C, H, W) in pixel space.
        Returns:
            loss: Computed loss value.Wh
        """
        top_idx, bot_idx = self.encode_to_indices(x)

        # compute histogram over the whole batch
        flat = top_idx.flatten()
        counts = torch.bincount(flat, minlength=self.top_vocab).float()
        freqs = counts / counts.sum()

        # fraction of clamped zeros (dead codes mapped to 0)
        dead_frac = (flat == 0).float().mean()

        # log to TensorBoard
        self.log("dist/top_dead_frac", dead_frac, prog_bar=False)
        self.logger.experiment.add_histogram("dist/top_code_freqs", freqs.cpu(), self.global_step)

        # -- Top loss --------------------------------------------- #
        inp_top = top_idx[:, :-1]  # teacher-forced input (B, top_len-1)
        tgt_top = top_idx[:, 1:]  # target (B, top_len-1)
        logits_top = self.top_prior(inp_top)
        loss_top = F.cross_entropy(
            logits_top.reshape(-1, logits_top.size(-1)),
            tgt_top.reshape(-1),
        )

        # -- Bottom loss ------------------------------------------ #
        bot_idx_off = bot_idx + self.bottom_offset
        full_seq = torch.cat([top_idx, bot_idx_off], dim=1)  # (B, seq_len_bot)

        inp_bot = full_seq[:, :-1]
        tgt_bot = full_seq[:, 1:]

        logits_bot = self.bot_prior(inp_bot)  # (B, L-1, V_tot)

        # ignore the first top_len targets that belong to the prefix
        mask = torch.ones_like(tgt_bot, dtype=torch.bool)
        mask[:, : self.top_len - 1] = False

        loss_bot = F.cross_entropy(
            logits_bot.reshape(-1, logits_bot.size(-1)),
            tgt_bot.reshape(-1),
            reduction="none",
        )
        loss_bot = (loss_bot * mask.reshape(-1)).sum() / mask.sum()

        # -- Total loss ------------------------------------------- #
        loss = loss_top + loss_bot
        return loss, {"total_loss": loss, "loss_top": loss_top, "loss_bottom": loss_bot}

    # Lightning API
    def training_step(self, batch, _):
        loss, logs = self(batch)
        logs = {f"train/{k}": v for k, v in logs.items()}
        self.log_dict(logs, on_step=True, on_epoch=False, sync_dist=True)
        return loss


    def validation_step(self, batch, _):
        loss, logs = self(batch)
        logs = {f"val/{k}": v for k, v in logs.items()}
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def setup(self, stage=None):
        if stage == "fit":
            self.total_steps = self.trainer.estimated_stepping_batches

    def configure_optimizers(self):
        # Separate parameters into those with and without weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or "LayerNorm.weight" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

        def lr_lambda(step):
            if step < 1_000:
                return float(step) / 1_000
            return 0.5 * (
                1 + math.cos(math.pi * (step - 1_000) / (self.total_steps - 1_000))
            )

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    @torch.no_grad()
    def sample(self, n, temperature=1.0, top_k=None):
        """
        Generate `n` images (decoded to pixel/latent space of VQ-VAE).
        Args:
            n: Number of samples to generate.
            temperature: Softmax temperature for sampling.
            top_k: If specified, only sample from the top-k logits.
        Returns:
            imgs: Generated images (B, C, H, W) in pixel space.
        """
        device = next(self.parameters()).device
        # -- 1) sample top codes ---------------------------------- #
        prefix_top = torch.zeros((n, 1), dtype=torch.long, device=device)
        top_seq = self.top_prior.generate(prefix_top, temperature, top_k)  # (n, top_len)
        # Clamp to handle out-of-range tokens
        top_seq = top_seq.clamp(0, self.top_vocab - 1)

        # -- 2) sample bottom codes ------------------------------- #
        # start with full prefix = top_seq
        bot_prefix = top_seq
        full_seq = self.bot_prior.generate(bot_prefix, temperature, top_k)  # (n, seq_len_bot)
        bot_seq_off = full_seq[:, self.top_len :]  # bottom (offset still applied)

        # Valid bottom tokens are in [bottom_offset, bottom_offset + bot_vocab ‑ 1]
        bot_seq_off = bot_seq_off.clamp(self.bottom_offset, self.bottom_offset + self.bot_vocab - 1)

        bot_seq = bot_seq_off - self.bottom_offset  # shift back to [0, bot_vocab ‑ 1]

        # -- 3) decode -------------------------------------------- #
        imgs = self.vqvae.decode_code(code_t=top_seq, code_b=bot_seq)
        return imgs.clamp(-1, 1)  # match VQ-VAE output range


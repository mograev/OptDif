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
from src.models.vqvae2 import VQVAE2


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
        include_bos: bool = False,
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
            include_bos: Whether to include an explicit BOS token.
        """
        super().__init__()
        # Optional explicit beginning‑of‑sequence (BOS) token
        self.include_bos = include_bos
        if include_bos:
            self.bos_idx = vocab_size          # reserve last index for BOS
            vocab_size += 1                    # expand vocabulary
        else:
            self.bos_idx = None
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
    def generate(self, prefix, temperature=1.0, top_k=None, seed=None):
        """
        Sampling helper for autoregressive generation.
        Autoregressively complete a sequence until `self.seq_len`.
        Args:
            prefix: Initial sequence to start generation from (B, <=seq_len).
            temperature: Softmax temperature for sampling.
            top_k: If specified, only sample from the top-k logits.
            seed: Optional random seed for reproducibility.
        Returns:
            seq: Generated sequence of indices.
        """
        B, cur = prefix.shape
        device = prefix.device
        seq = torch.full(
            (B, self.seq_len), fill_value=0, dtype=torch.long, device=device
        )
        seq[:, :cur] = prefix

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        for t in range(cur, self.seq_len):
            logits = self(seq[:, : t + 1])[:, -1] / temperature  # (B, vocab)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
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
    def generate(self, prefix: torch.LongTensor, temperature: float = 1.0, top_k: int | None = None, seed: int | None = None):
        """
        Autoregressive sampling for GridTransformer.
        Args:
            prefix: Initial sequence of token indices, shape (B, cur_len).
            temperature: Softmax temperature.
            top_k: If specified, use top-k sampling.
            seed: Optional random seed for reproducibility.
        Returns:
            seq: Generated full sequence of shape (B, self.seq_len).
        """
        B, cur = prefix.shape
        device = prefix.device

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

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
            next_tok = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            seq[:, t] = next_tok
        return seq


class HierarchicalTransformerPrior(pl.LightningModule):
    """
    Two-stage Transformer prior (top-level + bottom-level) for LatentVQVAE-2 or VQVAE-2.
    """
    def __init__(
        self,
        vqvae : LatentVQVAE2 | VQVAE2,
        d_model=512,
        n_layers=8,
        n_heads=8,
        lr=3e-4,
        weight_decay=0.0,
        freeze_bottom_steps=5_000,
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
        self.seq_len_bot = self.top_len + self.bot_len + 1  # +1 for BOS

        # When we use a GridTransformer for the bottom prior the sequence length
        # must be a perfect H×W grid.  We embed the BOS + all top‑level tokens
        # as a *prefix* that occupies a few extra rows at the top of the grid.
        self.extra_rows = math.ceil((self.top_len + 1) / bot_res)  # rows needed for prefix
        self.height_bot = bot_res + self.extra_rows                # total grid height
        # full sequence length seen by the bottom prior
        self.seq_len_bot_grid = self.height_bot * bot_res

        # ── Transformer priors with explicit BOS tokens ─────────────────── #
        self.top_prior = ARTransformer(
            vocab_size=self.top_vocab,
            seq_len=self.top_len + 1,        # +1 to accommodate BOS
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            include_bos=True,
        )

        # Bottom‑level prior: 2‑D GridTransformer that still sees the prefix tokens.
        self.bot_prior = GridTransformer(
            vocab_size=self.top_vocab + self.bot_vocab,
            height=self.height_bot,
            width=bot_res,
            d_model=int(d_model*1.5),
            n_layers=n_layers+4,
            n_heads=n_heads*2//3,
            dropout=0.0,
        )

        self.lr = lr
        self.freeze_bottom_steps = freeze_bottom_steps
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
            loss: Computed loss value.
        """
        top_idx, bot_idx = self.encode_to_indices(x)

        # compute histogram over the whole batch
        flat = top_idx.flatten()
        counts = torch.bincount(flat, minlength=self.top_vocab).float()
        freqs = counts / counts.sum()

        # fraction of clamped zeros (dead codes mapped to 0)
        dead_frac = (flat == 0).float().mean()

        # log to TensorBoard
        self.log("dist/top_dead_frac", dead_frac, prog_bar=False, sync_dist=True)
        self.logger.experiment.add_histogram("dist/top_code_freqs", freqs.cpu(), self.global_step)

        # do the same for bottom indices
        flat_bot = bot_idx.flatten()
        counts_bot = torch.bincount(flat_bot, minlength=self.bot_vocab).float()
        freqs_bot = counts_bot / counts_bot.sum()
        dead_frac_bot = (flat_bot == 0).float().mean()
        self.log("dist/bot_dead_frac", dead_frac_bot, prog_bar=False, sync_dist=True)
        self.logger.experiment.add_histogram("dist/bot_code_freqs", freqs_bot.cpu(), self.global_step)

        # -- Top loss --------------------------------------------- #
        bos_tok = torch.full((top_idx.size(0), 1),
                             self.top_prior.bos_idx,
                             dtype=torch.long,
                             device=top_idx.device)
        inp_top = torch.cat([bos_tok, top_idx[:, :-1]], dim=1)  # (B, top_len)
        tgt_top = top_idx                                        # (B, top_len)
        logits_top = self.top_prior(inp_top)
        loss_top = F.cross_entropy(
            logits_top.reshape(-1, logits_top.size(-1)),
            tgt_top.reshape(-1),
        )

        # -- Bottom loss ------------------------------------------ #
        bot_idx_off = bot_idx + self.bottom_offset
        bos_bot = torch.full((bot_idx_off.size(0), 1),
                             0,  # will be BOS row in the grid (token 0)
                             dtype=torch.long,
                             device=bot_idx_off.device)
        full_seq = torch.cat([bos_bot, top_idx, bot_idx_off], dim=1)         # (B, top_len+1+bot_len)
        # Pad so it fits the H×W grid expected by GridTransformer
        pad_len = self.seq_len_bot_grid - full_seq.size(1)
        if pad_len > 0:
            full_seq = F.pad(full_seq, (0, pad_len), value=0)               # pad with dummy token 0

        inp_bot = full_seq[:, :-1]
        tgt_bot = full_seq[:, 1:]

        logits_bot = self.bot_prior(inp_bot)  # (B, L-1, V_tot)

        # valid targets = bottom tokens only (ignore prefix + padding)
        mask = torch.zeros_like(tgt_bot, dtype=torch.bool)
        start_bot = self.top_len                                  # skip BOS + top codes
        end_bot   = start_bot + self.bot_len
        mask[:, start_bot:end_bot] = True

        loss_bot = F.cross_entropy(
            logits_bot.reshape(-1, logits_bot.size(-1)),
            tgt_bot.reshape(-1),
            reduction="none",
        )
        loss_bot = (loss_bot * mask.reshape(-1)).sum() / mask.sum()

        # -- Total loss ------------------------------------------- #
        loss = loss_top + 4 * loss_bot
        return loss, {"total_loss": loss, "loss_top": loss_top, "loss_bottom": loss_bot}

    # Lightning API
    def training_step(self, batch, _):
        loss, logs = self(batch)
        logs = {f"train/{k}": v for k, v in logs.items()}
        logs["train/bottom_frozen"] = float(self.global_step < self.freeze_bottom_steps)
        self.log_dict(logs, on_step=True, on_epoch=False, sync_dist=True)
        return loss
    
    def on_train_start(self) -> None:
        # Freeze bottom prior parameters until the scheduled number of steps has passed
        for p in self.bot_prior.parameters():
            p.requires_grad_(False)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx: int = 0) -> None:
        # Un‑freeze once the scheduled number of steps has passed
        if self.global_step == self.freeze_bottom_steps:
            for p in self.bot_prior.parameters():
                p.requires_grad_(True)

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
    def sample(self, n, temperature=1.0, top_k=None, seed=None):
        """
        Generate `n` images (decoded to pixel/latent space of VQ-VAE).
        Args:
            n: Number of samples to generate.
            temperature: Softmax temperature for sampling.
            top_k: If specified, only sample from the top-k logits.
            seed: Optional random seed for reproducibility.
        Returns:
            imgs: Generated images (B, C, H, W) in pixel space.
        Note: bottom-level codes are sampled on a grid, with BOS+top codes as a prefix occupying extra rows.
        """
        device = next(self.parameters()).device
        # -- 1) sample top codes ---------------------------------- #
        bos_top = torch.full((n, 1),
                             self.top_prior.bos_idx,
                             dtype=torch.long,
                             device=device)
        top_seq_full = self.top_prior.generate(bos_top, temperature, top_k, seed)  # (n, top_len+1)
        top_seq = top_seq_full[:, 1:]  # drop BOS
        # Clamp to handle out-of-range tokens
        top_seq = top_seq.clamp(0, self.top_vocab - 1)

        # -- 2) sample bottom codes ------------------------------- #
        bos_bot = torch.full((n, 1),
                             0,  # will be BOS row in the grid
                             dtype=torch.long,
                             device=device)
        bot_prefix = torch.cat([bos_bot, top_seq], dim=1)

        # pad prefix to grid length
        pad_len = self.seq_len_bot_grid - bot_prefix.size(1)
        bot_prefix = F.pad(bot_prefix, (0, pad_len), value=0)

        full_seq = self.bot_prior.generate(bot_prefix, temperature, top_k, seed)  # (n, seq_len_bot_grid)
        bot_seq_off = full_seq[:, self.top_len + 1 : self.top_len + 1 + self.bot_len]

        # Valid bottom tokens are in [bottom_offset, bottom_offset + bot_vocab ‑ 1]
        bot_seq_off = bot_seq_off.clamp(self.bottom_offset, self.bottom_offset + self.bot_vocab - 1)

        bot_seq = bot_seq_off - self.bottom_offset  # shift back to [0, bot_vocab ‑ 1]

        # -- 3) decode -------------------------------------------- #
        imgs = self.vqvae.decode_code(code_t=top_seq, code_b=bot_seq)
        return imgs.clamp(-1, 1)  # match VQ-VAE output range


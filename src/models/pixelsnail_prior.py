"""
Hierarchical PixelSNAIL prior for Latent-VQVAE-2-style models
(closely mirrors the “HierarchicalTransformerPrior” but
replaces the AR-Transformers with 2-D PixelSNAIL networks).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.latent_models import LatentVQVAE2


# Xausal shift helpers (Checkerboard PixelCNN trick)
def shift_down(x: torch.Tensor) -> torch.Tensor:
    """Shift the tensor down by one pixel, zero-filling the new first row."""
    return F.pad(x, (0, 0, 1, 0))[:, :, :-1, :]

def shift_right(x: torch.Tensor) -> torch.Tensor:
    """Shift the tensor right by one pixel, zero-filling the new first column."""
    return F.pad(x, (1, 0, 0, 0))[:, :, :, :-1]


class CausalConv2d(nn.Conv2d):
    """
    Mask-A or Mask-B causal convolution used in PixelCNN/PixelSnail.
    * mask_type = "A": current pixel is masked (first layer)
    * mask_type = "B": current pixel is visible (subsequent layers)
    """
    def __init__(self, in_c, out_c, kernel_size, mask_type="B", **kw):
        super().__init__(in_c, out_c, kernel_size, padding=kernel_size//2, **kw)
        assert mask_type in {"A", "B"}
        self.register_buffer("mask", torch.ones_like(self.weight))
        k = kernel_size // 2
        self.mask[..., k + 1 :, :] = 0.          # rows below
        self.mask[..., k, k + (mask_type == "A") :] = 0.  # row = center, right of (and inc.) center
        self.weight.data *= self.mask            # mask once lazily

    def forward(self, x):
        self.weight.data *= self.mask            # ensure still masked after DDP sync
        return super().forward(x)


class GatedResidual(nn.Module):
    """
    Gated residual unit from van den Oord et al. (2016) used inside PixelSnail.
    Optionally conditions on a spatial tensor `cond` (same HxW) or on a global
    embedding `class_emb` (e.g. class label or latent top stack).
    """
    def __init__(self, n_chan, kernel_size=3, mask_type="B", cond_chan=None):
        super().__init__()
        self.conv = CausalConv2d(n_chan, 2 * n_chan, kernel_size, mask_type)
        self.cond_proj = None
        if cond_chan is not None:
            self.cond_proj = nn.Conv2d(cond_chan, 2 * n_chan, 1)

    def forward(self, x, cond=None):
        h = self.conv(x)
        if cond is not None and self.cond_proj is not None:
            h = h + self.cond_proj(cond)
        a, b = h.chunk(2, dim=1)
        h = torch.tanh(a) * torch.sigmoid(b)
        return x + h


class PixelBlock(nn.Module):
    """
    One PixelSnail block = N gated residual units (+ attention every K steps).
    """
    def __init__(self, n_chan, n_residual, kernel_size, cond_chan=None,
                 attn_heads=4, attn_every=2, dropout=0.1):
        super().__init__()
        self.res_units = nn.ModuleList(
            [
                GatedResidual(
                    n_chan,
                    kernel_size=kernel_size,
                    mask_type="B",          # all internal layers use mask‑B
                    cond_chan=cond_chan,
                )
                for _ in range(n_residual)
            ]
        )
        self.attn_every = attn_every
        self.self_attn = nn.MultiheadAttention(
            n_chan, attn_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond=None):
        H, W = x.shape[-2:]
        for i, res in enumerate(self.res_units):
            x = res(x, cond)
            # 1×1 global self‑attention every `attn_every` residual units
            if (i + 1) % self.attn_every == 0:
                B, C, H, W = x.shape
                L = H * W
                h = x.flatten(2).transpose(1, 2)  # B, L, C
                mask = torch.triu(torch.ones(L, L, device=h.device, dtype=torch.bool), diagonal=1)
                attn_out, _ = self.self_attn(h, h, h, attn_mask=mask)
                x = x + self.dropout(
                    attn_out.transpose(1, 2).view(-1, x.size(1), H, W)
                )
        return x


class PixelSNAIL2D(nn.Module):
    """
    Plain PixelSnail implementation (close to https://github.com/vvvm23/PixelSnail).
    Uses:
      • 1 mask-A causal opener
      • stacked gated residual / attention blocks
      • optional spatial conditioning feature map
    """
    def __init__(
        self,
        vocab_size,
        shape,              # (H, W)
        n_chan=128,
        n_blocks=2,         # number of PixelBlocks
        n_residual=4,       # residual units per PixelBlock
        n_heads=4,
        kernel_size=3,
        dropout=0.1,
        add_pos_emb=True,
        cond_chan=None,
    ):
        super().__init__()
        self.shape = shape
        H, W = shape
        self.tok_emb = nn.Embedding(vocab_size, n_chan)

        # first layer mask‑A conv (current pixel hidden)
        self.opener = CausalConv2d(
            n_chan, n_chan, kernel_size, mask_type="A"
        )

        self.blocks = nn.ModuleList(
            [
                PixelBlock(
                    n_chan,
                    n_residual=n_residual,
                    kernel_size=kernel_size,
                    cond_chan=cond_chan,
                    attn_heads=n_heads,
                    attn_every=2,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.out = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(n_chan, n_chan, 1),
            nn.ELU(),
            nn.Conv2d(n_chan, vocab_size, 1),
        )

        if add_pos_emb:
            self.row_emb = nn.Parameter(torch.randn(H, n_chan))
            self.col_emb = nn.Parameter(torch.randn(W, n_chan))
        else:
            self.row_emb = self.col_emb = None

    def add_positional(self, x):
        if self.row_emb is None:
            return x
        B, C, H, W = x.shape
        x = (
            x
            + self.row_emb.permute(1, 0)[None, :, :, None]
            + self.col_emb.permute(1, 0)[None, :, None, :]
        )
        return x

    def forward(self, idx, cond=None):
        B, H, W = idx.shape
        assert (H, W) == self.shape, "input grid shape mismatch"
        x = self.tok_emb(idx).permute(0, 3, 1, 2).contiguous()   # B,C,H,W
        x = self.add_positional(x)
        x = self.opener(x)

        for blk in self.blocks:
            x = blk(x, cond)

        logits = self.out(x)           # B,vocab,H,W
        return logits

    @torch.no_grad()
    def generate(self, B, device, temperature=1.0, top_k=None, cond=None):
        H, W = self.shape
        seq = torch.zeros(B, H, W, dtype=torch.long, device=device)
        for y in range(H):
            for x in range(W):
                logits = self(seq, cond)[:, :, y, x]
                logits = logits.float() / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                seq[:, y, x] = torch.multinomial(probs, 1).squeeze(-1)
        return seq
    

class HierarchicalPixelSnailPrior(pl.LightningModule):
    """
    Two-stage PixelSNAIL prior that learns p(top) x p(bottom | top)
    over the discrete latent grids produced by a LatentVQVAE-2 encoder.
    """
    def __init__(self, vqvae : LatentVQVAE2, n_chan=128, n_blocks=8, n_heads=4, lr=3e-4, weight_decay=0.0, dropout=0.1):
        """
        Initialize the HierarchicalPixelSnailPrior.
        Args:
            vqvae: Pre-trained LatentVQVAE2 model to use for encoding.
            n_chan: Number of channels in the PixelSNAIL networks.
            n_blocks: Number of blocks in the PixelSNAIL networks.
            n_heads: Number of attention heads in the PixelSNAIL networks.
            lr: Learning rate for the optimizer.
            weight_decay: Weight decay for the optimizer.
            dropout: Dropout rate for the PixelSNAIL networks.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["vqvae"])

        # Freeze VQ-VAE
        self.vqvae = vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False

        # Sizes/Vocab/Offsets
        self.top_vocab = self.vqvae.quantize_top.n_e
        self.bot_vocab = self.vqvae.quantize_bottom.n_e
        self.bottom_offset = self.top_vocab

        Ht, Wt, _ = self.vqvae.latent_dim_top    # e.g. 8×8×C
        Hb, Wb, _ = self.vqvae.latent_dim_bottom # e.g. 16×16×C

        # PixelSNAILs
        self.top_prior = PixelSNAIL2D(
            vocab_size=self.top_vocab,
            shape=(Ht, Wt),
            n_chan=n_chan,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
            cond_chan=None,
        )
        # bottom gets spatial conditioning from decoded top (channel = embed_dim)
        cond_chan = n_chan  # project top codebook to this below
        self.bot_prior = PixelSNAIL2D(
            vocab_size=self.top_vocab + self.bot_vocab,
            shape=(Hb, Wb),
            n_chan=n_chan,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
            cond_chan=cond_chan,
        )
        self.top_code_proj = nn.Conv2d(self.vqvae.quantize_top.e_dim,
                                       cond_chan, 1)

        self.lr = lr
        self.weight_decay = weight_decay

    @torch.no_grad()
    def encode_to_idx(self, x):
        """
        Encode input images to discrete indices for top and bottom latents.
        Args:
            x: Input tensor of shape (B, C, H, W).
        Returns:
            A tuple of tensors (top_idx, bot_idx) where:
            - top_idx: Indices for the top latent grid of shape (B, Ht, Wt).
            - bot_idx: Indices for the bottom latent grid of shape (B, Hb, Wb).
        """
        _, _, _, flat_idx = self.vqvae.encode(x)
        flat_idx = flat_idx.clamp_min(0)
        Ht, Wt, _ = self.vqvae.latent_dim_top
        top_len = Ht * Wt
        top = flat_idx[:, :top_len].view(-1, Ht, Wt)
        bot = flat_idx[:, top_len:].view(-1, *self.vqvae.latent_dim_bottom[:2])
        return top.long(), bot.long()

    def forward(self, x):
        """
        Forward pass through the hierarchical PixelSNAIL prior.
        Args:
            x: Input tensor of shape (B, C, H, W) containing images.
        Returns:
            A tuple (loss, logs) where:
            - loss: Total loss combining top and bottom losses.
            - logs: Dictionary containing individual losses for logging.
        """
        top_idx, bot_idx = self.encode_to_idx(x)

        # -- Top loss --------------------------------------------- #
        logits_top = self.top_prior(top_idx)

        # >>>>>>>>>>>>>>  DEBUG  <<<<<<<<<<<<<<
        with torch.no_grad():
            pred_top = logits_top.argmax(dim=1)          # B,Ht,Wt
            acc_top  = (pred_top == top_idx).float().mean()
            p_max    = logits_top.softmax(-1).amax(-1).mean()
        # These keys will later be picked up by training_step
        extra_logs = {
            "acc_top":  acc_top,
            "pmax_top": p_max,
        }
        # >>>>>>>>>>>>>>  /DEBUG  <<<<<<<<<<<<<<

        loss_top = F.cross_entropy(
            logits_top.permute(0, 2, 3, 1).reshape(-1, self.top_vocab),
            top_idx.reshape(-1),
        )

        # -- Bottom loss ------------------------------------------ #
        bot_idx_off = bot_idx + self.bottom_offset

        # spatial conditioning from top latent (projected embedding)
        with torch.no_grad():
            quant_top = self.vqvae.quantize_top.get_codebook_entry(
                top_idx.reshape(x.size(0), -1),
                shape=(x.size(0), *self.vqvae.latent_dim_top),
            )
        cond_feat = self.top_code_proj(quant_top)  # B,C,Ht,Wt
        # up-sample to bottom grid (nearest) so shapes match
        cond_feat = F.interpolate(cond_feat, size=bot_idx.shape[-2:], mode="nearest")

        logits_bot = self.bot_prior(bot_idx_off, cond=cond_feat)
        loss_bot = F.cross_entropy(
            logits_bot.permute(0, 2, 3, 1).reshape(-1, self.top_vocab + self.bot_vocab),
            bot_idx_off.reshape(-1),
        )

        loss = loss_top + loss_bot
        return loss, dict(total_loss=loss, loss_top=loss_top, loss_bottom=loss_bot, **extra_logs) # <<< DEBUG >>>

    def training_step(self, batch, *_):
        """Lightning training step."""
        loss, logs = self(batch)
        self.log_dict({f"train/{k}": v for k, v in logs.items()},
                      prog_bar=False, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, *_):
        """Lightning validation step."""
        loss, logs = self(batch)
        self.log_dict({f"val/{k}": v for k, v in logs.items()},
                      prog_bar=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr,
                                 weight_decay=self.weight_decay)

    @torch.no_grad()
    def sample(self, n, temperature=1.0, top_k=None):
        """
        Sample SD latents from the hierarchical PixelSNAIL prior.
        Args:
            n: Number of images to sample.
            temperature: Temperature for sampling (higher = more random).
            top_k: If specified, only consider the top-k logits for sampling.
        Returns:
            A tensor of shape (n, C, H, W) containing sampled SD latents.
        """
        device = next(self.parameters()).device
        # -- Sample top ------------------------------------------- #
        top_idx = self.top_prior.generate(
            B=n, device=device, temperature=temperature, top_k=top_k
        )  # (B,Ht,Wt)

        # project to bottom‐conditioning feature map
        quant_top = self.vqvae.quantize_top.get_codebook_entry(
            top_idx.reshape(n, -1),
            shape=(n, *self.vqvae.latent_dim_top),
        )
        cond_feat = self.top_code_proj(quant_top)
        cond_feat = F.interpolate(cond_feat,
                                  size=self.vqvae.latent_dim_bottom[:2],
                                  mode="nearest")

        # -- Sample bottom ---------------------------------------- #
        bot_idx_off = self.bot_prior.generate(
            B=n,
            device=device,
            temperature=temperature,
            top_k=top_k,
            cond=cond_feat,
        )
        # strip offset
        bot_idx = (bot_idx_off - self.bottom_offset).clamp_min_(0)

        # -- Decode to SD latents --------------------------------- #
        return self.vqvae.decode_code(code_t=top_idx, code_b=bot_idx)

    @torch.no_grad()
    def _free_run_nll(self, images):
        top_idx, _ = self.encode_to_idx(images)
        B, H, W = top_idx.shape
        seq = torch.zeros_like(top_idx)
        nll = 0.0
        for y in range(H):
            for x in range(W):
                logits = self.top_prior(seq)[:, :, y, x]          # B,V
                logp   = F.log_softmax(logits, -1)
                nll   += -logp.gather(-1, top_idx[:, y, x].unsqueeze(-1)).sum()
                seq[:, y, x] = top_idx[:, y, x]                   # teacher-force
        # nats per token
        return nll / (B * H * W)
"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import time

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import PAD_IDX, SOS_IDX, EOS_IDX


# ══════════════════════════════════════════════════════════════════════
#   LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    The <pad> index receives zero probability and is excluded from
    the denominator of the mean loss.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]
            target : shape [batch * tgt_len]
        Returns:
            Scalar loss value.
        """
        # Build smooth target distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(logits, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0.0
            # Mask out pad positions entirely
            pad_mask = (target == self.pad_idx)
            smooth_dist[pad_mask] = 0.0

        # KL-divergence: sum( p * log(p/q) ) = -sum( p * log_softmax(logits) ) + const
        log_prob = torch.log_softmax(logits, dim=-1)
        loss = -(smooth_dist * log_prob).sum(dim=-1)

        # Average over non-pad tokens
        n_tokens = (~pad_mask).sum().clamp(min=1)
        return loss.sum() / n_tokens


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0
    start_time   = time.time()

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch_idx, (src, tgt) in enumerate(data_iter):
            src = src.to(device)
            tgt = tgt.to(device)

            # Decoder input: all tokens except last (<eos> stripped)
            tgt_in  = tgt[:, :-1]
            # Decoder target: all tokens except first (<sos> stripped)
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx=PAD_IDX).to(device)

            # Forward pass
            logits = model(src, tgt_in, src_mask, tgt_mask)
            # logits: (batch, tgt_len-1, vocab_size)

            # Flatten for loss
            batch_size, tgt_len, vocab_size = logits.shape
            logits_flat  = logits.reshape(-1, vocab_size)
            targets_flat = tgt_out.reshape(-1)

            loss = loss_fn(logits_flat, targets_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping (standard practice for Transformers)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            n_tokens     = (tgt_out != PAD_IDX).sum().item()
            total_loss  += loss.item() * n_tokens
            total_tokens += n_tokens

            if batch_idx % 50 == 0:
                elapsed = time.time() - start_time
                mode    = "TRAIN" if is_train else "EVAL"
                print(
                    f"[{mode}] Epoch {epoch_num} | Step {batch_idx} | "
                    f"Loss {loss.item():.4f} | Elapsed {elapsed:.1f}s"
                )

    avg_loss = total_loss / max(total_tokens, 1)
    print(f"  → Epoch {epoch_num} {'train' if is_train else 'val'} avg loss: {avg_loss:.4f}")
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)  # (1, src_len, d_model)

        # Start with <sos>
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX).to(device)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            # logits: (1, cur_len, vocab_size) — take last position
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
            ys = torch.cat([ys, next_token], dim=1)

            if next_token.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    from torchtext.data.metrics import bleu_score as torchtext_bleu

    model.eval()
    all_hypotheses  = []
    all_references  = []

    # Resolve vocabulary lookup method
    def idx_to_token(idx):
        if hasattr(tgt_vocab, 'lookup_token'):
            return tgt_vocab.lookup_token(idx)
        return tgt_vocab.itos[idx]

    special_indices = {SOS_IDX, EOS_IDX, PAD_IDX}

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)

            ys = greedy_decode(
                model, src, src_mask,
                max_len=max_len,
                start_symbol=SOS_IDX,
                end_symbol=EOS_IDX,
                device=device,
            )

            # Convert hypothesis to list of tokens (strip special tokens)
            hyp_tokens = [
                idx_to_token(idx.item())
                for idx in ys[0]
                if idx.item() not in special_indices
            ]
            # Convert reference to list of tokens (strip special tokens)
            ref_tokens = [
                idx_to_token(idx.item())
                for idx in tgt[0]
                if idx.item() not in special_indices
            ]

            all_hypotheses.append(hyp_tokens)
            all_references.append([ref_tokens])   # torchtext expects list of lists

    # torchtext bleu_score returns value in [0, 1]; multiply by 100
    score = torchtext_bleu(all_hypotheses, all_references) * 100.0
    return score


# ══════════════════════════════════════════════════════════════════════
#   CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """Save model + optimiser + scheduler state to disk."""
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config':         model.config,
    }, path)
    print(f"  Checkpoint saved → {path}  (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    print(f"  Checkpoint loaded from {path}  (epoch {epoch})")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
    """
    #import wandb
    from dataset import get_dataloaders
    from lr_scheduler import NoamScheduler

    # ── Hyperparameters ───────────────────────────────────────────────
    config = dict(
        d_model      = 256,
        N            = 3,
        num_heads    = 8,
        d_ff         = 512,
        dropout      = 0.1,
        batch_size   = 128,
        num_epochs   = 25,
        warmup_steps = 4000,
        smoothing    = 0.1,
    )

    #wandb.init(project="da6401-a3", config=config)
    #cfg = wandb.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=config['batch_size']
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = config['d_model'],
        N              = config['N'],
        num_heads      = config['num_heads'],
        d_ff           = config['d_ff'],
        dropout        = config['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optimiser & Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=config['d_model'], warmup_steps=config['warmup_steps'])

    # ── Loss ──────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=config['smoothing'],
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        #wandb.log({
        #    'epoch':      epoch,
        #    'train_loss': train_loss,
        #    'val_loss':   val_loss,
        #    'lr':         optimizer.param_groups[0]['lr'],
        #})

        save_checkpoint(model, optimizer, scheduler, epoch, path=f"checkpoint_epoch{epoch}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint_best.pt")

    # ── Final BLEU ────────────────────────────────────────────────────
    best_epoch = load_checkpoint("checkpoint_best.pt", model)
    model.to(device)

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"\nTest BLEU: {bleu:.2f}")
    #wandb.log({'test_bleu': bleu, 'best_epoch': best_epoch})

    #wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
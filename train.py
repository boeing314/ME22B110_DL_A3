import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import time
from model import Transformer, make_src_mask, make_tgt_mask
from dataset import PAD_IDX, SOS_IDX, EOS_IDX

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            smooth_dist = torch.full_like(logits, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0.0
            pad_mask = (target == self.pad_idx)
            smooth_dist[pad_mask] = 0.0
        log_prob = torch.log_softmax(logits, dim=-1)
        loss = -(smooth_dist * log_prob).sum(dim=-1)
        n_tokens = (~pad_mask).sum().clamp(min=1)
        return loss.sum() / n_tokens

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
    model.train() if is_train else model.eval()
    total_loss   = 0.0
    total_tokens = 0
    start_time   = time.time()
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, (src, tgt) in enumerate(data_iter):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx=PAD_IDX).to(device)
            logits = model(src, tgt_in, src_mask, tgt_mask)
            batch_size, tgt_len, vocab_size = logits.shape
            logits_flat  = logits.reshape(-1, vocab_size)
            targets_flat = tgt_out.reshape(-1)
            loss = loss_fn(logits_flat, targets_flat)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
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

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)  # (1, src_len, d_model)

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

def _corpus_bleu(hypotheses: list, references: list) -> float:
    import math
    from collections import Counter

    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    clipped_counts = [0] * 4
    total_counts   = [0] * 4
    hyp_len = 0
    ref_len = 0
    for hyp, refs in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref = min(refs, key=lambda r: abs(len(r) - len(hyp)))
        ref_len += len(ref)
        for n in range(1, 5):
            hyp_ng = ngrams(hyp, n)
            ref_ng = ngrams(ref, n)
            for gram, cnt in hyp_ng.items():
                clipped_counts[n-1] += min(cnt, ref_ng.get(gram, 0))
            total_counts[n-1] += max(len(hyp) - n + 1, 0)
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len >= ref_len else math.exp(1.0 - ref_len / hyp_len)
    log_avg = 0.0
    for n in range(4):
        num = clipped_counts[n] + 1e-10
        den = total_counts[n]   + 1e-10
        log_avg += math.log(num / den)
    return bp * math.exp(log_avg / 4) * 100.0

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:

    model.eval()
    all_hypotheses = []
    all_references = []

    def idx_to_token(idx):
        if hasattr(tgt_vocab, 'lookup_token'):
            return tgt_vocab.lookup_token(idx)
        return tgt_vocab.itos[idx]
    special_indices = {SOS_IDX, EOS_IDX, PAD_IDX}
    with torch.no_grad():
        for src, tgt in test_dataloader:

            src = src.to(device)
            tgt = tgt.to(device)

            batch_size = src.size(0)

            for i in range(batch_size):

                src_i = src[i:i+1]
                tgt_i = tgt[i:i+1]

                src_mask_i = make_src_mask(
                    src_i,
                    pad_idx=PAD_IDX
                ).to(device)

                ys = greedy_decode(
                    model=model,
                    src=src_i,
                    src_mask=src_mask_i,
                    max_len=max_len,
                    start_symbol=SOS_IDX,
                    end_symbol=EOS_IDX,
                    device=device,
                )

                hyp_tokens = [
                    idx_to_token(idx.item())
                    for idx in ys[0]
                    if idx.item() not in special_indices
                ]

                ref_tokens = [
                    idx_to_token(idx.item())
                    for idx in tgt_i[0]
                    if idx.item() not in special_indices
                ]

                all_hypotheses.append(hyp_tokens)
                all_references.append([ref_tokens])

    bleu_score = _corpus_bleu(
        all_hypotheses,
        all_references
    )

    return bleu_score

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config':         model.config,
        # ── vocab bundled here — loaded by Transformer.__init__ ──────
        'src_itos':             model._src_vocab.itos,
        'tgt_itos':             model._tgt_vocab.itos,
    }, path)
    print(f"  Checkpoint saved → {path}  (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    print(f"  Checkpoint loaded from {path}  (epoch {epoch})")
    return epoch

def run_training_experiment() -> None:
    import wandb
    from dataset import get_dataloaders
    from lr_scheduler import NoamScheduler

    config = dict(
        d_model       = 256,
        N             = 3,
        num_heads     = 8,
        d_ff          = 1024,
        dropout       = 0.05,
        batch_size    = 128,
        num_epochs    = 10,
        warmup_steps  = 2000,
        smoothing     = 0.02,
    )
    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=cfg.batch_size
    )

    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = cfg.d_model,
        N              = cfg.N,
        num_heads      = cfg.num_heads,
        d_ff           = cfg.d_ff,
        dropout        = cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model=cfg.d_model,
        warmup_steps=cfg.warmup_steps
    )

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=cfg.smoothing,
    )

    best_val_loss = float('inf')
    best_bleu = 0.0

    for epoch in range(cfg.num_epochs):

        print(f"\n========== Epoch {epoch} ==========")

        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
        )

        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch_num=epoch,
            is_train=False,
            device=device,
        )

        bleu = evaluate_bleu(
            model,
            val_loader,
            tgt_vocab,
            device=device,
        )

        print(
            f"Epoch {epoch} Summary | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"BLEU: {bleu:.2f}"
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "bleu": bleu,
        })

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            path=f"checkpoint_epoch{epoch}.pt"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                path="checkpoint_best_loss.pt"
            )

            print(f"New best validation loss: {best_val_loss:.4f}")

        if bleu > best_bleu:
            best_bleu = bleu

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                path="checkpoint_best_bleu.pt"
            )

            print(f"New best BLEU: {best_bleu:.2f}")

    print("\nLoading best BLEU checkpoint...")

    load_checkpoint("checkpoint_best_bleu.pt", model)

    model.to(device)

    test_bleu = evaluate_bleu(
        model,
        test_loader,
        tgt_vocab,
        device=device
    )

    print(f"\nFinal Test BLEU: {test_bleu:.2f}")

    wandb.log({
        "final_test_bleu": test_bleu
    })

    wandb.finish()

if __name__ == "__main__":
    run_training_experiment()
"""
patch_checkpoint.py
───────────────────
Run this ONCE on your training machine after training is complete.
It adds src_itos / tgt_itos to an existing checkpoint so that
Transformer() can reconstruct the vocabulary without needing the
`datasets` library at inference time.

Usage:
    python patch_checkpoint.py                         # patches checkpoint_best.pt
    python patch_checkpoint.py my_checkpoint.pt        # patches a specific file
"""

import sys
import torch


def patch(ckpt_path: str = "checkpoint_best.pt"):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "src_itos" in ckpt and "tgt_itos" in ckpt:
        print(f"✓ {ckpt_path} already has vocab — nothing to do.")
        return

    # Build vocab from Multi30k training split
    print("Building vocab from Multi30k training split ...")
    import importlib, os

    # Temporarily drop local path so HuggingFace datasets is found
    local_paths = [p for p in sys.path if os.path.exists(os.path.join(p, "datasets.py"))]
    for p in local_paths:
        sys.path.remove(p)
    try:
        hf = importlib.import_module("datasets")
        if not hasattr(hf, "load_dataset"):
            importlib.reload(hf)
        load_dataset = hf.load_dataset
    finally:
        for p in local_paths:
            sys.path.insert(0, p)

    import spacy
    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")

    SPECIAL = ["<unk>", "<pad>", "<sos>", "<eos>"]
    src_itos, tgt_itos = SPECIAL[:], SPECIAL[:]
    src_set,  tgt_set  = set(src_itos), set(tgt_itos)

    dataset = load_dataset("bentrevett/multi30k", split="train")
    for ex in dataset:
        for tok in spacy_de.tokenizer(ex["de"]):
            t = tok.text.lower()
            if t not in src_set:
                src_itos.append(t); src_set.add(t)
        for tok in spacy_en.tokenizer(ex["en"]):
            t = tok.text.lower()
            if t not in tgt_set:
                tgt_itos.append(t); tgt_set.add(t)

    ckpt["src_itos"] = src_itos
    ckpt["tgt_itos"] = tgt_itos
    torch.save(ckpt, ckpt_path)
    print(f"✓ Saved {ckpt_path} with vocab ({len(src_itos)} src, {len(tgt_itos)} tgt tokens)")
    print("  → Re-upload this file to Google Drive (same file ID, replace existing).")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint_best.pt"
    patch(path)
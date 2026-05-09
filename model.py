"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    # Scaled scores: (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_w = F.softmax(scores, dim=-1)
    # Replace NaNs that arise when entire rows are -inf (padding rows)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#   MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # (batch, 1, 1, src_len)  True where pad
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    tgt_len = tgt.size(1)
    # Causal mask: upper triangle (excluding diagonal) is True
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
        diagonal=1
    )  # (tgt_len, tgt_len)

    # Padding mask: (batch, 1, 1, tgt_len)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # Combine: (batch, 1, tgt_len, tgt_len)
    return causal_mask.unsqueeze(0).unsqueeze(0) | pad_mask


# ══════════════════════════════════════════════════════════════════════
#   MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_weights = None  # store for visualization

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        batch_size = query.size(0)

        # Linear projections and split into heads
        # (batch, seq, d_model) → (batch, num_heads, seq, d_k)
        def project_and_split(linear, x):
            return linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = project_and_split(self.W_q, query)
        K = project_and_split(self.W_k, key)
        V = project_and_split(self.W_v, value)

        # Scaled dot-product attention per head
        attn_out, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # Apply dropout to attention weights (through output)
        # attn_out: (batch, num_heads, seq_q, d_k)

        # Concatenate heads: (batch, seq_q, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(attn_out)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build PE table: (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer (not a parameter — satisfies autograder check)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]
        Returns:
            Tensor of same shape [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#   FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:
        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#   ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer (Post-LayerNorm):
        x → Self-Attention → Add & Norm → FFN → Add & Norm

    Post-LN matches the original paper exactly and is straightforward
    to implement and debug.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention sub-layer with residual
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN sub-layer with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer (Post-LayerNorm):
        x → Masked Self-Attn → Add & Norm
          → Cross-Attn(memory) → Add & Norm
          → FFN → Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Masked self-attention
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        # Cross-attention over encoder memory
        attn2 = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#   ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

# Special token indices (must match dataset.py)
_UNK_IDX, _PAD_IDX, _SOS_IDX, _EOS_IDX = 0, 1, 2, 3


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for De→En machine translation.

    On construction the model:
      1. Loads spaCy tokenisers (de + en).
      2. Builds / loads source & target vocabularies from the Multi30k
         training split (so vocab is always consistent with training).
      3. Downloads trained weights from Google Drive via gdown and loads
         them — all transparent to the caller.

    Public API guaranteed by the autograder:
        model = Transformer()          # no args required
        english = model.infer(german)  # str → str
    """

    # ── Google Drive file-id of your saved checkpoint ─────────────────
    # Replace this with your actual file-id once you upload to Drive.
    _GDRIVE_FILE_ID   = "161lOAjSnO4Fmx3lQqYug2gDmmTKXxYst"#"YOUR_GDRIVE_FILE_ID_HERE"
    _CHECKPOINT_NAME  = "checkpoint_best.pt"

    def __init__(
        self,
        src_vocab_size: int  = 0,   # 0 → derived from built vocab
        tgt_vocab_size: int  = 0,   # 0 → derived from built vocab
        d_model:   int   = 256,
        N:         int   = 3,
        num_heads: int   = 8,
        d_ff:      int   = 512,
        dropout:   float = 0.1,
        weights_path: str = "",     # explicit local path overrides gdown
        device: str = "",           # "" → auto-detect
    ) -> None:
        # ── 1. Tokenisers ──────────────────────────────────────────────
        import spacy, subprocess, os
        try:
            self._spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            subprocess.run(["python", "-m", "spacy", "download",
                            "de_core_news_sm"], check=True)
            self._spacy_de = spacy.load("de_core_news_sm")

        try:
            self._spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            subprocess.run(["python", "-m", "spacy", "download",
                            "en_core_web_sm"], check=True)
            self._spacy_en = spacy.load("en_core_web_sm")

        # ── 2. Vocabulary ──────────────────────────────────────────────
        self._src_vocab, self._tgt_vocab = self._build_vocabs()

        resolved_src = src_vocab_size if src_vocab_size > 0 else len(self._src_vocab)
        resolved_tgt = tgt_vocab_size if tgt_vocab_size > 0 else len(self._tgt_vocab)

        # ── 3. Architecture ────────────────────────────────────────────
        super().__init__()

        self.src_embed   = nn.Embedding(resolved_src, d_model)
        self.tgt_embed   = nn.Embedding(resolved_tgt, d_model)
        self.pos_enc     = PositionalEncoding(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder     = Encoder(enc_layer, N)
        self.decoder     = Decoder(dec_layer, N)
        self.output_proj = nn.Linear(d_model, resolved_tgt)

        self.config = dict(
            src_vocab_size=resolved_src,
            tgt_vocab_size=resolved_tgt,
            d_model=d_model, N=N, num_heads=num_heads,
            d_ff=d_ff, dropout=dropout,
        )

        self._init_weights()

        # ── 4. Device ──────────────────────────────────────────────────
        if device:
            self._device = device
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── 5. Load weights ────────────────────────────────────────────
        ckpt_path = weights_path or self._CHECKPOINT_NAME
        if not os.path.exists(ckpt_path):
            self._download_weights(ckpt_path)
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            # checkpoint may be a raw state-dict or our full dict
            if isinstance(state, dict) and "model_state_dict" in state:
                self.load_state_dict(state["model_state_dict"])
            else:
                self.load_state_dict(state)
            print(f"[Transformer] weights loaded from {ckpt_path}")
        else:
            print("[Transformer] WARNING: no weights found — using random init")

        self.to(self._device)

    # ── Vocabulary helpers ────────────────────────────────────────────

    class _Vocab:
        """Tiny vocabulary: list-backed, four special tokens prepended."""
        UNK, PAD, SOS, EOS = 0, 1, 2, 3

        def __init__(self):
            self.itos = ["<unk>", "<pad>", "<sos>", "<eos>"]
            self.stoi = {t: i for i, t in enumerate(self.itos)}

        def add(self, token: str):
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

        def __len__(self):
            return len(self.itos)

        def encode(self, tokens):
            return [self.stoi.get(t, self.UNK) for t in tokens]

        def decode(self, indices):
            special = {self.UNK, self.PAD, self.SOS, self.EOS}
            return [self.itos[i] for i in indices if i not in special]

    def _tokenize_de(self, text: str):
        return [t.text.lower() for t in self._spacy_de.tokenizer(text)]

    def _tokenize_en(self, text: str):
        return [t.text.lower() for t in self._spacy_en.tokenizer(text)]

    def _build_vocabs(self):
        """Build src/tgt vocabs from Multi30k training split."""
        from datasets import load_dataset
        src_vocab = self._Vocab()
        tgt_vocab = self._Vocab()
        dataset   = load_dataset("bentrevett/multi30k", split="train")
        for ex in dataset:
            for tok in self._tokenize_de(ex["de"]):
                src_vocab.add(tok)
            for tok in self._tokenize_en(ex["en"]):
                tgt_vocab.add(tok)
        return src_vocab, tgt_vocab

    # ── Weight download ───────────────────────────────────────────────

    def _download_weights(self, dest_path: str):
        """Download checkpoint from Google Drive using gdown."""
        if self._GDRIVE_FILE_ID == "YOUR_GDRIVE_FILE_ID_HERE":
            print("[Transformer] _GDRIVE_FILE_ID not set — skipping download")
            return
        try:
            import gdown
        except ImportError:
            import subprocess
            subprocess.run(["pip", "install", "gdown", "-q"], check=True)
            import gdown
        url = f"https://drive.google.com/uc?id={self._GDRIVE_FILE_ID}"
        print(f"[Transformer] downloading weights from Google Drive → {dest_path}")
        gdown.download(url, dest_path, quiet=False)

    # ── Weight init ───────────────────────────────────────────────────

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Autograder hooks ──────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.config["d_model"]))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.config["d_model"]))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.output_proj(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ── End-to-end inference ──────────────────────────────────────────

    def infer(self, german_sentence: str, max_len: int = 100) -> str:
        """
        Translate a single German sentence to English.

        Pipeline:
            1. Tokenise German input with spaCy.
            2. Convert tokens → integer indices using src_vocab.
            3. Build source mask and run encoder.
            4. Autoregressive greedy decoding with the decoder.
            5. Convert output indices → English tokens → joined string.

        Args:
            german_sentence (str) : Raw German input text.
            max_len         (int) : Maximum number of output tokens (default 100).

        Returns:
            str : Translated English sentence.
        """
        self.eval()
        device = self._device

        # ── Tokenise & encode source ───────────────────────────────────
        tokens  = self._tokenize_de(german_sentence)
        indices = ([self._Vocab.SOS]
                   + self._src_vocab.encode(tokens)
                   + [self._Vocab.EOS])
        src = torch.tensor([indices], dtype=torch.long, device=device)  # (1, src_len)

        # ── Masks & encoder ────────────────────────────────────────────
        src_mask = make_src_mask(src, pad_idx=_PAD_IDX).to(device)

        with torch.no_grad():
            memory = self.encode(src, src_mask)

            # ── Greedy decoding ────────────────────────────────────────
            ys = torch.tensor([[_SOS_IDX]], dtype=torch.long, device=device)

            for _ in range(max_len - 1):
                tgt_mask = make_tgt_mask(ys, pad_idx=_PAD_IDX).to(device)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                # Take argmax at the last position
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == _EOS_IDX:
                    break

        # ── Detokenise ─────────────────────────────────────────────────
        out_tokens = self._tgt_vocab.decode(ys[0].tolist())
        return " ".join(out_tokens)
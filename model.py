import math
import copy
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    tgt_len = tgt.size(1)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
        diagonal=1
    )

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    return causal_mask.unsqueeze(0).unsqueeze(0) | pad_mask

class MultiHeadAttention(nn.Module):
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
        batch_size = query.size(0)
        def project_and_split(linear, x):
            return linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = project_and_split(self.W_q, query)
        K = project_and_split(self.W_k, key)
        V = project_and_split(self.W_v, value)

        attn_out, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):

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

class DecoderLayer(nn.Module):

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
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        attn2 = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x

class Encoder(nn.Module):

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):

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

_UNK_IDX, _PAD_IDX, _SOS_IDX, _EOS_IDX = 0, 1, 2, 3


class Transformer(nn.Module):
    _GDRIVE_FILE_ID   = "161lOAjSnO4Fmx3lQqYug2gDmmTKXxYst"
    _CHECKPOINT_NAME  = "checkpoint_best.pt"

    def __init__(
        self,
        src_vocab_size: int  = 0,   
        tgt_vocab_size: int  = 0,   
        d_model:   int   = 256,
        N:         int   = 3,
        num_heads: int   = 8,
        d_ff:      int   = 1024,
        dropout:   float = 0.05,
        weights_path: str = "",     
        device: str = "",           
    ) -> None:
        import sys, spacy, subprocess, os

        def _load_spacy(model_name: str):
            try:
                return spacy.load(model_name)
            except OSError:
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", model_name],
                    check=True
                )
                return spacy.load(model_name)

        self._spacy_de = _load_spacy("de_core_news_sm")
        self._spacy_en = _load_spacy("en_core_web_sm")

        self._device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        ckpt_path = weights_path or self._CHECKPOINT_NAME
        if not os.path.exists(ckpt_path):
            self._download_weights(ckpt_path)

        _ckpt_state = None
        if os.path.exists(ckpt_path):
            _ckpt_state = torch.load(ckpt_path, map_location="cpu")

        if (isinstance(_ckpt_state, dict)
                and "src_itos" in _ckpt_state
                and "tgt_itos" in _ckpt_state):
            self._src_vocab, self._tgt_vocab = self._restore_vocabs(
                _ckpt_state["src_itos"], _ckpt_state["tgt_itos"]
            )
            print("[Transformer] vocab loaded from checkpoint")
        else:
            print("[Transformer] building vocab from Multi30k dataset ...")
            self._src_vocab, self._tgt_vocab = self._build_vocabs()

        resolved_src = src_vocab_size if src_vocab_size > 0 else len(self._src_vocab)
        resolved_tgt = tgt_vocab_size if tgt_vocab_size > 0 else len(self._tgt_vocab)

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

        if _ckpt_state is not None:
            sd = (_ckpt_state.get("model_state_dict")
                  if isinstance(_ckpt_state, dict) else _ckpt_state)
            self.load_state_dict(sd)
            print(f"[Transformer] weights loaded from {ckpt_path}")
        else:
            print("[Transformer] WARNING: no checkpoint found — using random init")

        self.to(self._device)

    class _Vocab:
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

    def _restore_vocabs(self, src_itos: list, tgt_itos: list):
        def _from_itos(itos):
            v = self._Vocab()
            v.itos = itos
            v.stoi = {t: i for i, t in enumerate(itos)}
            return v
        return _from_itos(src_itos), _from_itos(tgt_itos)

    def _download_weights(self, dest_path: str):
        if self._GDRIVE_FILE_ID == "YOUR_GDRIVE_FILE_ID_HERE":
            print("[Transformer] _GDRIVE_FILE_ID not set — skipping download")
            return

        try:
            import gdown
        except ImportError:
            import sys, subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"],
                           check=True)
            import gdown

        url = f"https://drive.google.com/uc?id={self._GDRIVE_FILE_ID}"
        print(f"[Transformer] downloading weights via gdown → {dest_path}")
        gdown.download(url, dest_path, quiet=False)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

    def infer(self, german_sentence: str, max_len: int = 100) -> str:
        
        self.eval()
        device = self._device
        tokens  = self._tokenize_de(german_sentence)
        indices = ([self._Vocab.SOS]
                   + self._src_vocab.encode(tokens)
                   + [self._Vocab.EOS])
        src = torch.tensor([indices], dtype=torch.long, device=device)  # (1, src_len)
        src_mask = make_src_mask(src, pad_idx=_PAD_IDX).to(device)

        with torch.no_grad():
            memory = self.encode(src, src_mask)
            ys = torch.tensor([[_SOS_IDX]], dtype=torch.long, device=device)

            for _ in range(max_len - 1):
                tgt_mask = make_tgt_mask(ys, pad_idx=_PAD_IDX).to(device)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                # Take argmax at the last position
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == _EOS_IDX:
                    break

        out_tokens = self._tgt_vocab.decode(ys[0].tolist())
        return " ".join(out_tokens)
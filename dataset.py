import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']


class Vocabulary:

    def __init__(self):
        self.itos = SPECIAL_TOKENS[:]          # index → token
        self.stoi = {t: i for i, t in enumerate(self.itos)}  # token → index

    def __len__(self):
        return len(self.itos)

    def add_token(self, token: str):
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def lookup_indices(self, tokens):
        return [self.stoi.get(t, UNK_IDX) for t in tokens]

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx] if idx < len(self.itos) else '<unk>'


class Multi30kDataset(Dataset):

    def __init__(self, split: str = 'train'):
        self.split = split

        # Load spaCy models
        try:
            self.spacy_de = spacy.load('de_core_news_sm')
        except OSError:
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'de_core_news_sm'], check=True)
            self.spacy_de = spacy.load('de_core_news_sm')

        try:
            self.spacy_en = spacy.load('en_core_web_sm')
        except OSError:
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
            self.spacy_en = spacy.load('en_core_web_sm')
        raw = load_dataset('bentrevett/multi30k')

        self.src_vocab = Vocabulary()
        self.tgt_vocab = Vocabulary()
        self._build_vocab(raw['train'])

        self.data = self._process_split(raw[split])

    def tokenize_de(self, text: str):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self):
        pass  # vocab is built in __init__

    def _build_vocab(self, train_split):
        for example in train_split:
            for tok in self.tokenize_de(example['de']):
                self.src_vocab.add_token(tok)
            for tok in self.tokenize_en(example['en']):
                self.tgt_vocab.add_token(tok)

    def process_data(self):
        pass  # data is processed in __init__

    def _encode(self, tokens, vocab: Vocabulary):
        return [SOS_IDX] + vocab.lookup_indices(tokens) + [EOS_IDX]

    def _process_split(self, split):
        examples = []
        for example in split:
            src_tokens = self.tokenize_de(example['de'])
            tgt_tokens = self.tokenize_en(example['en'])
            src_ids = self._encode(src_tokens, self.src_vocab)
            tgt_ids = self._encode(tgt_tokens, self.tgt_vocab)
            examples.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))
        return examples
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def get_dataloaders(batch_size: int = 128):
    train_ds = Multi30kDataset('train')
    val_ds   = Multi30kDataset('validation')
    test_ds  = Multi30kDataset('test')

    # Share vocabs built on training data
    val_ds.src_vocab  = train_ds.src_vocab
    val_ds.tgt_vocab  = train_ds.tgt_vocab
    test_ds.src_vocab = train_ds.src_vocab
    test_ds.tgt_vocab = train_ds.tgt_vocab

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, train_ds.src_vocab, train_ds.tgt_vocab
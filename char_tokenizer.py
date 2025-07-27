import json
from collections import Counter

class CharTokenizer:
    def __init__(self, text_path, min_freq=1):
        self.text_path = text_path
        self.min_freq = min_freq
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"

        self.build_vocab()

    def build_vocab(self):
        counter = Counter()
        with open(self.text_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    counter.update(list(line))

        # Special tokens
        chars = [self.pad_token, self.sos_token, self.eos_token]
        chars += [char for char, freq in counter.items() if freq >= self.min_freq and char not in chars]

        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.pad_token_id = self.char2idx[self.pad_token]
        self.sos_token_id = self.char2idx[self.sos_token]
        self.eos_token_id = self.char2idx[self.eos_token]

    def encode(self, text):
        return [self.char2idx[self.sos_token]] + [self.char2idx.get(c, self.pad_token_id) for c in text] + [self.char2idx[self.eos_token]]

    def decode(self, indices):
        tokens = []
        for i in indices:
            token = self.idx2char.get(i, "")
            if token == self.eos_token:
                break
            if token not in [self.sos_token, self.pad_token]:
                tokens.append(token)
        return ''.join(tokens)

    def save_vocab(self, path="vocab.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path="vocab.json"):
        with open(path, "r", encoding="utf-8") as f:
            self.char2idx = json.load(f)
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.pad_token_id = self.char2idx[self.pad_token]
        self.sos_token_id = self.char2idx[self.sos_token]
        self.eos_token_id = self.char2idx[self.eos_token]

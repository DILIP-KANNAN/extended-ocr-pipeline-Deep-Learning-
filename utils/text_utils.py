
from .config import CHARACTERS, BLANK_IDX
class TextMapper:
    def __init__(self, characters=CHARACTERS):
        # blank index reserved as 0 for CTC
        self.chars = characters
        self.idx_to_char = {i+1: c for i,c in enumerate(self.chars)}
        self.idx_to_char[BLANK_IDX] = ""  # blank -> empty for decoding
        self.char_to_idx = {c: i+1 for i,c in enumerate(self.chars)}
        self.blank = BLANK_IDX

    def encode(self, text):
        # returns list[int]
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices):
        # greedy decode: collapse repeats and remove blanks
        s = []
        prev = None
        for idx in indices:
            if idx == self.blank:
                prev = idx
                continue
            if idx != prev:
                s.append(self.idx_to_char.get(idx, ""))
            prev = idx
        return ''.join(s)

import json
import regex as re
from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        if special_tokens:
            for token in special_tokens:
                b_token = token.encode("utf-8")
                if b_token not in set(self.vocab.values()):
                    self.vocab[len(self.vocab)] = b_token
        self.vocab_inverse = {v:k for k, v in self.vocab.items()}
        self.merges_ranks = {(start, end):i for i, (start, end) in enumerate(merges)}
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath) as f:
            vocab = json.load(f)
        merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merges.append(tuple(line.split(" ")))
        tokenizer = cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
        return tokenizer
    
    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        token_list = []
        last_idx = 0

        if self.special_tokens:
            special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(map(re.escape, special_tokens_sorted))
            for m in re.finditer(special_pattern, text, ):
                split_chunk = text[last_idx:m.start()]
                last_idx = m.end()

                for word_match in re.finditer(PAT, split_chunk):
                    word = word_match.group()
                    bytes_tokens = [bytes([i]) for i in word.encode("utf-8")]
                    
                    while True:
                        pairs = [(i, (bytes_tokens[i], bytes_tokens[i+1]))
                                for i in range(len(bytes_tokens)-1)]
                        min_pair = min(pairs, key=lambda x: self.merges_ranks.get(x[1], float("inf")), default=None)
                        if not min_pair or self.merges_ranks.get(min_pair[1], float("inf")) == float("inf"):
                            break

                        idx, (start, end) = min_pair
                        bytes_tokens[idx] = start + end
                        bytes_tokens.pop(idx+1)

                    token_list.extend([self.vocab_inverse[i] for i in bytes_tokens])
                token_list.append(self.vocab_inverse[m.group().encode("utf-8")])

        remainder = text[last_idx:]
        for word_match in re.finditer(PAT, remainder):
            word = word_match.group()
            bytes_tokens = [bytes([i]) for i in word.encode("utf-8")]
            
            while True:
                pairs = [(i, (bytes_tokens[i], bytes_tokens[i+1]))
                        for i in range(len(bytes_tokens)-1)]
                min_pair = min(pairs, key=lambda x: self.merges_ranks.get(x[1], float("inf")), default=None)
                if not min_pair or self.merges_ranks.get(min_pair[1], float("inf")) == float("inf"):
                    break

                idx, (start, end) = min_pair
                bytes_tokens[idx] = start + end
                bytes_tokens.pop(idx+1)

            token_list.extend([self.vocab_inverse[i] for i in bytes_tokens])

        return token_list
   
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            token_ids = self.encode(text)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        bytes_list = [self.vocab[i] for i in ids]
        bytes_text: bytes = b''
        for bytes_i in bytes_list:
            bytes_text += bytes_i
        return bytes_text.decode("utf-8", errors="replace")
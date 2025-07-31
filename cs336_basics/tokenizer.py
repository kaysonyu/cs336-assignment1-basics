import os
from typing import BinaryIO
import regex as re
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def word_to_bytes(word: str) -> tuple[bytes, ...]:
    return tuple(bytes([i]) for i in word.encode(encoding="utf-8"))


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def process_chunk(input_path, start, end, special_tokens):
    word_counter = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_data = f.read(end - start).decode("utf-8")

    # Removing special tokens before pre-tokenization
    split_chunks = re.split("|".join(map(re.escape, special_tokens)), chunk_data)
    for split_chunk in split_chunks:
        for word_match in re.finditer(PAT, split_chunk):
            word = word_match.group(0)
            word_counter[word_to_bytes(word)] += 1

    return word_counter


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # step 1: Vocabulary initialization
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        b_token = token.encode(encoding="utf-8")
        if b_token not in vocab.values():
            vocab[len(vocab)] = b_token

    # step 2: Pre-tokenization
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, 4, split_special_token=b"<|endoftext|>")

    chunks = [(chunk_boundaries[i], chunk_boundaries[i + 1]) for i in range(len(chunk_boundaries) - 1)]
    word_counter = Counter()
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, input_path, start, end, special_tokens) for start, end in chunks]
        for future in futures:
            word_counter.update(future.result())

    # step 3: Compute BPE merges
    merges = []
    iter_num = vocab_size - len(vocab)

    for _ in tqdm(range(iter_num), desc="Compute BPE merges"):
        pair_counter: dict[tuple[bytes, bytes], int] = Counter()

        # compute max
        for bytes_array, cnt in word_counter.items():
            for start, end in zip(bytes_array[:-1], bytes_array[1:]):
                pair_counter[(start, end)] += cnt

        max_num = max(pair_counter.values())
        max_pair: tuple[bytes, bytes] = max([k for k, v in pair_counter.items() if v == max_num])

        a, b = max_pair
        new_bytes = a + b

        # change word_counter
        changes = []
        for bytes_array, cnt in word_counter.items():
            matchs = [i for i in range(len(bytes_array) - 1) if bytes_array[i : i + 2] == max_pair]
            if matchs:
                new_bytes_array = []
                i = 0
                while i < len(bytes_array):
                    if i in matchs:
                        new_bytes_array.append(new_bytes)
                        i += 2
                    else:
                        new_bytes_array.append(bytes_array[i])
                        i += 1
                new_bytes_array = tuple(new_bytes_array)
                changes.append((bytes_array, new_bytes_array, cnt))

        for bytes_array, new_bytes_array, cnt in changes:
            word_counter[new_bytes_array] = word_counter.get(new_bytes_array, 0) + cnt
            del word_counter[bytes_array]

        vocab[len(vocab)] = new_bytes
        merges.append(max_pair)

    return vocab, merges

if __name__ == "__main__":
    from time import time
    st = time()
    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 1000, special_tokens=["<|endoftext|>"])
    end = time()
    print(end-st)
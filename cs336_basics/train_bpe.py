import codecs
import os
import multiprocessing as mp
from typing import TypeAlias, Counter, BinaryIO, Generator
import regex


Vocab: TypeAlias = dict[int, bytes]
Merges: TypeAlias = list[tuple[bytes, bytes]]

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _process_span(
        path: str | os.PathLike,
        start: int,
        end: int,
        special_tokens: list[str],
) -> Counter[tuple[bytes, ...]]:
    r = regex.compile(PAT, flags=regex.V1)
    counts: Counter[tuple[bytes, ...]] = Counter()

    buf_size = 8 << 20 # 8MB
    dec = codecs.getincrementaldecoder('utf-8')()
    carry = ""  # blok sonundan taşan yarım token metni
    special_carry = ""

    with open(path, 'rb', buffering=0) as f:
        f.seek(start)
        left = end - start
        while left > 0:
            buf = f.read(min(buf_size, left))
            if not buf:
                break
            left -= len(buf)

            text = special_carry + carry + dec.decode(buf, final=False)
            carry = ""
            special_carry = ""

            # Decode special tokens to strings for comparison with text

            # buffer sonunda special token var mı kontrol edelim.
            for special_token in special_tokens:
                for i in range(len(special_token) - 1, 0, -1):
                    if text.endswith(special_token[:i]):
                        special_carry = special_token[:i]
                        text = text[:-i]
                        break

            # split for special tokens
            pattern = "|".join(map(regex.escape, special_tokens))
            segments = regex.split(pattern, text)

            for i, segment in enumerate(segments):
                if i < len(segments) - 1:  # Son segment değilse
                    for m in r.finditer(segment, partial=False):
                        token = tuple(bytes([x]) for x in m.group(0).encode('utf-8'))
                        counts[token] += 1
                else:
                    for m in r.finditer(segment, partial=True):
                        if m.partial:
                            carry = segment[m.start():]
                            break
                        token = tuple(bytes([x]) for x in m.group(0).encode('utf-8'))
                        counts[token] += 1

        # son batch'i işleyelim
        text = carry + dec.decode(b"", final=True)
        pattern = "|".join(map(regex.escape, special_tokens))
        segments = regex.split(pattern, text)
        for segment in segments:
            for m in r.finditer(segment, partial=True):
                if m.partial:
                    break
                token = tuple(bytes([x]) for x in m.group(0).encode('utf-8'))
                counts[token] += 1

    return counts

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
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


def pre_tokenize(input_path: str | os.PathLike, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    num_workers = os.cpu_count()

    # find spans to distribute to processes
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext|>")
    args_iter = ((input_path, s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:]))

    pre_token_counts: Counter[tuple[bytes, ...]] = Counter()
    with mp.get_context('spawn').Pool(processes=num_workers) as pool:
        for res in pool.starmap(_process_span, args_iter, chunksize=1):
            pre_token_counts.update(res)

    return pre_token_counts


def get_pairs_in_token(token: tuple[bytes, ...]) -> Generator[tuple[bytes, bytes], None, None]:
    for i in range(len(token) - 1):
        yield token[i], token[i + 1]


def merge_token(token: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    new_token: tuple[bytes, ...] = ()
    i = 0
    while i < len(token):
        if i < len(token) - 1 and token[i] == pair[0] and token[i + 1] == pair[1]:
            new_token += (token[i] + token[i + 1],)
            i += 2
        else:
            new_token += (token[i],)
            i += 1

    return new_token

def _merge_vocabulary(
        vocab: Vocab,
        max_vocab_length: int,
        pre_tokenized_corpus: Counter[tuple[bytes, ...]],
) -> Merges:
    pairs: Counter[tuple[bytes, bytes]] = Counter()
    occs: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    merges: Merges = []

    # initial scan for pairs
    for key, value in pre_tokenized_corpus.items():
        for i in range(len(key) - 1):
            pair = (key[i], key[i+1])
            pairs[pair] += value
            if pair not in occs:
                occs[pair] = set()
            occs[pair].add(key)

    # incremental merge
    while len(vocab) < max_vocab_length:
        most_common, _ = max(pairs.items(), key=lambda x: (x[1], x[0]))
        new_vocabulary_item = (most_common[0] + most_common[1])

        # calculate the pair changes from affected occurrences
        pair_changes: Counter[tuple[bytes, bytes]] = Counter()
        affected_occs = occs[most_common].copy()
        for affected_occ in affected_occs:
            # Merge the token first to see what actually happens
            new_token = merge_token(affected_occ, most_common)

            # Count all pairs in old token
            old_pairs: Counter[tuple[bytes, bytes]] = Counter(get_pairs_in_token(affected_occ))
            # Count all pairs in new token
            new_pairs: Counter[tuple[bytes, bytes]] = Counter(get_pairs_in_token(new_token))

            # Calculate the difference
            freq = pre_tokenized_corpus[affected_occ]
            for pair, count in old_pairs.items():
                pair_changes[pair] -= count * freq
            for pair, count in new_pairs.items():
                pair_changes[pair] += count * freq


        # now replace corpus items affected by merge
        for affected_occ in affected_occs:
            pre_tokenized_corpus[merge_token(affected_occ, most_common)] = pre_tokenized_corpus[affected_occ]
            del pre_tokenized_corpus[affected_occ]

        # now apply pair incr/decrs
        for pair, change in pair_changes.items():
            pairs[pair] += change
            if pairs[pair] == 0:
                del pairs[pair]

        # update occs - first remove old tokens
        for affected_occ in affected_occs:
            for i in range(len(affected_occ) - 1):
                old_pair = (affected_occ[i], affected_occ[i+1])
                if old_pair in occs:
                    occs[old_pair].discard(affected_occ)
                    if len(occs[old_pair]) == 0:
                        del occs[old_pair]

            new_token = merge_token(affected_occ, most_common)
            for i in range(len(new_token) - 1):
                new_pair = (new_token[i], new_token[i+1])
                if new_pair not in occs:
                    occs[new_pair] = set()
                occs[new_pair].add(new_token)


        # add new item to vocab
        vocab[len(vocab)] = new_vocabulary_item

        # add most common pair to merges
        merges.append(most_common)

    return merges

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[Vocab, Merges]:
    """
    Train a byte-pair encoding (BPE) tokenizer.
    """

    vocab: Vocab = {}

    for special_token in special_tokens:
        vocab[len(vocab)] = bytes(special_token, 'utf-8')

    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    pre_tokenized_corpus = pre_tokenize(input_path, special_tokens)
    merges = _merge_vocabulary(vocab, vocab_size, pre_tokenized_corpus)

    return vocab, merges

if __name__ == "__main__":
    train_bpe("./data/owt_valid.txt", 10000, special_tokens=['<|endoftext|>'])

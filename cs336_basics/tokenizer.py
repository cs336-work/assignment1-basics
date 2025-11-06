import gzip
import pickle
import regex
from typing import Iterable, Iterator, Self




class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None
    ):
        self._vocab = vocab
        self._merges = merges
        self._special_tokens = set(special_tokens) if special_tokens else set()

        # create reverse vocab
        self._reverse_vocab: dict[bytes, int] = dict((token, codepoint) for codepoint, token in vocab.items())

        # add special tokens to vocab
        for special_token in self._special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in self._reverse_vocab:
                vocab[len(vocab)] = byte_encoded_special_token
                self._reverse_vocab[byte_encoded_special_token] = len(vocab) - 1

        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._r = regex.compile(pattern, flags=regex.V1)


    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None
    ) -> Self:
        with gzip.open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)

        with gzip.open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        out: list[int] = []

        # split text with special tokens
        segments = [text]
        if len(self._special_tokens) > 0:
            sorted_special_tokens = sorted(self._special_tokens, key=len, reverse=True)
            split_pattern = "|".join(map(regex.escape, sorted_special_tokens))

            segments = regex.split(f"({split_pattern})", text)

        for segment in segments:
            if segment in self._special_tokens:
                out.append(self._reverse_vocab[segment.encode()])
                continue

            # get pre-tokenized text
            pre_tokens: list[tuple[bytes, ...]] = []
            for m in self._r.finditer(segment):
                token = tuple(bytes([x]) for x in m.group(0).encode('utf-8'))
                pre_tokens.append(token)

            for pre_token in pre_tokens:
                for merge in self._merges:
                    for i in range(len(pre_token) - 1):
                        if pre_token[i:i+2] == merge:
                            pre_token = pre_token[:i] + (merge[0] + merge[1],) + pre_token[i+2:]

                    if len(pre_token) == 1:
                        break

                for pre_token_part in pre_token:
                    out.append(self._reverse_vocab[pre_token_part])

        return out


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)


    def decode(self, tokens: list[int]) -> str:
        out = b''.join(self._vocab[token] for token in tokens)
        return out.decode("utf-8", errors="replace")


def test_encode():
    """Test function for the encode method based on the example."""
    # Create vocabulary from the example
    vocab = {
        0: b' ',
        1: b'a',
        2: b'c',
        3: b'e',
        4: b'h',
        5: b't',
        6: b'th',
        7: b' c',
        8: b' a',
        9: b'the',
        10: b' at',
        11: b'<|endoftext|>'
    }

    # Create merges from the example
    merges = [
        (b't', b'h'),      # merge 1
        (b' ', b'c'),      # merge 2
        (b' ', b'a'),      # merge 3
        (b'th', b'e'),     # merge 4
        (b' a', b't'),     # merge 5
    ]

    # Create tokenizer with special tokens
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])

    # Test 1: Basic encoding without special tokens
    text = "the cat ate"
    tokens = tokenizer.encode(text)

    print(f"Test 1 - Basic encoding")
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Expected: [9, 7, 1, 5, 10, 3]")

    # Test decoding
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: '{decoded}'")

    # Verify expected output
    expected = [9, 7, 1, 5, 10, 3]
    assert tokens == expected

    # Verify round-trip
    assert decoded == text, f"Round-trip failed: '{text}' != '{decoded}'"
    print("✓ Test 1 passed!\n")

    # Test 2: Encoding with special tokens
    text_with_special = "the<|endoftext|> cat"
    tokens_special = tokenizer.encode(text_with_special)

    print(f"Test 2 - Encoding with special tokens")
    print(f"Text: '{text_with_special}'")
    print(f"Tokens: {tokens_special}")

    # Decode each token to see if special token is preserved
    tokenized_string = [tokenizer.decode([x]) for x in tokens_special]
    print(f"Tokenized string: {tokenized_string}")
    print(f"Special token count: {tokenized_string.count('<|endoftext|>')}")

    # Verify special token is preserved
    assert tokenized_string.count("<|endoftext|>") == 1, "Special token not preserved!"

    # Verify round-trip
    decoded_special = tokenizer.decode(tokens_special)
    print(f"Decoded: '{decoded_special}'")
    assert decoded_special == text_with_special, f"Round-trip failed: '{text_with_special}' != '{decoded_special}'"
    print("✓ Test 2 passed!")

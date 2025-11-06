import gzip
import pickle

from cs336_basics.train_bpe import train_bpe
import time
import psutil
import os

if __name__ == "__main__":
    # Get process for memory monitoring
    process = psutil.Process(os.getpid())

    # Record initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Start timing
    start_time = time.time()

    # Train BPE
    vocab, merges = train_bpe("./data/TinyStoriesV2-GPT4-train.txt", 10000, special_tokens=['<|endoftext|>'])

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Record final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Write vocab to file
    with gzip.open("train_bpe_owl_vocab.pkl.gz", "wb") as f:
        pickle.dump(vocab, f, protocol=5)

    # Write merges to file
    with gzip.open("train_bpe_owl_merges.pkl.gz", "wb") as f:
        pickle.dump(merges, f, protocol=5)

    # Print statistics
    print("=" * 60)
    print("BPE Training Statistics")
    print("=" * 60)
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"\nExecution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"\nMemory usage:")
    print(f"  Initial: {initial_memory:.2f} MB")
    print(f"  Final: {final_memory:.2f} MB")
    print(f"  Increase: {final_memory - initial_memory:.2f} MB")
    print(f"\nOutput files:")
    print(f"  Vocab saved to: train_bpe_owl_vocab.pkl.gz")
    print(f"  Merges saved to: train_bpe_owl_merges.pkl.gz")
    print("=" * 60)


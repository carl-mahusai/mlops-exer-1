from collections import Counter

def build_vocab_util(texts, max_vocab_size):
    counter = Counter()

    for text in texts:
        counter.update(
            str(text).lower().split()
        )

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for token, _ in counter.most_common(
        max_vocab_size - 2
    ):
        vocab[token] = len(vocab)

    return vocab
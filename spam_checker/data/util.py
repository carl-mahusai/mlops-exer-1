import spacy
from collections import Counter
from collections import defaultdict
from spam_checker.util import build_token_list


# nlp = spacy.load("en_core_web_sm")

# doc = nlp("spaCy is successfully working in sms spam!")
# for token in doc:
#     print(token.text, token.pos_)

# def clean_text(text):
#     doc = nlp(text)  # Process the text with spaCy
#     cleaned_tokens = []
#     for token in doc:
#         # Remove stop words, punctuation, and retain only alphabetic words
#         if not token.is_stop and not token.is_punct and token.is_alpha:
#             cleaned_tokens.append(token.lemma_)  # Append the lemmatized version of the token
#     return " ".join(cleaned_tokens)

# def spacy_tokenizer(text):
#     # Use the clean_lemm function first
#     cleaned_text = clean_text(text)
#     # Then tokenize the cleaned text
#     return cleaned_text.split()

# def build_vocab_util(texts, max_vocab_size, min_freq=3, specials=('<PAD>', '<UNK>')):

#     text_iterator = texts

#     vocab = {
#         "<PAD>": 0,
#         "<UNK>": 1,
#     }

#     token_counts = defaultdict(int)
#     for text in text_iterator:
#         for token in spacy_tokenizer(text):
#             token_counts[token] += 1

#     vocab = {token: idx for idx, (token, count) in enumerate(token_counts.items()) if count >= min_freq}
#     # for special in specials:
#     #     if special not in vocab:
#     #         vocab[special] = len(vocab)

#     print(vocab)
#     return vocab

def build_vocab_util(texts, max_vocab_size):
    counter = Counter()
    # tokens = build_token_list(text)
    for text in texts:
        counter.update(
            # str(text).lower().split()
            # tokens
            build_token_list(text)
        )

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for token, _ in counter.most_common(
        max_vocab_size - 2
    ):
        vocab[token] = len(vocab)
    # print(vocab)
    return vocab
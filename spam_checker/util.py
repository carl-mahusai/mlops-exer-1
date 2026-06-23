import contextlib
import hashlib
import os
import re
from pathlib import Path
from typing import Union

from urllib.request import urlretrieve
from tqdm import tqdm

import spacy

@contextlib.contextmanager
def temporary_working_directory(working_dir: Union[str, Path]):
    """Temporarily switches to a directory, then returns to the original directory on exit."""
    curdir = os.getcwd()
    os.chdir(working_dir)
    try:
        yield
    finally:
        os.chdir(curdir)


def compute_sha256(filename: Union[Path, str]):
    """Return SHA256 checksum of a file."""
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parameters
        ----------
        blocks: int, optional
            Number of blocks transferred so far [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


def download_url(url, filename):
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # noqa: S310

nlp = spacy.load("en_core_web_sm")

doc = nlp("spaCy is successfully working in sms spam!")
for token in doc:
    print(token.text, token.pos_)

def clean_text(text):
    doc = nlp(text)  # Process the text with spaCy
    cleaned_tokens = []
    for token in doc:
        # Remove stop words, punctuation, and retain only alphabetic words
        if not token.is_stop and not token.is_punct and token.is_alpha:
            cleaned_tokens.append(token.lemma_)  # Append the lemmatized version of the token
    return " ".join(cleaned_tokens)

# def spacy_tokenizer(text):
#     # Use the clean_lemm function first
#     cleaned_text = clean_text(text)
#     # Then tokenize the cleaned text
#     return cleaned_text.split()

def build_token_list(text):
    # return str(text).lower().split()
    # text = str(text).lower()
    # text = re.sub(r"[^\w\s]", "", text)
    # return text.split()
    # Use the clean_lemm function first
    cleaned_text = clean_text(text)
    # Then tokenize the cleaned text
    return cleaned_text.split()
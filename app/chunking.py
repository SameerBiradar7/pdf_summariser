# app/chunking.py

from typing import List
import re
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")   # same tokenizer used by many LLMs

def clean_text(text: str) -> str:
    """
    Remove noisy sections like references, bibliography, and excessive blank lines.
    """
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(Bibliography|References)[\s\S]*$', '', text, flags=re.I)
    return text.strip()


def count_tokens(text: str) -> int:
    """
    Count tokens with tiktoken.
    """
    return len(ENC.encode(text))


def chunk_text(
    text: str,
    max_tokens: int = 900,
    overlap_tokens: int = 150
) -> List[str]:
    """
    Splits text into overlapping chunks based on token length.
    Good defaults:
      - 700–1200 tokens per chunk
      - 100–200 token overlap
    """
    tokens = ENC.encode(text)
    chunks = []

    start = 0
    end = max_tokens

    while start < len(tokens):
        chunk_tokens = tokens[start:end]
        chunk_text = ENC.decode(chunk_tokens)
        chunks.append(chunk_text)

        # next window with overlap
        start = end - overlap_tokens
        if start < 0:
            start = 0
        end = start + max_tokens

    return chunks


def prepare_chunks(raw_text: str) -> List[str]:
    """
    Full pipeline: clean -> chunk
    """
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned)
    return chunks
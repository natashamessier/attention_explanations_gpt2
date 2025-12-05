"""
Preprocessing utilities for text data.
"""

import re
from typing import List, Dict, Optional

# Constants
DEFAULT_MAX_LENGTH = 512  # Default maximum sequence length


def clean_text(text: str, remove_html: bool = True, lowercase: bool = False) -> str:
    """
    Clean text data for model input.

    Args:
        text: Input text to clean
        remove_html: Whether to remove HTML tags
        lowercase: Whether to convert to lowercase

    Returns:
        Cleaned text

    Raises:
        ValueError: If text is empty or None
    """
    if not text or not isinstance(text, str):
        raise ValueError("text must be a non-empty string")

    if remove_html:
        # Remove HTML tags (common in IMDb reviews)
        text = re.sub(r'<br\s*/?>', ' ', text)
        text = re.sub(r'<[^>]+>', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if lowercase:
        text = text.lower()

    return text


def truncate_text(text: str, max_length: int = DEFAULT_MAX_LENGTH,
                  tokenizer=None, strategy: str = "start") -> str:
    """
    Truncate text to fit model's max input length.

    Args:
        text: Input text
        max_length: Maximum length (in tokens if tokenizer provided, else chars)
        tokenizer: Optional tokenizer to count tokens
        strategy: 'start', 'end', or 'middle' - which part to keep

    Returns:
        Truncated text
    """
    if tokenizer is not None:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_length:
            return text

        if strategy == "start":
            tokens = tokens[:max_length]
        elif strategy == "end":
            tokens = tokens[-max_length:]
        elif strategy == "middle":
            half = max_length // 2
            tokens = tokens[:half] + tokens[-half:]

        return tokenizer.decode(tokens)
    else:
        # Fallback to character-based truncation
        if len(text) <= max_length:
            return text

        if strategy == "start":
            return text[:max_length]
        elif strategy == "end":
            return text[-max_length:]
        elif strategy == "middle":
            half = max_length // 2
            return text[:half] + text[-half:]

    return text


def prepare_batch(texts: List[str],
                  labels: List[int],
                  tokenizer,
                  max_length: int = DEFAULT_MAX_LENGTH,
                  clean: bool = True) -> Dict:
    """
    Prepare a batch of texts for model input.

    Args:
        texts: List of input texts
        labels: List of labels
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        clean: Whether to clean text before tokenization

    Returns:
        Dictionary with tokenized inputs and labels

    Raises:
        ValueError: If texts or labels are empty or have mismatched lengths
    """
    if not texts:
        raise ValueError("texts list cannot be empty")

    if not labels:
        raise ValueError("labels list cannot be empty")

    if len(texts) != len(labels):
        raise ValueError(f"Mismatch: {len(texts)} texts but {len(labels)} labels")

    if clean:
        texts = [clean_text(t) for t in texts]

    # Tokenize
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    encoding['labels'] = labels
    return encoding


if __name__ == "__main__":
    # Test preprocessing
    sample_text = "<br />This movie was <b>amazing</b>!   Best film ever."
    print("Original:", sample_text)
    print("Cleaned:", clean_text(sample_text))

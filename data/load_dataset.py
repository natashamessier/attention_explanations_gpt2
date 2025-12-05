"""
Load and prepare the IMDb dataset for interpretability analysis.
"""

from datasets import load_dataset
from typing import Dict, List, Optional
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)


def load_imdb_dataset(split: str = "train", num_samples: Optional[int] = None, 
                      use_shortest: bool = True) -> Dict:
    """
    Load the IMDb sentiment analysis dataset from HuggingFace.

    Args:
        split: Dataset split to load ('train', 'test', or 'unsupervised')
        num_samples: Optional number of samples to load (for testing/debugging)
        use_shortest: If True, return n samples from the 50 shortest reviews.
                     If False, return n random samples (default behavior).

    Returns:
        Dictionary containing the dataset
    """
    logger.info(f"Loading IMDb dataset ({split} split)...")

    # Load from HuggingFace
    dataset = load_dataset("stanfordnlp/imdb", split=split)

    if use_shortest:
        # Find the 50 shortest samples, then return n of them
        logger.info("Finding 50 shortest reviews...")
        
        # Add length column
        def calculate_review_length(example):
            """Calculate the length of the review text."""
            return {"review_length": len(example["text"])}
        
        dataset_with_lengths = dataset.map(calculate_review_length)
        
        # Sort by length and get 50 shortest
        sorted_dataset = dataset_with_lengths.sort("review_length")
        shortest_50 = sorted_dataset.select(range(min(50, len(sorted_dataset))))
        
        logger.info(f"Found 50 shortest reviews (length range: {shortest_50[0]['review_length']}-{shortest_50[-1]['review_length']} chars)")
        
        # If num_samples specified, randomly sample from the 50 shortest
        if num_samples is not None:
            if num_samples > len(shortest_50):
                logger.warning(f"Requested {num_samples} samples but only {len(shortest_50)} shortest available. Returning all {len(shortest_50)}.")
                num_samples = len(shortest_50)
            
            # Shuffle and select n
            shortest_50 = shortest_50.shuffle(seed=42)
            dataset = shortest_50.select(range(num_samples))
            logger.info(f"Selected {num_samples} samples from 50 shortest reviews")
        else:
            dataset = shortest_50
            logger.info(f"Returning all {len(dataset)} shortest reviews")
    else:
        # Original behavior: random samples
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            logger.info(f"Selected {num_samples} random samples")
        else:
            logger.info(f"Returning all {len(dataset)} examples")

    logger.info(f"Loaded {len(dataset)} examples")
    return dataset


def preview_examples(dataset, n: int = 3) -> None:
    """
    Print a few examples from the dataset for inspection.

    Args:
        dataset: The loaded dataset
        n: Number of examples to preview
    """
    print(f"\n{'='*80}")
    print(f"Preview of {n} examples:")
    print(f"{'='*80}\n")

    for i in range(min(n, len(dataset))):
        example = dataset[i]
        label = "Positive" if example['label'] == 1 else "Negative"
        text = example['text'][:200] + "..." if len(example['text']) > 200 else example['text']

        print(f"Example {i+1}:")
        print(f"Label: {label}")
        print(f"Text: {text}")
        print(f"{'-'*80}\n")


if __name__ == "__main__":
    # Test the loader - random samples
    print("Testing random samples:")
    dataset = load_imdb_dataset(split="test", num_samples=10, use_shortest=False)
    preview_examples(dataset, n=3)
    
    print("\n" + "="*80)
    print("Testing shortest samples:")
    # Test shortest samples
    dataset_short = load_imdb_dataset(split="test", num_samples=10, use_shortest=True)
    preview_examples(dataset_short, n=3)

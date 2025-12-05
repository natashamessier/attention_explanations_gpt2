"""
Utility functions for model operations.
"""

import torch
from typing import List, Dict


def get_model_info(model) -> Dict:
    """
    Get information about the model architecture.

    Args:
        model: HuggingFace model

    Returns:
        Dictionary with model information
    """
    config = model.config

    info = {
        'model_type': config.model_type,
        'vocab_size': config.vocab_size,
        'n_layers': config.n_layer if hasattr(config, 'n_layer') else config.num_hidden_layers,
        'n_heads': config.n_head if hasattr(config, 'n_head') else config.num_attention_heads,
        'hidden_size': config.hidden_size if hasattr(config, 'hidden_size') else config.n_embd,
        'max_position_embeddings': config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else config.n_positions,
        'total_params': sum(p.numel() for p in model.parameters())
    }

    return info


def print_model_summary(model) -> None:
    """
    Print a summary of the model architecture.

    Args:
        model: HuggingFace model
    """
    info = get_model_info(model)

    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    for key, value in info.items():
        print(f"{key:30s}: {value}")
    print("="*60 + "\n")


def count_parameters(model, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in the model.

    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters

    Returns:
        Total number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_device_info() -> Dict:
    """
    Get information about available compute devices.

    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1e9  # GB
        info['memory_reserved'] = torch.cuda.memory_reserved(0) / 1e9  # GB

    return info


def batch_iterator(data: List, batch_size: int):
    """
    Create batches from a list of data.

    Args:
        data: List of data items
        batch_size: Size of each batch

    Yields:
        Batches of data
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


if __name__ == "__main__":
    # Test device info
    device_info = get_device_info()
    print("\nDevice Information:")
    print("-" * 40)
    for key, value in device_info.items():
        print(f"{key}: {value}")

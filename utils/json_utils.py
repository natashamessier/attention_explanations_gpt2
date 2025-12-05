"""
JSON utilities for saving and loading example data.
"""

import json
import numpy as np
from typing import Dict, List, Any
from pathlib import Path


def convert_attentions_to_dict(attentions: tuple, n_layers: int, n_heads: int, seq_len: int) -> Dict:
    """
    Convert attention tensors to nested dictionary format for JSON.
    
    Args:
        attentions: Tuple of attention tensors [n_layers, batch, n_heads, seq_len, seq_len]
        n_layers: Number of layers
        n_heads: Number of heads
        seq_len: Sequence length
        
    Returns:
        Nested dictionary: {"layer_0": {"head_0": [...], "head_1": [...]}, ...}
    """
    attention_dict = {}
    
    for layer_idx in range(n_layers):
        layer_key = f"layer_{layer_idx}"
        layer_attn = attentions[layer_idx][0]  # [n_heads, seq_len, seq_len]
        
        head_dict = {}
        for head_idx in range(n_heads):
            head_key = f"head_{head_idx}"
            head_attn = layer_attn[head_idx].detach().cpu().numpy().tolist()
            head_dict[head_key] = head_attn
        
        attention_dict[layer_key] = head_dict
    
    return attention_dict


def save_example_json(data: Dict, filepath: str) -> None:
    """
    Save example data to JSON file.
    
    Args:
        data: Dictionary with text, tokens, attentions, ig_scores, shap_values, prediction
        filepath: Path to save JSON file
    """
    # Convert numpy arrays to lists
    json_data = {
        'text': data['text'],
        'tokens': data['tokens'],
        'attentions': data['attentions'],
        'ig_scores': data['ig_scores'].tolist() if isinstance(data['ig_scores'], np.ndarray) else data['ig_scores'],
        'shap_values': data['shap_values'].tolist() if isinstance(data['shap_values'], np.ndarray) else data['shap_values'],
        'prediction': data['prediction']
    }
    
    # Add actual_label if present (optional, for IMDb examples)
    if 'actual_label' in data:
        json_data['actual_label'] = data['actual_label']
    
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=2)


def load_example_json(filepath: str) -> Dict:
    """
    Load example data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary with loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays for scores
    data['ig_scores'] = np.array(data['ig_scores'])
    data['shap_values'] = np.array(data['shap_values'])
    
    return data


def list_example_files(results_dir: str = "results") -> List[str]:
    """
    List all example JSON files in results directory.
    
    Args:
        results_dir: Directory containing JSON files
        
    Returns:
        List of file paths
    """
    results_path = Path(results_dir)
    json_files = list(results_path.glob("example_*.json"))
    return sorted([str(f) for f in json_files])


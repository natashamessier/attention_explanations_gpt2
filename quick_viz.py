"""
Terminal command for quick visualization testing.
Usage: python quick_viz.py --text "Your text here"
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from pipeline import run_pipeline
from utils.json_utils import load_example_json
from data.load_dataset import load_imdb_dataset


def create_heatmap(tokens: list, attention_matrix: np.ndarray, save_path: str):
    """
    Create token-token attention heatmap.
    
    Args:
        tokens: List of tokens
        attention_matrix: Attention matrix [seq_len, seq_len]
        save_path: Path to save image
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Truncate if too many tokens for readability
    max_tokens = 30
    if len(tokens) > max_tokens:
        tokens_display = tokens[:max_tokens]
        attention_display = attention_matrix[:max_tokens, :max_tokens]
    else:
        tokens_display = tokens
        attention_display = attention_matrix
    
    # Create heatmap
    sns.heatmap(
        attention_display,
        xticklabels=tokens_display,
        yticklabels=tokens_display,
        cmap='Blues',
        cbar_kws={'label': 'Attention Weight'},
        ax=ax,
        fmt='.2f'
    )
    
    ax.set_title('Token-Token Attention Heatmap (Last Layer, Averaged Heads)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Attended To', fontsize=12)
    ax.set_ylabel('Attending From', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap to {save_path}")


def create_importance_bars(tokens: list, attention_scores: np.ndarray,
                          ig_scores: np.ndarray, shap_values: np.ndarray,
                          save_path: str):
    """
    Create token importance bar chart comparing methods.
    
    Args:
        tokens: List of tokens
        attention_scores: Attention importance scores
        ig_scores: IG importance scores
        shap_values: SHAP importance scores
        save_path: Path to save image
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Truncate if too many tokens
    max_tokens = 30
    if len(tokens) > max_tokens:
        tokens_display = tokens[:max_tokens]
        attention_display = attention_scores[:max_tokens]
        ig_display = ig_scores[:max_tokens]
        shap_display = shap_values[:max_tokens]
    else:
        tokens_display = tokens
        attention_display = attention_scores
        ig_display = ig_scores
        shap_display = shap_values
    
    x = np.arange(len(tokens_display))
    width = 0.25
    
    ax.bar(x - width, attention_display, width, label='Attention', alpha=0.8, color='#3498db')
    ax.bar(x, ig_display, width, label='Integrated Gradients', alpha=0.8, color='#e74c3c')
    ax.bar(x + width, shap_display, width, label='SHAP', alpha=0.8, color='#2ecc71')
    
    ax.set_xlabel('Tokens', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title('Token Importance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tokens_display, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved importance bars to {save_path}")


def compute_attention_importance(attention_dict: dict, layer: int = -1, head: str = "average") -> np.ndarray:
    """
    Compute token importance from attention weights.
    
    Args:
        attention_dict: Nested attention dictionary
        layer: Layer index (-1 for last layer)
        head: Head index or "average"
        
    Returns:
        Array of importance scores [seq_len]
    """
    # Get layer
    n_layers = len([k for k in attention_dict.keys() if k.startswith('layer_')])
    if layer == -1:
        layer_key = f"layer_{n_layers - 1}"
    else:
        layer_key = f"layer_{layer}"
    
    layer_attn = attention_dict[layer_key]
    
    # Get head(s)
    if head == "average":
        # Average across all heads
        head_matrices = []
        for head_key in sorted(layer_attn.keys()):
            head_matrices.append(np.array(layer_attn[head_key]))
        attention_matrix = np.mean(head_matrices, axis=0)
    else:
        head_key = f"head_{head}"
        attention_matrix = np.array(layer_attn[head_key])
    
    # Compute importance: sum of attention received by each token
    importance = attention_matrix.sum(axis=0)  # Sum over "attending from" dimension
    
    # Normalize
    if importance.max() > 0:
        importance = (importance - importance.min()) / (importance.max() - importance.min())
    
    return importance


def main():
    parser = argparse.ArgumentParser(description='Quick visualization for interpretability analysis')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to analyze')
    parser.add_argument('--imdb', action='store_true',
                       help='Use IMDb dataset example')
    parser.add_argument('--example-id', type=int, default=1,
                       help='IMDb example ID (if using --imdb)')
    parser.add_argument('--json-file', type=str, default=None,
                       help='Load from existing JSON file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for JSON and images')
    
    args = parser.parse_args()
    
    print("="*80)
    print("QUICK VISUALIZATION")
    print("="*80)
    print()
    
    # Determine input source
    if args.json_file:
        # Load from existing JSON
        json_path = args.json_file
        print(f"Loading from existing JSON: {json_path}")
    elif args.imdb:
        # Load IMDb example
        print(f"Loading IMDb example {args.example_id}...")
        # Load enough samples to get the requested example
        dataset = load_imdb_dataset(split="test", num_samples=args.example_id)
        if args.example_id > len(dataset):
            print(f"Error: Only {len(dataset)} examples available in test split")
            return
        example = dataset[args.example_id - 1]  # 0-indexed
        text = example['text']
        label = "positive" if example['label'] == 1 else "negative"
        print(f"Label: {label}")
        print(f"Text: {text[:100]}...")
        print()
        
        # Run pipeline with actual label
        print("Step 1: Running pipeline...")
        json_path = run_pipeline(
            text, 
            output_dir=args.output_dir, 
            example_id=args.example_id,
            actual_label=label
        )
        print()
    elif args.text:
        # Use provided text
        print("Step 1: Running pipeline...")
        json_path = run_pipeline(args.text, output_dir=args.output_dir)
        print()
    else:
        # Default: use a simple example
        default_text = "This movie was absolutely fantastic and entertaining!"
        print("No input specified. Using default example...")
        print()
        print("Step 1: Running pipeline...")
        json_path = run_pipeline(default_text, output_dir=args.output_dir)
        print()
    
    # Step 2: Load JSON
    print("Step 2: Loading results...")
    data = load_example_json(json_path)
    print(f"  Loaded: {len(data['tokens'])} tokens")
    print()
    
    # Extract example ID from JSON path for unique filenames
    json_filename = Path(json_path).stem  # e.g., "example_1"
    if json_filename.startswith("example_"):
        example_id = json_filename.replace("example_", "")
    else:
        # Fallback: use a generic identifier
        example_id = "custom"
    
    # Step 3: Create visualizations
    print("Step 3: Creating visualizations...")
    
    # Get attention matrix for heatmap (last layer, averaged heads)
    attention_dict = data['attentions']
    n_layers = len([k for k in attention_dict.keys() if k.startswith('layer_')])
    last_layer_key = f"layer_{n_layers - 1}"
    
    # Average across heads
    head_matrices = []
    for head_key in sorted(attention_dict[last_layer_key].keys()):
        head_matrices.append(np.array(attention_dict[last_layer_key][head_key]))
    attention_matrix = np.mean(head_matrices, axis=0)
    
    # Create heatmap with example ID in filename
    heatmap_path = Path(args.output_dir) / f"quick_viz_heatmap_{example_id}.png"
    create_heatmap(data['tokens'], attention_matrix, str(heatmap_path))
    
    # Compute attention importance scores
    attention_scores = compute_attention_importance(attention_dict, layer=-1, head="average")
    
    # Create importance bars with example ID in filename
    bars_path = Path(args.output_dir) / f"quick_viz_bars_{example_id}.png"
    create_importance_bars(
        data['tokens'],
        attention_scores,
        data['ig_scores'],
        data['shap_values'],
        str(bars_path)
    )
    
    print()
    print("="*80)
    print("✓ VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  • {json_path}")
    print(f"  • {heatmap_path}")
    print(f"  • {bars_path}")


if __name__ == "__main__":
    main()


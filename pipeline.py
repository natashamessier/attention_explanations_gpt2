"""
Simple data pipeline: tokenize → GPT-2 → collect attentions/IG/SHAP → save JSON.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.model_utils import GPT2Model
from utils.interpretability import compute_ig, compute_shap
from utils.json_utils import convert_attentions_to_dict, save_example_json
from data.load_dataset import load_imdb_dataset
from data.preprocessing import clean_text


def run_pipeline(text: str, model_name: str = "gpt2", device: str = "auto", 
                 output_dir: str = "results", example_id: int = None,
                 actual_label: str = None) -> str:
    """
    Run the complete pipeline for a single text example.
    
    Args:
        text: Input text to analyze
        model_name: GPT-2 model name
        device: Device to run on
        output_dir: Directory to save JSON file
        example_id: Optional example ID (if None, auto-increments)
        actual_label: Optional actual label from dataset (e.g., "positive" or "negative")
        
    Returns:
        Path to saved JSON file
    """
    print(f"Processing: {text[:100]}...")
    
    # Clean text
    text = clean_text(text)
    
    # Initialize model
    print("Loading GPT-2 model...")
    model = GPT2Model(model_name=model_name, device=device)
    
    # Step 1: Tokenize and run GPT-2
    print("Running GPT-2 with attention extraction...")
    result = model.process_text(text)
    
    tokens = result['tokens']
    attentions = result['attentions']  # Tuple of [n_layers, batch, n_heads, seq_len, seq_len]
    logits = result['logits']
    
    # Get dimensions
    n_layers = len(attentions)
    n_heads = attentions[0][0].shape[0]  # [batch, n_heads, seq_len, seq_len]
    seq_len = attentions[0][0].shape[2]
    
    print(f"  Layers: {n_layers}, Heads: {n_heads}, Sequence length: {seq_len}")
    
    # Step 2: Get sentiment prediction
    print("Computing sentiment prediction...")
    prediction = model.get_sentiment_prediction(logits)
    print(f"  Prediction: {prediction}")
    
    # Step 3: Compute IG scores
    print("Computing Integrated Gradients...")
    ig_scores = compute_ig(model.model, model.tokenizer, text, device=str(model.device))
    print(f"  IG scores shape: {ig_scores.shape}")
    
    # Step 4: Compute SHAP scores
    print("Computing SHAP values...")
    shap_values = compute_shap(model.model, model.tokenizer, text, device=str(model.device))
    print(f"  SHAP values shape: {shap_values.shape}")
    
    # Step 5: Convert attentions to dict format
    print("Converting attention tensors to JSON format...")
    attention_dict = convert_attentions_to_dict(attentions, n_layers, n_heads, seq_len)
    
    # Step 6: Prepare data for JSON
    data = {
        'text': text,
        'tokens': tokens,
        'attentions': attention_dict,
        'ig_scores': ig_scores,
        'shap_values': shap_values,
        'prediction': prediction
    }
    
    # Add actual label if provided (for IMDb examples)
    if actual_label is not None:
        data['actual_label'] = actual_label
    
    # Step 7: Save JSON
    if example_id is None:
        # Auto-increment: find next available ID
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        existing_files = list(output_path.glob("example_*.json"))
        if existing_files:
            existing_ids = [int(f.stem.split('_')[1]) for f in existing_files if f.stem.split('_')[1].isdigit()]
            example_id = max(existing_ids) + 1 if existing_ids else 1
        else:
            example_id = 1
    
    filepath = Path(output_dir) / f"example_{example_id}.json"
    print(f"Saving to {filepath}...")
    save_example_json(data, str(filepath))
    
    print(f"✓ Pipeline complete! Saved to {filepath}")
    return str(filepath)


def process_imdb_examples(split: str = "test", num_examples: int = 10, 
                         output_dir: str = "results", device: str = "auto"):
    """
    Process multiple examples from IMDb dataset.
    
    Args:
        split: Dataset split ('train', 'test', or 'unsupervised')
        num_examples: Number of examples to process
        output_dir: Directory to save JSON files
        device: Device to run on
    """
    print("="*80)
    print(f"PROCESSING {num_examples} EXAMPLES FROM IMDb DATASET")
    print("="*80)
    print()
    
    # Load IMDb dataset
    print(f"Loading IMDb dataset ({split} split)...")
    dataset = load_imdb_dataset(split=split, num_samples=num_examples)
    print(f"Loaded {len(dataset)} examples\n")
    
    # Process each example
    for i, example in enumerate(dataset):
        text = example['text']
        label = "positive" if example['label'] == 1 else "negative"
        
        print(f"[{i+1}/{len(dataset)}] Processing example {i+1} (label: {label})...")
        print(f"  Text: {text[:100]}...")
        
        try:
            filepath = run_pipeline(
                text, 
                model_name="gpt2",
                device=device,
                output_dir=output_dir,
                example_id=i+1,
                actual_label=label
            )
            print(f"  ✓ Saved to {filepath}\n")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}\n")
            continue
    
    print("="*80)
    print(f"✓ PROCESSING COMPLETE! Processed {len(dataset)} examples")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process IMDb examples or single text')
    parser.add_argument('--text', type=str, default=None,
                       help='Single text to analyze (if not provided, uses IMDb dataset)')
    parser.add_argument('--imdb', action='store_true',
                       help='Process IMDb dataset examples')
    parser.add_argument('--split', type=str, default='test',
                       help='IMDb dataset split (train/test/unsupervised)')
    parser.add_argument('--num-examples', type=int, default=10,
                       help='Number of IMDb examples to process')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to run on')
    
    args = parser.parse_args()
    
    if args.text:
        # Process single text
        print("="*80)
        print("SIMPLE PIPELINE - Single Text Example")
        print("="*80)
        print()
        filepath = run_pipeline(args.text, output_dir=args.output_dir, device=args.device)
        print(f"\nExample JSON saved to: {filepath}")
    elif args.imdb:
        # Process IMDb examples
        process_imdb_examples(
            split=args.split,
            num_examples=args.num_examples,
            output_dir=args.output_dir,
            device=args.device
        )
    else:
        # Default: process a few IMDb examples
        print("No --text or --imdb specified. Processing 5 IMDb examples by default...")
        process_imdb_examples(
            split='test',
            num_examples=5,
            output_dir=args.output_dir,
            device=args.device
        )


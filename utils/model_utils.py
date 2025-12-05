"""
Simplified GPT-2 model wrapper for the pipeline.
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import Dict, List, Tuple
import numpy as np

DEFAULT_MAX_LENGTH = 512


class GPT2Model:
    """
    Simplified GPT-2 wrapper for interpretability analysis.
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto"):
        """
        Initialize GPT-2 model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("cuda", "cpu", or "auto")
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True
        )
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
        self.model.to(self.device)
        self.model.eval()
        
        self.config = self.model.config
    
    def process_text(self, text: str) -> Dict:
        """
        Process text: tokenize and run through model.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with tokens, input_ids, attentions, and logits
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH
        ).to(self.device)
        
        input_ids = encoding['input_ids']
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=encoding.get('attention_mask'),
                output_attentions=True,
                output_hidden_states=True
            )
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return {
            'text': text,
            'tokens': tokens,
            'input_ids': input_ids,
            'attentions': outputs.attentions,  # Tuple of [n_layers, batch, n_heads, seq_len, seq_len]
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states
        }
    
    def get_sentiment_prediction(self, logits: torch.Tensor) -> str:
        """
        Get sentiment prediction from logits using sentiment tokens.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size] or [batch, vocab_size]
            
        Returns:
            "positive" or "negative"
        """
        # Use last position for next-token prediction
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]  # [batch, vocab_size]
        
        # Sentiment tokens
        positive_tokens = [' positive', ' good', ' great', ' excellent', ' wonderful', ' amazing']
        negative_tokens = [' negative', ' bad', ' terrible', ' awful', ' horrible', ' poor']
        
        # Get token IDs
        positive_ids = [self.tokenizer.encode(t, add_special_tokens=False)[0] for t in positive_tokens]
        negative_ids = [self.tokenizer.encode(t, add_special_tokens=False)[0] for t in negative_tokens]
        
        # Extract logits
        positive_logits = logits[:, positive_ids].mean(dim=1)
        negative_logits = logits[:, negative_ids].mean(dim=1)
        
        # Predict
        sentiment_score = (positive_logits - negative_logits).item()
        return "positive" if sentiment_score > 0 else "negative"


"""
Simplified interpretability methods: IG and SHAP computation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from captum.attr import IntegratedGradients

# Default sentiment tokens
DEFAULT_SENTIMENT_TOKENS = {
    'positive': [' positive', ' good', ' great', ' excellent', ' wonderful', ' amazing'],
    'negative': [' negative', ' bad', ' terrible', ' awful', ' horrible', ' poor']
}

EPSILON = 1e-8


def compute_ig(model, tokenizer, text: str, device: str = "cpu", n_steps: int = 50) -> np.ndarray:
    """
    Compute Integrated Gradients attribution for text.
    
    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        text: Input text
        device: Device to run on
        n_steps: Number of integration steps
        
    Returns:
        Array of importance scores [seq_len]
    """
    # Get sentiment token IDs
    positive_token_ids = [
        tokenizer.encode(token, add_special_tokens=False)[0]
        for token in DEFAULT_SENTIMENT_TOKENS['positive']
    ]
    negative_token_ids = [
        tokenizer.encode(token, add_special_tokens=False)[0]
        for token in DEFAULT_SENTIMENT_TOKENS['negative']
    ]
    
    def forward_func(input_embeds: torch.Tensor) -> torch.Tensor:
        """Forward function for IG."""
        outputs = model(inputs_embeds=input_embeds)
        logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
        
        positive_logits = logits[:, positive_token_ids].mean(dim=1)
        negative_logits = logits[:, negative_token_ids].mean(dim=1)
        sentiment_score = positive_logits - negative_logits
        
        return sentiment_score
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    input_ids = inputs['input_ids']
    input_embeds = model.transformer.wte(input_ids)
    baseline_embeds = torch.zeros_like(input_embeds)
    
    # Compute IG
    ig = IntegratedGradients(forward_func)
    
    with torch.set_grad_enabled(True):
        attributions = ig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            n_steps=n_steps,
            return_convergence_delta=False
        )
    
    # Aggregate and normalize
    token_importance = attributions.abs().sum(dim=-1)[0].detach().cpu().numpy()
    token_importance = (token_importance - token_importance.min()) / \
                      (token_importance.max() - token_importance.min() + EPSILON)
    
    return token_importance


def compute_shap(model, tokenizer, text: str, device: str = "cpu") -> np.ndarray:
    """
    Compute SHAP attribution using token masking.
    
    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        text: Input text
        device: Device to run on
        
    Returns:
        Array of importance scores [seq_len]
    """
    # Get sentiment token IDs
    positive_token_ids = [
        tokenizer.encode(token, add_special_tokens=False)[0]
        for token in DEFAULT_SENTIMENT_TOKENS['positive']
    ]
    negative_token_ids = [
        tokenizer.encode(token, add_special_tokens=False)[0]
        for token in DEFAULT_SENTIMENT_TOKENS['negative']
    ]
    
    def get_sentiment_score(text_input: str) -> float:
        """Get sentiment score for text."""
        inputs = tokenizer(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            
            positive_logits = logits[:, positive_token_ids].mean(dim=1)
            negative_logits = logits[:, negative_token_ids].mean(dim=1)
            sentiment_score = (positive_logits - negative_logits).item()
        
        return sentiment_score
    
    # Tokenize to get tokens
    encoding = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoding['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Get baseline prediction
    original_score = get_sentiment_score(text)
    
    # Compute importance by masking each token
    importance_scores = []
    for i in range(len(input_ids)):
        # Create masked version
        masked_ids = torch.cat([input_ids[:i], input_ids[i+1:]])
        
        if len(masked_ids) > 0:
            masked_text = tokenizer.decode(masked_ids, skip_special_tokens=True)
            masked_score = get_sentiment_score(masked_text)
            importance = abs(original_score - masked_score)
        else:
            importance = abs(original_score)
        
        importance_scores.append(importance)
    
    importance_scores = np.array(importance_scores)
    
    # Normalize
    if importance_scores.max() > 0:
        importance_scores = importance_scores / importance_scores.max()
    
    return importance_scores


import torch
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd

MAX_LEN = 512

# Function to tokenize given sentences
def custom_tokenize(sentences, tokenizer, max_length=512):
    """
    Tokenize sentences and return input IDs
    """
    input_ids = []
    
    for sent in sentences:
        # Skip NaN values
        if pd.isna(sent) or not isinstance(sent, str):
            continue
        
        try:
            encoded_sent = tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
            )
        except (ValueError, TypeError):
            # Fallback for problematic sentences
            encoded_sent = tokenizer.encode(
                ' ',
                add_special_tokens=True,
                max_length=max_length,
            )
        
        input_ids.append(encoded_sent)
    
    return input_ids

# Create mask for the given inputs
def custom_att_masks(input_ids):
    """
    Create attention masks from input IDs
    """
    attention_masks = []
    
    for sent in input_ids:
        # Create the attention mask
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    
    return attention_masks

# Combine features and return input IDs and attention masks
def combine_features(sentences, tokenizer, max_length=512, labels=None):
    """
    Tokenize sentences and create attention masks
    Returns input IDs, attention masks, and optionally filtered labels
    """
    
    # Find valid sentence indices and filter NaN values
    valid_indices = []
    valid_sentences = []
    for idx, sent in enumerate(sentences):
        if pd.isna(sent) or not isinstance(sent, str):
            continue
        valid_indices.append(idx)
        valid_sentences.append(sent)
    
    if not valid_sentences:
        raise ValueError("No valid sentences found after filtering NaN values")
    
    # Tokenize
    input_ids = custom_tokenize(valid_sentences, tokenizer, max_length)
    attention_masks = custom_att_masks(input_ids)
    
    # Pad sequences to the same length
    max_len = max([len(x) for x in input_ids])
    
    padded_input_ids = []
    for seq in input_ids:
        padded_seq = seq + [0] * (max_len - len(seq))
        padded_input_ids.append(padded_seq[:max_len])
    
    padded_attention_masks = []
    for mask in attention_masks:
        padded_mask = mask + [0] * (max_len - len(mask))
        padded_attention_masks.append(padded_mask[:max_len])
    
    # Convert to tensors
    input_ids = torch.tensor(padded_input_ids)
    attention_masks = torch.tensor(padded_attention_masks)
    
    # Filter labels if provided
    if labels is not None:
        filtered_labels = [labels[i] for i in valid_indices]
        return input_ids, attention_masks, filtered_labels
    
    return input_ids, attention_masks

# Function to return dataloader
def return_dataloader(input_ids, labels, attention_masks, batch_size=8, is_train=True):
    """
    Create and return DataLoader from pre-processed input IDs, labels, and attention masks
    """
    
    # Convert labels to tensor if needed with long dtype for CrossEntropyLoss
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)
    
    # Create the DataLoader
    data = TensorDataset(input_ids, attention_masks, labels)
    
    if is_train:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    
    dataloader = DataLoader(
        data,
        sampler=sampler,
        batch_size=batch_size
    )
    
    return dataloader

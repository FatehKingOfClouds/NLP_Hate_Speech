import random
import numpy as np
import torch
import time
from datetime import datetime

def fix_the_random(seed_val=42):
    """
    Fix random seeds for reproducibility
    """
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def save_model(model, tokenizer, params):
    """
    Save fine-tuned model and tokenizer
    
    Params:
    - model: the PyTorch model to save
    - tokenizer: the tokenizer to save
    - params: dictionary containing 'path_files' (model type) and 'language'
    """
    
    import os
    
    bert_model = params.get('path_files', 'xlm-roberta-base').split('/')[-1]
    language = params.get('language', 'multilingual')
    
    # Create save directory with absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    save_dir = os.path.join(parent_dir, 'XLM_RoBERTa', 'models_saved', f'{bert_model}_{language}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save model and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print(f"Model saved to {save_dir}")

def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a given dataloader
    
    Params:
    - model: the PyTorch model
    - dataloader: DataLoader to evaluate on
    - device: torch device (cpu or cuda)
    
    Returns:
    - avg_val_loss: average validation loss
    """
    
    model.eval()
    total_eval_loss = 0
    
    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():
            outputs = model(b_input_ids,
                          token_type_ids=None,
                          attention_mask=b_input_mask,
                          labels=b_labels)
        
        loss = outputs[0]
        total_eval_loss += loss.item()
    
    avg_val_loss = total_eval_loss / len(dataloader)
    
    return avg_val_loss

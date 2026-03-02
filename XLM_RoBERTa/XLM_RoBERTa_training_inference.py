# XLM-RoBERTa Training Script (based on BERT Classifier architecture)
import transformers 
import torch
import glob 
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import random
import os
import time
import sys
sys.path.insert(0, '..')

from xlm_codes.feature_generation import combine_features, return_dataloader
from xlm_codes.data_extractor import data_collector
from xlm_codes.utils import *
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# Try to import neptune
try:
    import neptune
    from api_config import project_name, proxies, api_token
    NEPTUNE_AVAILABLE = True
except:
    NEPTUNE_AVAILABLE = False
    print("⚠️ Neptune not available - running offline")

# Check GPU availability
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Initialize neptune
if NEPTUNE_AVAILABLE:
    try:
        neptune.init(project_name, api_token=api_token, proxies=proxies)
    except:
        print("⚠️ Skipping Neptune logging (running offline)")
        neptune = None
else:
    neptune = None

def select_model(model_name, weights=None):
    """
    Load and return XLM-RoBERTa model
    """
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    return model

def save_model(model, tokenizer, params):
    """
    Save fine-tuned model and tokenizer
    """
    bert_model = params.get('path_files', 'xlm-roberta-base').split('/')[-1]
    language = params.get('language', 'multilingual')
    
    # Create save directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'models_saved', f'{bert_model}_{language}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save model and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print(f"Model saved to {save_dir}")

def Eval_phase(params, which_files='test', model=None):
    """
    Evaluate XLM-RoBERTa model on validation/test data
    
    Params:
    - params: dictionary with training parameters
    - which_files: 'train', 'val', or 'test'
    - model: the model to evaluate
    
    Returns:
    - fscore: F1 score
    - accuracy: Accuracy score
    """
    
    # Load the files to test on
    if which_files == 'train':
        path = os.path.join(params['files'], 'train', params['csv_file'])
    elif which_files == 'val':
        path = os.path.join(params['files'], 'val', params['csv_file'])
    elif which_files == 'test':
        path = os.path.join(params['files'], 'test', params['csv_file'])
    else:
        raise ValueError("which_files must be 'train', 'val', or 'test'")
    
    test_files = glob.glob(path)
    
    if not test_files:
        print(f'Warning: No files found for {which_files}')
        return 0.0, 0.0
    
    print(f'Loading XLM-RoBERTa tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(params['path_files'])
    
    # Put the model in evaluation mode
    model.eval()
    
    # Load the dataset
    df_test = data_collector(test_files, params, False)
    
    if df_test.empty:
        print(f'Warning: No data found for {which_files}')
        return 0.0, 0.0
    
    sentences_test = df_test.text.values
    labels_test = df_test.label.values
    
    # Encode the dataset using the tokenizer
    input_test_ids, att_masks_test = combine_features(sentences_test, tokenizer, params['max_length'])
    test_dataloader = return_dataloader(input_test_ids, labels_test, att_masks_test, 
                                        batch_size=params['batch_size'], is_train=False)
    
    print(f"Running eval on {which_files}...")
    
    # Tracking variables 
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            outputs = model(b_input_ids, 
                          token_type_ids=None, 
                          attention_mask=b_input_mask)
            
            logits = outputs[0]
            predictions = torch.argmax(logits, dim=1)
            
            pred_labels.extend(predictions.cpu().numpy())
            true_labels.extend(b_labels.cpu().numpy())
    
    # Calculate metrics
    fscore = f1_score(true_labels, pred_labels, average='binary', zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)
    
    model.train()
    return fscore, accuracy

def train_model(params, best_val_fscore=0):
    """
    Main training function
    """
    
    # Load training data
    train_path = os.path.join(params['files'], 'train', params['csv_file'])
    train_files = glob.glob(train_path)
    df_train = data_collector(train_files, params, True)
    
    if df_train.empty:
        print("Error: No training data found")
        return 0.0, best_val_fscore
    
    print(f"Loaded {len(df_train)} training samples")
    
    # Load validation data
    val_path = os.path.join(params['files'], 'val', params['csv_file'])
    val_files = glob.glob(val_path)
    df_val = data_collector(val_files, params, False)
    
    print(f"Loaded {len(df_val)} validation samples")
    
    # Load tokenizer and model
    print(f'Loading XLM-RoBERTa tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(params['path_files'])
    model = select_model(params['path_files'], params['weights'])
    model.to(device)
    
    # Prepare data
    sentences_train = df_train.text.values
    labels_train = df_train.label.values.astype(int)
    
    sentences_val = df_val.text.values
    labels_val = df_val.label.values.astype(int)
    
    # Encode the dataset using the tokenizer (also filters labels to match valid sentences)
    print("Encoding training data...")
    input_train_ids, att_masks_train, labels_train = combine_features(sentences_train, tokenizer, params['max_length'], labels_train)
    
    print("Encoding validation data...")
    input_val_ids, att_masks_val, labels_val = combine_features(sentences_val, tokenizer, params['max_length'], labels_val)
    
    # Create dataloaders
    train_dataloader = return_dataloader(input_train_ids, labels_train, att_masks_train, 
                                         batch_size=params['batch_size'], is_train=True)
    validation_dataloader = return_dataloader(input_val_ids, labels_val, att_masks_val, 
                                              batch_size=params['batch_size'], is_train=False)
    
    # Initialize AdamW optimizer
    optimizer = AdamW(model.parameters(),
                      lr=params['learning_rate'],
                      eps=params['epsilon'])
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * params['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(total_steps/10),
                                                num_training_steps=total_steps)
    
    # Fix random seed
    fix_the_random(seed_val=params['random_seed'])
    
    # Store loss values for plotting
    loss_values = []
    
    # Create neptune experiment if available
    if neptune and params['logging'] == 'neptune':
        bert_model = params['path_files']
        language = params['language']
        name_one = bert_model + "_" + language
        neptune.create_experiment(name_one, params=params, send_hardware_metrics=False, 
                                  run_monitoring_thread=False)
        neptune.append_tag(bert_model)
        neptune.append_tag(language)
    
    # The best val fscore obtained till now
    best_val_fscore = best_val_fscore
    
    # Training loop
    for epoch_i in range(0, params['epochs']):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
        print('Training...')
        
        t0 = time.time()
        total_loss = 0
        model.train()
        
        # For each batch of training data...
        for step, batch in tqdm(enumerate(train_dataloader)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            model.zero_grad()
            
            # Get the model outputs
            outputs = model(b_input_ids,
                          token_type_ids=None,
                          attention_mask=b_input_mask,
                          labels=b_labels)
            
            loss = outputs[0]
            
            if neptune and params['logging'] == 'neptune':
                neptune.log_metric('batch_loss', loss)
            
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # Calculate average loss
        avg_train_loss = total_loss / len(train_dataloader)
        
        if neptune and params['logging'] == 'neptune':
            neptune.log_metric('avg_train_loss', avg_train_loss)
        
        print(f"Average training loss: {avg_train_loss:.4f}")
        loss_values.append(avg_train_loss)
        
        # Compute metrics on validation and test sets
        val_fscore, val_accuracy = Eval_phase(params, 'val', model)
        test_fscore, test_accuracy = Eval_phase(params, 'test', model)
        
        # Report metrics
        print(f"Validation - F1: {val_fscore:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"Test - F1: {test_fscore:.4f}, Accuracy: {test_accuracy:.4f}")
        
        if neptune and params['logging'] == 'neptune':
            neptune.log_metric('val_fscore', val_fscore)
            neptune.log_metric('val_acc', val_accuracy)
            neptune.log_metric('test_fscore', test_fscore)
            neptune.log_metric('test_accuracy', test_accuracy)
        
        # Save the model only if the validation fscore improves
        if val_fscore > best_val_fscore:
            print(f"New best validation F1: {val_fscore:.4f} (previous: {best_val_fscore:.4f})")
            best_val_fscore = val_fscore
            save_model(model, tokenizer, params)
    
    if neptune and params['logging'] == 'neptune':
        neptune.stop()
    
    del model
    torch.cuda.empty_cache()
    return val_fscore, best_val_fscore


# Explanation of all the params used below
# 'logging': where logging {'local','neptune'}
# 'language': language {'Arabic', 'English','German','Indonesian','Italian','Polish','Portuguese','Spanish','French'}
# 'is_train': whether train dataset 
# 'learning_rate': Adam parameter lr
# 'files': Path to the dataset folder (containing the train, val and test subfolders)
# 'csv_file': The regex used by glob to load the datasets. {'*_full.csv','*_translated.csv'} for untranslated and translated datasets respectively
# 'samp_strategy': The way in which we sample the training data points. {'stratified'}
# 'epsilon': Adam parameter epsilon
# 'path_files': model path from where the XLM-RoBERTa model should be loaded
# 'take_ratio': Whether the sample ratio is ratio of total points or absolute number of points needed
# 'sample_ratio': ratio or the number of the training data points to take
# 'how_train': how the model is trained possible option {'all','baseline','all_but_one'}
# 'epochs': number of epochs to train
# 'batch_size': batch size
# 'to_save': whether to save the model or not
# 'weights': weights for binary classifier
# 'max_length': maximum length for input tokenization
# 'random_seed': seed value for reproducibility

params = {
    'logging': 'local',
    'language': 'English',
    'is_train': True,
    'learning_rate': 2e-5,
    'files': '../Dataset',
    'csv_file': '*_full.csv',
    'samp_strategy': 'stratified',
    'epsilon': 1e-8,
    'path_files': 'xlm-roberta-base',
    'take_ratio': False,
    'sample_ratio': 16,
    'how_train': 'baseline',
    'epochs': 5,
    'batch_size': 16,
    'to_save': True,
    'weights': [1.0, 1.0],
    'max_length': 128,
    'random_seed': 42,
}

if __name__ == '__main__':
    print("=" * 60)
    print("XLM-RoBERTa Training Configuration")
    print("=" * 60)
    print(f"Language: {params['language']}")
    print(f"Training mode: {params['how_train']}")
    print(f"Epochs: {params['epochs']}")
    print(f"Batch size: {params['batch_size']}")
    print(f"Learning rate: {params['learning_rate']}")
    print(f"Model: {params['path_files']}")
    print("=" * 60 + "\n")
    
    best_val_fscore = 0
    _, best_val_fscore = train_model(params, best_val_fscore)
    
    print('============================')
    print(f'Model for Language {params["language"]} trained successfully!')
    print('============================' )

# XLM-RoBERTa Inference Script (based on BERT Classifier architecture)
import transformers 
import torch
import glob 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import sys
sys.path.insert(0, '..')

from xlm_codes.feature_generation import combine_features, return_dataloader
from xlm_codes.data_extractor import data_collector
from xlm_codes.utils import *
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

# Try to import neptune
try:
    import neptune
    from api_config import project_name, proxies, api_token
    NEPTUNE_AVAILABLE = True
except:
    NEPTUNE_AVAILABLE = False

# If gpu is available
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Initialize neptune for logging
if NEPTUNE_AVAILABLE:
    try:
        neptune.init(project_name, api_token=api_token, proxies=proxies)
    except:
        print("⚠️ Skipping Neptune logging (running offline)")
        neptune = None
else:
    neptune = None

def Eval_phase(params, which_files='test', model=None):
    """
    Evaluation function for XLM-RoBERTa
    
    Params:
    - params: dictionary with model configuration
    - which_files: 'train', 'val', or 'test'
    - model: the model to evaluate (optional, will load from params if None)
    """
    
    # Load the files to test on
    if which_files == 'train':
        path = os.path.join(params['files'], 'train', params['csv_file'])
        test_files = glob.glob(path)
    elif which_files == 'val':
        path = os.path.join(params['files'], 'val', params['csv_file'])
        test_files = glob.glob(path)
    elif which_files == 'test':
        path = os.path.join(params['files'], 'test', params['csv_file'])
        test_files = glob.glob(path)
    else:
        raise ValueError("which_files must be 'train', 'val', or 'test'")
    
    if not test_files:
        print(f'Warning: No files found for {which_files}')
        return 0.0, 0.0, "", None
    
    print('Loading XLM-RoBERTa tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(params['path_files'])
    
    # If model is passed, use it. Else load from saved location
    if model is None:
        print(f"Loading model from {params['path_files']}...")
        model = AutoModelForSequenceClassification.from_pretrained(params['path_files'], num_labels=2)
        model.to(device)
    
    # Put the model in evaluation mode
    model.eval()
    
    # Load the dataset
    df_test = data_collector(test_files, params, False)
    
    if df_test.empty:
        print(f'Warning: No data found for {which_files}')
        return 0.0, 0.0, "", None
    
    sentences_test = df_test.text.values
    labels_test = df_test.label.values.astype(int)
    
    # Encode the dataset using the tokenizer (also filters labels to match valid sentences)
    input_test_ids, att_masks_test, labels_test = combine_features(sentences_test, tokenizer, params['max_length'], labels_test)
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
    report = classification_report(true_labels, pred_labels, 
                                   target_names=['normal', 'hate'], zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels)
    
    model.train()
    
    return fscore, accuracy, report, cm

# Params dictionary for inference
params = {
    'logging': 'local',
    'language': 'English',
    'is_train': False,
    'files': '../Dataset',
    'csv_file': '*_full.csv',
    'path_files': './models_saved/xlm-roberta-base_English',
    'batch_size': 16,
    'max_length': 128,
}

if __name__ == '__main__':
    
    print("=" * 70)
    print("XLM-RoBERTa Inference")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists(params['path_files']):
        print(f"\n❌ Model not found at: {params['path_files']}")
        print("\nPlease train the model first using:")
        print("   python XLM_RoBERTa_training_inference.py")
        exit(1)
    
    # Run inference on test set
    print(f"\nEvaluating on test set...")
    print("-" * 70)
    
    fscore, accuracy, report, cm = Eval_phase(params, 'test', model=None)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"F1 Score: {fscore:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    print("=" * 70)

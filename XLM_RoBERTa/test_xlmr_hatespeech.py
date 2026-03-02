"""
XLM-RoBERTa Hate Speech Benchmark (DE-LIMIT paths)
Uses params['files'], params['language'] format
"""

import os
import pandas as pd
import torch
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# === DE-LIMIT CONFIG PARAMETERS ===
params = {
    'model_path': 'XLM_RoBERTa-HateSpeech',      # Path to saved model folder
    'files': '../Dataset',                       # Path to dataset folder (containing train, val, test)
    'language': 'English',                       # Language to evaluate on
    'batch_size': 16,                            # Batch size for inference
    'max_length': 128,                           # Max sequence length
}

# Override with command line arguments if provided
import sys
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            if key in params:
                if key == 'batch_size' or key == 'max_length':
                    params[key] = int(value)
                else:
                    params[key] = value

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("XLM-RoBERTa Hate Speech Evaluation")
print("=" * 70)
print(f"Model: {params['model_path']}")
print(f"Files: {params['files']}")
print(f"Language: {params['language']}")
print(f"Batch size: {params['batch_size']}")
print(f"Device: {DEVICE}")
print("=" * 70)

# === LOAD MODEL ===
print("\nLoading pre-trained hate speech model...")
tokenizer = AutoTokenizer.from_pretrained(params['model_path'])
model = AutoModelForSequenceClassification.from_pretrained(params['model_path'])
model.to(DEVICE)
model.eval()
print("Ready for inference!")

def predict_batch(texts, tokenizer, model, device, batch_size=32, max_length=128):
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
        
        predictions.extend(preds)
    return np.array(predictions)

# === DE-LIMIT PATHS & EVALUATION ===
phases = ['train','val', 'test']
results = {}
phase_times = {}

for phase in phases:
    phase_start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"PHASE: {phase.upper()}")
    print(f"{'='*70}")
    
    # DE-LIMIT exact path format
    phase_path = os.path.join(params['files'], phase, params['language'] + "*_full.csv")
    print(f"Looking for: {phase_path}")
    
    # Find matching files
    phase_dir = os.path.join(params['files'], phase)
    if not os.path.exists(phase_dir):
        print(f"Directory missing: {phase_dir}")
        continue
    
    csv_files = [f for f in os.listdir(phase_dir) if f.startswith(params['language']) and f.endswith("_full.csv")]
    
    if not csv_files:
        print(f"No {params['language']}*_full.csv files found")
        continue
    
    print(f"Found: {csv_files}")
    
    all_texts, all_labels = [], []
    
    # Load DE-LIMIT CSVs
    for csv_file in csv_files:
        filepath = os.path.join(phase_dir, csv_file)
        df = pd.read_csv(filepath)
        
        # DE-LIMIT text/label detection
        text_col = next((col for col in df.columns if 'text' in col.lower()), df.columns[0])
        label_col = next((col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                         and len(df[col].unique()) <= 10), None)
        
        texts = df[text_col].fillna('').astype(str).tolist()
        labels = df[label_col].fillna(0).astype(int).tolist() if label_col else [0]*len(texts)
        
        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"  {csv_file}: {len(texts)} samples")
    
    print(f"\nPredicting {len(all_texts)} samples...")
    predictions = predict_batch(all_texts, tokenizer, model, DEVICE, batch_size=params['batch_size'], 
                                max_length=params['max_length'])
    
    # DE-LIMIT metrics
    accuracy = accuracy_score(all_labels, predictions)
    f1_macro = f1_score(all_labels, predictions, average='macro', zero_division=0)
    
    phase_elapsed_time = time.time() - phase_start_time
    phase_times[phase] = phase_elapsed_time
    
    results[phase] = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'samples': len(all_texts),
        'label_dist': dict(pd.Series(all_labels).value_counts()),
        'time_seconds': phase_elapsed_time
    }
    
    print(f"\n{phase.upper()} RESULTS:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   F1-macro:  {f1_macro:.4f}")
    print(f"   Samples:   {len(all_texts)}")
    print(f"   Time:      {phase_elapsed_time:.2f} seconds")
    print(f"   Labels:    {results[phase]['label_dist']}")
    print(classification_report(all_labels, predictions, zero_division=0))

# === DE-LIMIT BENCHMARK TABLE ===
print("\n" + "="*90)
print("XLM-RoBERTa Hate Speech vs DE-LIMIT MODELS")
print("="*90)
summary_df = pd.DataFrame(results).T[['accuracy', 'f1_macro', 'samples', 'time_seconds']]
print(summary_df.round(4))

# Calculate and display total time
total_time = sum(phase_times.values())
print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# Save DE-LIMIT compatible CSV
results_df = pd.DataFrame({
    'phase': list(results.keys()),
    'model': 'XLM_RoBERTa-HateSpeech-NoTraining',
    'accuracy': [results[p]['accuracy'] for p in results],
    'f1_macro': [results[p]['f1_macro'] for p in results],
    'samples': [results[p]['samples'] for p in results],
    'time_seconds': [results[p]['time_seconds'] for p in results]
})
results_df.to_csv("DE-LIMIT_xlmr_results.csv", index=False)
print("\nSaved: DE-LIMIT_xlmr_results.csv")

print("\nBenchmark complete! Ready for comparison table.")

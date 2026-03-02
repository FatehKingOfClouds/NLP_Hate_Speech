import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Create target folder (same structure as mBERT)
target_dir = "XLM_RoBERTa-HateSpeech"
os.makedirs(target_dir, exist_ok=True)

print("Downloading XLM-RoBERTa Multilingual Hate Speech model... (~2.2GB)")

# Download & save tokenizer
print("1/2 Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("christinacdl/XLM_RoBERTa-Multilingual-Hate-Speech-Detection-New")
tokenizer.save_pretrained(target_dir)
print("Tokenizer saved!")

# Download & save model (fine-tuned classifier!)
print("2/2 Downloading model...")
model = AutoModelForSequenceClassification.from_pretrained("christinacdl/XLM_RoBERTa-Multilingual-Hate-Speech-Detection-New")
model.save_pretrained(target_dir)
print("Model saved! (XLM-RoBERTa-large + hate speech head)")

print(f"🎉 All files saved to: {os.path.abspath(target_dir)}")
print("\nFolder contents:")
for file in os.listdir(target_dir):
    size = os.path.getsize(os.path.join(target_dir, file)) / (1024*1024)  # MB
    print(f"  {file}: {size:.1f}MB")

print("\nUsage:")
print(f'tokenizer = AutoTokenizer.from_pretrained("{os.path.abspath(target_dir)}")')
print(f'model = AutoModelForSequenceClassification.from_pretrained("{os.path.abspath(target_dir)}")')

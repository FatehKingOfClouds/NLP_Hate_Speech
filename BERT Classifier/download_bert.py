import os
from transformers import AutoTokenizer, AutoModel

# Create target folder
target_dir = "BERT Classifier/multilingual_bert"
os.makedirs(target_dir, exist_ok=True)

print("Downloading multilingual BERT...")

# Download & save tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenizer.save_pretrained(target_dir)
print("Tokenizer saved!")

# Download & save model  
model = AutoModel.from_pretrained("bert-base-multilingual-cased")
model.save_pretrained(target_dir)
print("Model saved!")

print(f"All files saved to: {os.path.abspath(target_dir)}")
print("Files: config.json, pytorch_model.bin (~420MB), tokenizer files")

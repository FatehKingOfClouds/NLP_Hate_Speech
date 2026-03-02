# Dummy BERT_inference.py for CNN_GRU compatibility
# Comment out all Neptune/model loading - just return empty functions

print("⚠️ Skipping BERT inference (CNN_GRU mode)")

# Mock the functions that CNN_GRU expects
def predict_bert(texts, model_path="./BERT Classifier/multilingual_bert"):
    print("Skipping BERT predictions - using CNN-GRU only")
    return [0] * len(texts)  # Dummy predictions

def get_bert_embeddings(texts, model_path="./BERT Classifier/multilingual_bert"):
    print("Skipping BERT embeddings - using CNN-GRU only") 
    return []  # Dummy embeddings

# Mock any other functions that might be called
class DummyBERT:
    def __init__(self): pass
    def predict(self, texts): return predict_bert(texts)

# If script checks for neptune (you already disabled these)
neptune = None

print("BERT_inference compatibility layer ready")

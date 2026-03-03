# Multilingual Hate Speech Detection Benchmark

This repository implements a benchmark comparison of three deep learning approaches for multilingual hate speech detection across 9 languages. The implementation is based on the DE-LIMIT dataset and uses the foundational work by [Aluru et al. (2021)](https://arxiv.org/abs/2004.06465) as a skeleton for our implementation.

## Overview

This project compares three different neural architectures for hate speech classification:

1. **BERT Classifier** - Multilingual BERT (mBERT) fine-tuning approach
2. **CNN-GRU** - CNN-GRU model with MUSE word embeddings
3. **XLM-RoBERTa** - Cross-lingual RoBERTa for multilingual transfer learning

Each model can be trained on different language combinations and evaluated on the DE-LIMIT dataset.

## Directory Structure

```
DE-LIMIT-master/
├── Dataset/                 # Dataset folder with train/val/test splits
│   ├── full_data/          # Raw text files
│   ├── train/              # Training data (language-specific CSVs)
│   ├── val/                # Validation data (language-specific CSVs)
│   └── test/               # Test data (language-specific CSVs)
├── BERT Classifier/         # mBERT fine-tuning implementation
│   ├── bert_codes/         # Core modules for feature generation and utilities
│   ├── BERT_training_inference.py
│   ├── BERT_inference.py
│   └── README.md           # Detailed instructions for BERT models
├── CNN_GRU/                # CNN-GRU model implementation
│   ├── Models/             # Model architecture definitions
│   ├── Preprocess/         # Data preprocessing with Ekphrasis
│   ├── bert_codes/         # Feature generation and utilities
│   ├── CNN_GRU.py
│   └── README.md           # Detailed instructions for CNN-GRU model
├── XLM_RoBERTa/            # XLM-RoBERTa implementation
│   ├── xlm_codes/          # Core modules for feature generation and utilities
│   ├── XLM_RoBERTa_training_inference.py
│   ├── test_xlmr_hatespeech.py
│   ├── download_model.py
│   └── README.md           # Detailed instructions for XLM-RoBERTa model
├── LASER+LR/               # Logistic regression on LASER embeddings
├── Example/                # Example notebooks for model usage
└── README.md               # This file
```

## Requirements

Make sure to use **Python 3.11.9**. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset

Detailed information about the dataset, including data collection, preprocessing, language coverage (9 languages), and dataset statistics can be found in the `Dataset/README.md` file. 

**Key points:**
- 9 languages: Arabic, English, French, German, Indonesian, Italian, Polish, Portuguese, Spanish
- Multiple training modes: baseline (single language), all (all languages), zero-shot, all-but-one
- CSV format with columns: `text` and `label` (0 for normal, 1 for hate speech)

## Models

### 1. BERT Classifier

Uses multilingual BERT with different training strategies. Supports baseline, all-languages, and transfer learning approaches.

**Quick Start:**
```bash
cd BERT\ Classifier
python BERT_training_inference.py
```

For detailed instructions, see `BERT Classifier/README.md`

### 2. CNN-GRU

CNN-GRU architecture with MUSE word embeddings optimized for social media text with extensive preprocessing.

**Quick Start:**
```bash
cd CNN_GRU
python CNN_GRU.py
```

For detailed instructions, see `CNN_GRU/README.md`

### 3. XLM-RoBERTa

State-of-the-art cross-lingual RoBERTa model pre-trained on 100+ languages.

**Quick Start:**
```bash
cd XLM_RoBERTa
python download_model.py  # Download pre-trained model
python XLM_RoBERTa_training_inference.py  # Train model
python test_xlmr_hatespeech.py  # Evaluate model
```

For detailed instructions, see `XLM_RoBERTa/README.md`

## Testing and Evaluation

Each model directory contains its own evaluation script with customizable parameters:

- **BERT Classifier:** See `BERT Classifier/README.md` for evaluation instructions
- **CNN-GRU:** See `CNN_GRU/README.md` for evaluation instructions  
- **XLM-RoBERTa:** Run `python test_xlmr_hatespeech.py language=<lang> batch_size=<size>` with custom parameters

Results are typically saved as CSV files with metrics (accuracy, F1-score) for each evaluation phase (train/val/test).

## Citation

If you use this implementation, please cite the original DE-LIMIT paper:

```bibtex
@inproceedings{aluru2021deep,
  title={A Deep Dive into Multilingual Hate Speech Classification},
  author={Aluru, Sai Saketh and Mathew, Binny and Saha, Punyajoy and Mukherjee, Animesh},
  booktitle={Machine Learning and Knowledge Discovery in Databases. Applied Data Science and Demo Track: European Conference, ECML PKDD 2020, Ghent, Belgium, September 14--18, 2020, Proceedings, Part V},
  pages={423--439},
  year={2021},
  organization={Springer International Publishing}
}
```

## Notes

- Check individual model README files for model-specific configuration options
- GPU is recommended for faster training
- Each model supports different training modes: baseline, all, zero-shot, all-but-one
- Detailed hyperparameter information is available in each model's README.md


	
### References

The original DE-LIMIT implementation and baseline models:
- Original Paper: [Deep Learning Models for Multilingual Hate Speech Classification](https://arxiv.org/abs/2004.06465)
- Original Code: [CNERG DE-LIMIT Repository](https://github.com/hate-alert/DE-LIMIT)
- Original BERT Models: [HuggingFace Hub - Hate-speech-CNERG](https://huggingface.co/Hate-speech-CNERG)

Referenced implementations:
1. MUSE embeddings from [Facebook MUSE Repository](https://github.com/facebookresearch/MUSE)
2. BERT fine-tuning approach from [Chris McCormick's Blog](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
3. CNN-GRU original implementation from [CHASE Repository](https://github.com/ziqizhang/chase)
4. LASER embeddings from [Facebook LASER Repository](https://github.com/facebookresearch/LASER)

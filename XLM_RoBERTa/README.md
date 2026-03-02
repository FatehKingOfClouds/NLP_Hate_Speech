------------------------------------------
***Instructions for XLM-RoBERTa models***
------------------------------------------

1. **Download the Model**
    1. Run `python download_model.py` to download and save the XLM-RoBERTa model weights to the `XLM_RoBERTa-HateSpeech/` folder.

2. **Training (Optional, not fully tested and not recommanded)**
    1. Set the `language` you wish to train on in the `params` dictionary of `XLM_RoBERTa_training_inference.py`.
    2. Load the datasets using the parameters `files` to specify the dataset directory, and `csv_file` set as `*_full.csv` for untranslated data.
    3. Set the `how_train` parameter to:
        - `baseline`: Train on single target language only
        - `all`: Train on all languages combined
        - `zero_shot`: Train on all languages except target language
        - `all_but_one`: Train on all languages except one specified
    4. Set parameters like `sample_ratio`, `batch_size`, `learning_rate`, and `epochs` depending on your experimental setup.
    5. Call `python XLM_RoBERTa_training_inference.py` to train the model. The best model is automatically saved.

3. **Evaluation**
    1. Run `python test_xlmr_hatespeech.py` to evaluate the model on test data.
    2. You can pass parameters via command-line arguments: `python test_xlmr_hatespeech.py language=Arabic batch_size=32`
    3. Available parameters: `model_path`, `files`, `language`, `batch_size`, `max_length`
    4. Results are saved to `DE-LIMIT_xlmr_results.csv` with timing information for each phase.
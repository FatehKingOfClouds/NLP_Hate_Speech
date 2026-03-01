# Dataset Description
The datasets used are as follows:
1. Arabic:
	1b. [Ousidhoum et al.](https://github.com/HKUST-KnowComp/MLMA_hate_speech)
2. English:
	2e. [Ousidhoum et al.](https://github.com/HKUST-KnowComp/MLMA_hate_speech)
3. French:
	9a. [Ousidhoum et al.](https://github.com/HKUST-KnowComp/MLMA_hate_speech)

In cases where the actual text is not given by the source and only tweet ids and labels are given, use any twitter scraping tools to extract the texts.
In the above datasets, some of them contain multiple labels for the text such as hate-speech, abusive, offensive, etc. In such cases, only the text with either hate-speech and normal labels are used and others are discarded. 


## Instructions for getting the datasets
1. Download the datasets from the above sources and place it in the subfolder `Dataset/full_data`
2. Rename the files for each language respectively `Arabic_1b_full.csv`, `English_2e_full.csv`, `French_9a_full.csv`
3. Use the `Translation.ipynb ` to translate the datasets into english
4. Use the ids given in `ID Mapping` folder for splitting the datasets into train, val and test splits. Use the file `Stratified Split.ipynb` for doing the splits. 
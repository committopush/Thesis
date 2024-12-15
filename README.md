# EXAMINING POLITICAL POLARIZATION IN THE GERMAN BUNDESTAG USING LARGE LANGUAGE MODELS: HISTORICAL TRENDS AND A CONTEMPORARY ANALYSIS

### Overview

This repository contains the code and resources for our thesis, which focuses on analyzing polarization in the speeches of the German Bundestag. We aim to detect, evaluate and analyse polarizing language through sentiment analysis, structural analysis, and fine-tuned transformer models.

#### Contributors
- [Anna Amenda](mailto:52267@novasbe.pt)
- [Silja Bienert](mailto:58250@novasbe.pt)
- [Antonius Greiner](mailto:58544@novasbe.pt)
- [Benedikt Gros](mailto:60508@novasbe.pt)
- [Antonie Thomas](mailto:59117@novasbe.pt)
---
### Key Components

1. **Data Processing Pipelines**  
   The repository includes robust pipelines for ingesting, cleaning, and preprocessing all speeches provided by the german bundestag, leveraging the open discourse project.

2. **Analyses**
   - Analysis of electoral terms 17-20, as well as electoral term 20 in-depth.
   - Sentiment analysis leveraging dictionaries and a bag-of-words approach.
   - Structural analysis to identify patterns and trends in speech interruptions, reactions, and debate types.

4. **Fine-Tuned Transformer Models**  
   - **BERT, GPT 4o Mini and LLAMA 3.1 8B Instruct**: LLMs fine-tuned to detect polarizing language in Bundestag speeches based on our manually labelled corpus.

5. **Model Comparison**  
   Provides a comparative analysis of the different transformer models implemented, highlighting their varying effectiveness in identifying polarization individually and using different ensemble approaches.

---

### Repository Structure

#### 1. **`analyses/`**
   - Contains Jupyter Notebooks for key analyses:
     - **RQ1_Developments.ipynb**: Analysis of electoral terms 17-20.
     - **RQ2_Electoral_Term_20.ipynb**: Analysis specific to the 20th electoral term.
     - **Sentiment_Analysis.ipynb**: Detailed Analysis of speech sentiment.
     - **Structural_Analysis.ipynb**: Analysis of structural aspects of different kinds of bundestag debates, interruptions, reactions captured by the stenographs.
   
   
#### 2. **`data/`**
   - Contains scripts for scraping raw data from the bundestag API, scripts for data extraction, preprocessing, cleaning and exploratory data analysis
     - **`import/`**: (?) @Bene
     - **`open_discourse/`**: API calling and XML parsers for data ingestion, preprocessing and cleaning pipelines for data transformation. Leans heavily on the open discourse project.
     - **`postprocessing_speeches/`**: (?) @Bene
     - **`preprocessing/`**: Scripts for manual and automated data labeling.
     - **05-staedte.xlsx**: Dataset taken from [Statistisches Bundesamt (2022)](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/Administrativ/05-staedte.xlsx%3F__blob%3DpublicationFile&ved=2ahUKEwil9bGMtvyJAxXjdqQEHWLjDCEQFnoECBgQAQ&usg=AOvVaw3ILEGrnP35OHGg-QmAaTmb) to map birthplaces of politicians for data enrichment.
     - **Data_Cleaning.ipynb** Data cleaning and exploratory data analysis.
     - **Master_Dataframe.ipynb** Data integration for further analysis.

#### 3. **`model_comparison/`**
   - Evaluation notebooks and data comparing fine-tuned transformer models:
     - **BERTval.xlsx, GPTval.xlsx, LLAMAval.csv**: Validation set output of the fine-tuned transformer models.
     - **Model_Comparison.ipynb**: Comprehensive analysis and comparison of model results.

#### 4. **`sentiment/`**
   - Resources for sentiment analysis:
     - **sent_dictionary_1.csv**: Sentiment dictionary as taken from [Haselmayer et al. (2017)](http://link.springer.com/article/10.1007%2Fs11135-016-0412-4)
     - **sent_dictionary_2.csv**: Sentiment dictionary as taken from [Rauh (2018)](https://www.tandfonline.com/doi/full/10.1080/19331681.2018.1485608)
     - **sentiment_score_calculation.ipynb**: Implementation of sentiment score calculations.

#### 5. **`topic_modelling/`**
   - Includes resources for topic modeling:
     - **topic_list.csv**: Topics and associated words with weights identified from speeches.
     - **topic_modelling.ipynb**: Implementation of topic modeling using LDA.

#### 6. **`transformer_models/`**
   - Contains implementation and fine-tuning of transformer models:
     - **`BERT/`**
     - **`GPT/`**
     - **`LLAMA/`**
     - **Tokenization.ipynb**: Notebook to segment longer speeches for model ingestion.

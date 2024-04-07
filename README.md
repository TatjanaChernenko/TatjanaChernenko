*Unraveling mysteries hidden within datasets, a relentless data detective, transforming chaos into knowledge.*

# Introduction

- 👋 Hi, I’m @TatjanaChernenko
- 👀 I’m interested in Data Science, ML/DL, NLP and .
- 📫 How to reach me: tatjana.chernenko.work@gmail.com
- 📁 New Public Repository: This new public GitHub profile contains both old (starting from approx. 2015) and new my projects, uploaded now after years of working in a private capacity due to privacy policies of my employers. 
- 📁 Project Uploads: All projects uploaded here are from my personal endeavors and university research. Due to privacy policies at SAP SE, where I am employed, I am unable to share work-related projects publicly. These repositories exclusively feature my private projects and are newly uploaded to this fresh GitHub profile. Thank you for your understanding.

## Table of Contents

## My Projects
- ## Research Repositories
1. **CHERTOY: Word Sense Induction for Web Search Result Clustering**
   - GitHub: [CHERTOY System](https://github.com/TatjanaChernenko/word_sense_induction_CHERTOY_system)

2. **Data-to-text Generation**
   - GitHub: [Data-to-text Generation](https://github.com/TatjanaChernenko/image_description_generation)

3. **Text Summarization with LexRank**
   - GitHub: [Text Summarization with LexRank](https://github.com/TatjanaChernenko/text_summarization_LexRank_modified_ecnu)

### Predictive Maintenance (RUL, failure prediction, maintaince)

**LSTM for predictive maintenance of aircraft machines**
   - GitHub: [LSTM for predictive maintenance of aircraft machines: failure and RUL (remaining usefull life) prediction](https://github.com/TatjanaChernenko/Predictive-Maintenance-with-LSTM)

**Anomaly Detection for Time Series with IBM API (SVR), K-Means clustering, statsmodels decomposition and Fourier analysis** 
   - GitHub: [IBM API for anomaly detection, univariate data](https://github.com/TatjanaChernenko/ibm_anomaly_detection_time_series_univariate)
     

### Game AI

**Reinforcement Learning Agent for Bomberman**
   - GitHub: [RL Agent for Bomberman](https://github.com/TatjanaChernenko/reinforcement_learning_agent_Bomberman_game)

### Speech Recognition

**Speech-to-text with Transfer Learning**
   - GitHub: [Speech-to-text with Transfer Learning](https://github.com/TatjanaChernenko/automatic_speech_translation_transfer_learning)

### Data Augmentation

**Data Augmentation Techniques for Classification**
   - GitHub: [Data Augmentation Techniques](https://github.com/TatjanaChernenko/data_augmentation)

**[My Playground (smaller projects / samples)]**(#playground):

- [EDA (Explorative Data Analysis)](#eda-explorative-data-analysis)
- [Basic NLP Examples](#basic-nlp-examples)
- [Text Categorisation Task with ML](#text-categorisation-task-with-ml)
- [Dialogue Systems](#dialogue-systems)
- [Recommendation Systems](#recommendation-systems)
- [Sentiment Analysis](#sentiment-analysis)
- [Voice technologies (speech-to-text, speech-to-speech, text-to-speech)](#voice-technologies-speech-to-text-speech-to-speech-text-to-speech)
- [Various ML tasks](#various-ml-tasks)
- [Apps with ChatGPT and OpenAI](#apps-with-chatgpt-and-openai)
- [Databases, SQL, noSQL, webscrapping, email notifications](#databases-sql-nosql-webscrapping-email-notifications)
- [NMT](#nmt)
  
## Inspiration
- [Different](#different)
  - [Prediction, Time Series, Anomaly Detection](#prediction-time-series-anomaly-detection)
  - [Data Science Resources](#data-science-resources)
  - [NLP Resources](#nlp-resources)
  - [Evaluation Tasks](#evaluation-tasks)
  - [Image / Video Technologies](#image--video-technologies)
  - [Voice Technologies](#voice-technologies)
  - [Different ML Resources](#different-ml-resources)
  
## Industrial research
- [OpenAI](#openai)
- [Microsoft](#microsoft)
- [Meta Research](#meta-research)



# My Projects

## Research Repositories
### NLP / ML
- 2017/2018 [CHERTOY: Word Sense Induction for better web search result clustering](https://github.com/TatjanaChernenko/word_sense_induction_CHERTOY_system/tree/main) - An approach to improve **word sense induction systems (WSI)** for web search result clustering. Exploring the boundaries of vector space models for the WSI Task. CHERTOY system. Authors: Tatjana Chernenko, Utaemon Toyota.

Whitepaper - [link](https://github.com/TatjanaChernenko/word_sense_induction_CHERTOY_system/blob/main/Results.pdf)
  
*Key words: word sense induction, web search results clustering, ML, NLP, word2vec, sent2vec, NLP, data science, data processing.*

- 2018 [Data-to-text: Natural Language Generation from structured inputs](https://github.com/TatjanaChernenko/image_description_generation) - This project investigates the **generation of descriptions of images** focusing on spatial relationships between the objects and sufficient attributes for the objects. Leveraging an encoder-decoder architecture with LSTM cells (the Dong et al. (2017) is taken as basis), the system transforms normalized vector representations of attributes into fixed-length vectors. These vectors serve as initial states for a decoder generating target sentences from sequences in description sentences.

Whitepaper - [link](https://github.com/TatjanaChernenko/image_description_generation/blob/main/docs/Paper_NL_generation.pdf)

*Key words: natural language generation, encoder-decoder, ML, NLP, data science, feed-forward neural network, LSTMs.*

- 2018 [Text Summarization research: Optimizing LexRank system with ECNU features](https://github.com/TatjanaChernenko/text_summarization_LexRank_modified_ecnu) -  enhancing the LexRank-based **text summarization** system by incorporating semantic similarity measures from the ECNU system. The LexRank-based text summarization system employs a stochastic graph-based method to compute the relative importance of textual units for extractive multi-document text summarization. This implementation initially utilizes cosine similarity between sentences as a key metric. In this model, a connectivity matrix based on intra-sentence cosine similarity is used as the adjacency matrix of the graph representation of sentences. The objective is to explore the impact of replacing cosine similarity with a combination of features from the ECNU system, known for its semantic similarity measure. This modification aims to improve the summarization effectiveness of the LexRank approach.

Whitepaper - [link](https://github.com/TatjanaChernenko/text_summarization_LexRank_modified_ecnu/blob/main)

*Key words: natural language processing, text summarizaton, ML, NLP, data science, LexRank, ECNU, semantic similarity metrics, multi-document text summarization, cosine similarity, connectivity matrix, optimization.*

- 2019, [Reinforcement Learning agent for Bomberman game](https://github.com/TatjanaChernenko/reinforcement_learning_agent_Bomberman_game) Training a **RL agent** for the multi-player game Bomberman using reinforcement learning, deep Q-learning with a dueling network architecture and separate decision and target networks, prioritized experience replay.

Whitepaper - [link](https://github.com/TatjanaChernenko/reinforcement_learning_agent_Bomberman_game/blob/main/Report.pdf)

*Key words: reinforcement learning, q-learning.*

- 2018, [Speech-to-text: Transfer Learning for Automatic Speech Translation (playground)](https://github.com/TatjanaChernenko/automatic_speech_translation_transfer_learning) - Playground for the **Automated Speech Translation (AST)** with transfer learning vs. AST trained from scratch; hyperparameters tuning and evaluation.

Report - [link](https://github.com/TatjanaChernenko/automatic_speech_translation_transfer_learning/blob/main/AST_Transfer_Learning_report.pdf)

*Key words: transfer learning, automated speech translation*

- 2018, [Data Augmentation techniques for binary- and multi-label classification](https://github.com/TatjanaChernenko/data_augmentation) - Exploring **Data Augmentation techniques** (Thesaurus and Backtranslation, a winning Kaggle technique) to expand existing datasets, evaluating on binary- and multi-label classification task (spam/not spam and news articles classification). Important when training data is limited, especially in Machine Learning (ML) or Deep Learning (DL) applications. The primary concept involves altering text while retaining its meaning to enhance the dataset's diversity.

*Key words: data augmentation, data science, ML, DL, binary and multi-class classification*

## LSTM for predictive maintenance of aircraft machines 
- [LSTM for predictive maintenance of aircraft machines: failure and RUL (remaining usefull life) prediction](https://github.com/TatjanaChernenko/Predictive-Maintenance-with-LSTM) - Predictive Maintenance: Use LSTM to predict failure (binary classification) and RUL (remaining useful life or time to failure with regression) of aircraft engines.

     
## Anomaly Detection for Time Series with IBM API (SVR), K-Means clustering, statsmodels decomposition and Fourier analysis
- [IBM API for anomaly detection, univariate data](https://github.com/TatjanaChernenko/ibm_anomaly_detection_time_series_univariate) - Jupyter Notebook


## Text Categorisation Task with ML (Reuters)
- [Categorization task with ML Algorithms for Reuters text categorization benchmark dataset](https://github.com/TatjanaChernenko/text_categorisation_ML) - LinearSVC (Linear Support Vector Classifier), Decision Tree, Random Forest, Logistic Regression,k-Nearest Neighbors (k-NN),Naive Bayes, AdaBoost, LDA (Linear Discriminant Analysis),RBM (Restricted Boltzmann Machine),MLP (Multilayer Perceptron).


- Collection of **chatbots, dialogue systems**

(*coming soon*)

## Playground


## EDA (Explorative Data Analysis)
- [Explorative Data Analysis of Aibnb rental prices in New York, 2019](https://github.com/TatjanaChernenko/ml_playground/blob/main/Explorative%20Data%20Analysis%20-%20AirBnb%20Prices%20in%20New%20York.ipynb) - Jupyter Notebook

(*further projects coming soon*)


## Basic NLP Examples
- [NLP examples](https://github.com/TatjanaChernenko/basic_NLP_examples) - Jupyter Notebook with data preprocessing, top words, word cloud, frequencies, AgglomerativeClustering, PCA, Sentiment analysis, Topic Detection
- [REGEX examples](https://github.com/TatjanaChernenko/regex_examples) - simple summary of regex examples in Jupyter Notebook.


## Databases, SQL, noSQL, webscrapping, email notifications 
- [LinkedIn webscrapping, saving data to local MongoDB and csv, filtering and updating the user via email](https://github.com/TatjanaChernenko/mongodb_webscrapping) - automatically extracting job postings from LinkedIn according to the predefined settings, storing them in a local MongoDB database and csv file, searching for relevant positions based on the keywords, and notifying the user via email (Gmail API) about relevant job opportunities.

## Various ML tasks

- https://github.com/TatjanaChernenko/ml_playground
- [Regression Task: Predicting Airbnb rental prices in New York](https://github.com/TatjanaChernenko/ml_playground/blob/main/Explorative%20Data%20Analysis%20-%20AirBnb%20Prices%20in%20New%20York.ipynb) - **Regression task** to predict rental prices in New York, playground. Models used: Linear Regression, Decision Trees, NNs.

(*coming soon*)

## Apps with ChatGPT and OpenAI
- [OpenAI basic app](https://github.com/TatjanaChernenko/open_ai_basic_app) - updating the basic **OpenAI simple app** to generate pet names to correspond to the OpenAI changes in code (January, 2024)
- [fork: GPT Chatbot - customizable]https://github.com/TatjanaChernenko/customizable-gpt-chatbot) - A dynamic, scalable AI chatbot built with Django REST framework, supporting custom training from PDFs, documents, websites, and YouTube videos. Leveraging OpenAI's GPT-3.5, Pinecone, FAISS, and Celery for seamless integration and performance.
  
(*coming soon*)


## Dialogue Systems
- [Question answering with DistilBERT](https://huggingface.co/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amazônica+or+Amazônia%3B+Spanish%3A+Selva+Amazónica%2C+Amazonía+or+usually+Amazonia%3B+French%3A+Forêt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species) - Question answering with DistilBERT, HuggingFace
- [Document Question Answering with LayoutLM](https://huggingface.co/impira/layoutlm-document-qa) - This is a fine-tuned version of the multi-modal LayoutLM model for the task of question answering on documents. It has been fine-tuned using both the SQuAD2.0 and DocVQA datasets.


## Recommendation Systems
Own projects:

(*to be uploaded soon*)
Forks:
- [Recommendation System with TensorFlow, approx.2020](https://github.com/TatjanaChernenko/recommenders) - TensorFlow Recommenders is a library for building recommender system models using TensorFlow. Fork from smellslikeml
- [TF-Recommenders with Kubernetes](https://github.com/TatjanaChernenko/tf-recommenders-kubernetes) - Example of kubernetes deployment for tf-recommenders model


## Sentiment Analysis

(*to be uploaded soon*)

Forks:
- [Tweet Analysis](https://github.com/TatjanaChernenko/ChatGPT-Tweet-Analysis) - Analyzing ChatGPT-related tweets to observe technology interest trends over time
- 

## Voice technologies (speech-to-text, speech-to-speech, text-to-speech)

Own projects:
(*to be uploaded soon*)

Forks:
- [Whisper OpenAI](https://github.com/openai/whisper) - Robust Speech Recognition via Large-Scale Weak Supervision
- [WhisperX Timestamps (& Diarization)](https://github.com/m-bain/whisperX) - Automatic Speech Recognition with Word-level Timestamps (& Diarization)
- [Whisper real-time](https://github.com/TatjanaChernenko/whisper-stream) - real-time speech-to-text conversion with Whisper
- [SpeechGPT](https://github.com/TatjanaChernenko/SpeechGPT) - detects microphone input and coverts it to text using Google's Speech Recognition API. It then opens ChatGPT and inputs the recognized text using selenium.
It can be used with a wake word, and it can also use text to speech to repeat ChatGPT's answer to the query. 
- [Speaker Diarization Whisper](https://github.com/MahmoudAshraf97/whisper-diarization) - Whisper with with Speaker Diarization based on OpenAI Whisper
- [Speech-to-Text-WaveNet](https://github.com/TatjanaChernenko/speech-to-text-wavenet): End-to-end sentence level English speech recognition based on DeepMind's WaveNet and tensorflow (forked from buriburisuri)
- [Speech-to-text via Whisper and GPT-4](https://github.com/TatjanaChernenko/speech_to_text_with_whisper_to_GPT) - transcribe dictations to text using whisper, and then fixing the resulting transcriptions into usable text using gpt-4 (forked from MNoichl)
- [TensorFlow Speech Recognition](https://github.com/TatjanaChernenko/AUDIO-PREOCESSING-AND-SPEECH-CLASSIFICATION) - audio processing and speech classification with Tensorflow - convolution neural networks (forked from harshel)
- [Watson_STT_CustomModel](https://github.com/TatjanaChernenko/Watson_STT_CustomModel) - a custom speech model using IBM Watson Speech to Text; an old one (approx. 2018)
- [Simple Speech Recognition with Python](https://github.com/TatjanaChernenko/Speech-Recognition-Speech-To-Text) - very simple setup using SpeechRecognition Python module
- [CTTS](https://github.com/TatjanaChernenko/ctts) - Controllable Text-to-speech system, based on Microsoft's FastSpeech2
- [Google Sheets to Speech](https://github.com/TatjanaChernenko/sentences_to_speech) - Excel-to-speech, forked from Renoncio: A Python script for generating audio from a list of sentences in Google Sheets.
- [StreamlitTTS](https://github.com/TatjanaChernenko/StreamlitTTS) - Streamlit app allows you to convert text to audio files using the Microsoft Edge's online text-to-speech service. 
- [Dolla Llama: Real-Time Co-Pilot for Closing the Deal](https://github.com/TatjanaChernenko/dolla_llama) - forked from smellslikeml; power a real-time speech-to-text agent with retrieval augmented generation based on webscraped customer use-cases, implements speech-to-text (STT) and retrieval-augmented generation (RAG) to assist live sales calls.
- [Text-to-Speech on AWS](https://github.com/TatjanaChernenko/awstexttospeech) - forked from codets1989; using AWS Lambda and Polly converting text to speech and creating a automated pipeline
- [Whisper speech-to-text Telegram bot](https://github.com/TatjanaChernenko/Whisper_Speech-To-Text_Telegram_Bot) - forked from loyal-pelmen; Speech-to-Text Telegram bot
- [DeepSpeech on devices](https://github.com/TatjanaChernenko/DeepSpeech_for_devices) - embedded (offline, on-device) speech-to-text engine which can run in real time ranging from a Raspberry Pi 4 to high power GPU servers
- [Bash Whisper - using a Digital Voice Recorder (DVR)](https://github.com/TatjanaChernenko/bash-whisper-transcription) - Bash function to ease the transcription of audio files with OpenAI's whisper.
- [Awesome Whisper](https://github.com/sindresorhus/awesome-whisper) - model variants and playgrounds
- [TikTok Analyzer](https://github.com/TatjanaChernenko/tiktok-analyzer) - Video Scraping and Content Analysis Tool. Search & download Tiktok videos by username and/or video tag, and analyze video contents. Transcribe video speech to text and perform NLP analysis tasks (e.g., keyword and topic discovery; emotion/sentiment analysis). Isolate audio signal and perform signal processing analysis tasks (e.g., pitch, prosody and sentiment analysis). Isolate visual stream and perform image tasks (e.g., object detection; face detection).
- [SpeechBrain](https://github.com/TatjanaChernenko/speechbrain) - an open-source PyTorch toolkit that accelerates Conversational AI development; spans speech recognition, speaker recognition, speech enhancement, speech separation, language modeling, dialogue, and beyond. Over 200 competitive training recipes on more than 40 datasets supporting 20 speech and text processing tasks. Supports both training from scratch and fine-tuning pretrained models such as Whisper, Wav2Vec2, WavLM, Hubert, GPT2, Llama2, and beyond. The models on HuggingFace can be easily plugged in and fine-tuned.
- [Speech Synthesis Markup - SSML](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup) - XML-based markup language that you can use to fine-tune your text to speech output attributes (tutorial from Microsoft).
 
(*further projects coming soon*)

## NMT

(*coming soon*)


## Computer Vision

Own projects:
(*to be uploaded soon*)

Forks/Inspiration:
- [Laundry Sorting with a robotic arm](https://github.com/Jdka1/Laundry-Sorting) - Sorting laundry autonomously with a robotic arm and Computer Vision
- []() -
- []() -

## Other

- [AutoXlicker](https://github.com/TatjanaChernenko/AutoClicker) - A lightweight and customizable autoclicker tool for automating repetitive clicking tasks.
- [Hand Gesture Computer Interface]((https://github.com/TatjanaChernenko/Hand-Gesture-Computer-Interface) - Software that allows you to interact with your computer through hand gestures. 

------

## My Projects

| Category            | Project Title                                                                                                                     | GitHub                                                                                                                                   |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| Research Repositories | CHERTOY: Word Sense Induction for better web search result clustering                                                            | [CHERTOY System](https://github.com/TatjanaChernenko/word_sense_induction_CHERTOY_system/tree/main)                                     |
| Research Repositories | Data-to-text: Natural Language Generation from structured inputs                                                                  | [Data-to-text Generation](https://github.com/TatjanaChernenko/image_description_generation)                                             |
| Research Repositories | Text Summarization research: Optimizing LexRank system with ECNU features                                                         | [Text Summarization with LexRank](https://github.com/TatjanaChernenko/text_summarization_LexRank_modified_ecnu)                           |
| Research Repositories | Reinforcement Learning agent for Bomberman game                                                                                   | [RL Agent for Bomberman](https://github.com/TatjanaChernenko/reinforcement_learning_agent_Bomberman_game)                                |
| Research Repositories | Speech-to-text: Transfer Learning for Automatic Speech Translation (playground)                                                   | [Speech-to-text with Transfer Learning](https://github.com/TatjanaChernenko/automatic_speech_translation_transfer_learning)              |
| Research Repositories | Data Augmentation techniques for binary- and multi-label classification                                                          | [Data Augmentation Techniques](https://github.com/TatjanaChernenko/data_augmentation)                                                    |
| Predictive Maintenance | LSTM for predictive maintenance of aircraft machines: failure and RUL (remaining useful life) prediction                        | [Predictive Maintenance with LSTM](https://github.com/TatjanaChernenko/Predictive-Maintenance-with-LSTM)                                |
| Anomaly Detection   | Anomaly Detection for Time Series with IBM API (SVR), K-Means clustering, statsmodels decomposition and Fourier analysis        | [IBM API for anomaly detection, univariate data](https://github.com/TatjanaChernenko/ibm_anomaly_detection_time_series_univariate)       |
| Text Categorisation | Text Categorisation Task with ML (Reuters)                                                                                       | [Categorization task with ML Algorithms for Reuters text categorization benchmark dataset](https://github.com/TatjanaChernenko/text_categorisation_ML) |
| Playground          | Explorative Data Analysis of Airbnb rental prices in New York, 2019                                                              | [EDA of Airbnb Prices in New York](https://github.com/TatjanaChernenko/ml_playground/blob/main/Explorative%20Data%20Analysis%20-%20AirBnb%20Prices%20in%20New%20York.ipynb)                          |
| Playground          | Basic NLP Examples                                                                                                                | [NLP examples](https://github.com/TatjanaChernenko/basic_NLP_examples)                                                                   |
| Databases, SQL, noSQL, webscrapping, email notifications | LinkedIn webscrapping, saving data to local MongoDB and csv, filtering and updating the user via email               | [LinkedIn Webscrapping and Email Notifications](https://github.com/TatjanaChernenko/mongodb_webscrapping)                              |
| Various ML tasks   | Regression Task: Predicting Airbnb rental prices in New York                                                                     | [Regression Task with Airbnb Data](https://github.com/TatjanaChernenko/ml_playground/blob/main/Explorative%20Data%20Analysis%20-%20AirBnb%20Prices%20in%20New%20York.ipynb)                      |
| Dialogue Systems   | Question answering with DistilBERT                                                                                               | [DistilBERT Question Answering](https://huggingface.co/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amazônica+or+Amazônia%3B+Spanish%3A+Selva+Amazónica%2C+Amazonía+or+usually+Amazonia%3B+French%3A+Forêt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species) |
| Dialogue Systems   | Document Question Answering with LayoutLM                                                                                        | [LayoutLM Document QA](https://huggingface.co/impira/layoutlm-document-qa)                                                               |
| Recommendation Systems | Recommendation System with TensorFlow                                                                                           | [TensorFlow Recommenders](https://github.com/TatjanaChernenko/recommenders)                                                             |
| Sentiment Analysis | Sentiment Analysis                                                                                                               | (*to be uploaded soon*)                                                                                                                |
| Voice Technologies | Speech-to-Text-WaveNet                                                                                                          | [Speech-to-Text-WaveNet](https://github.com/TatjanaChernenko/speech-to-text-wavenet)                                                   |
| Voice Technologies | Speech-to-text via Whisper and GPT-4                                                                                             | [Speech-to-text with Whisper to GPT](https://github.com/TatjanaChernenko/speech_to_text_with_whisper_to_GPT)                            |
| Voice Technologies | TensorFlow Speech Recognition                                                                                                    | [TensorFlow Speech Recognition](https://github.com/TatjanaChernenko/AUDIO-PREOCESSING-AND-SPEECH-CLASSIFICATION)                        |
| Voice Technologies | Watson_STT_CustomModel                                                                                                           | [Watson STT Custom Model](https://github.com/TatjanaChernenko/Watson_STT_CustomModel)                                                   |
| Voice Technologies | Simple Speech Recognition with Python                                                                                            | [Simple Speech Recognition](https://github.com/TatjanaChernenko/Speech-Recognition-Speech-To-Text)                                      |
| Voice Technologies | CTTS                                                                                                                              | [CTTS](https://github.com/TatjanaChernenko/ctts)                                                                                       |
| Voice Technologies | Google Sheets to Speech                                                                                                           | [Google Sheets to Speech](https://github.com/TatjanaChernenko/sentences_to_speech)                                                      |
| Voice Technologies | StreamlitTTS                                                                                                                      | [StreamlitTTS](https://github.com/TatjanaChernenko/StreamlitTTS)                                                                         |
| Voice Technologies | Dolla Llama: Real-Time Co-Pilot for Closing the Deal                                                                             | [Dolla Llama](https://github.com/TatjanaChernenko/dolla_llama)                                                                          |
| Voice Technologies | Text-to-Speech on AWS                                                                                                             | [Text-to-Speech on AWS](https://github.com/TatjanaChernenko/awstexttospeech)                                                            |
| Voice Technologies | Whisper speech-to-text Telegram bot                                                                                              | [Whisper Speech-to-Text Telegram Bot](https://github.com/TatjanaChernenko/Whisper_Speech-To-Text_Telegram_Bot)                          |
| NMT                 | NMT (Neural Machine Translation)                                                                                                 | (*coming soon*)                                                                                                                         |

------

# Inspiration

## Different

### Prediction, Time Series, Anomaly Detection
- [AI for Time Series](https://github.com/DAMO-DI-ML/AI-for-Time-Series-Papers-Tutorials-Surveys) - papers, tutorials, surveys (!)
- [IBM Hub API Tutorial - Anomaly Detection](https://developer.ibm.com/learningpaths/get-started-anomaly-detection-api/industry-use-case/) - use IBM API for anomaly detection
- [IBM API for Anomaly Detection](https://github.com/TatjanaChernenko/IBM_anomaly-detection-code-pattern) - playing around with the Anomaly Detection service to be made available on IBM API Hub
- [AWS Forecast - end-to-end guide](https://github.com/aws-samples/amazon-forecast-samples) - Prediction with AWS
- [Amazon Monitron Guidance for Predictive Maintenance](https://aws.amazon.com/ru/solutions/guidance/predictive-maintenance-with-amazon-monitron/) - predictive maintenance management in industrial environments using Amazon Monitron and other AWS services.
- [Azure Predictive Maintenance Template](https://gallery.azure.ai/Collection/Predictive-Maintenance-Template-3) - Regression: predict the Remaining Useful Life (RUL), or Time to Failure (TTF); binary classification: predict if an asset will fail within certain time frame (e.g. days). Multi-class classification: Predict if an asset will fail in different time windows: E.g., fails in window [1, w0] days; fails in the window [w0+1,w1] days; not fail within w1 days.
- [AI for Time Series](https://github.com/DAMO-DI-ML/AI-for-Time-Series-Papers-Tutorials-Surveys) - Tutorials
- [PredictionIO](https://github.com/apache/predictionio) - Apache; a machine learning server for developers and ML engineers.
- [Conforal Prediction Tutorials](https://github.com/valeman/awesome-conformal-prediction) - A professionally curated list of awesome Conformal Prediction videos, tutorials, books, papers, PhD and MSc theses, articles and open-source libraries.
- [Time Series Prediction](https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction) - LSTM Neural Network for Time Series Prediction
- [Stock Prediction Models](https://github.com/huseinzol05/Stock-Prediction-Models) - gathers machine learning and deep learning models for Stock forecasting including trading bots and simulations
- [Lime](https://github.com/marcotcr/lime): Explaining the predictions of any machine learning classifier
- [Time Series Prediction](https://github.com/tgjeon/TensorFlow-Tutorials-for-Time-Series) - TensorFlow Tutorial for Time Series Prediction
- [Awesome-time-series](https://github.com/cure-lab/Awesome-time-series) - A comprehensive survey on the time series domains
- [Predictive Maintenance Datasets](https://github.com/TatjanaChernenko/predictive-maintenance-datasets) - Datasets for predictive maintenance


### Data Science Resources

- [Data Science Resources - learning](https://github.com/datasciencemasters/go) - The open-source curriculum for learning to be a Data Scientist (quite basic, but nice links to books, etc.)
- [Data Science Resources](https://github.com/academic/awesome-datascience) - An Data Science repository to learn and apply for real world problems.
- [Data Science Cheatsheets](https://github.com/FavioVazquez/ds-cheatsheets) - List of Data Science Cheatsheets to rule the world
- [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) - full text in Jupyter Notebooks
- [Data science Python notebooks](https://github.com/donnemartin/data-science-ipython-notebooks) - Deep learning (TensorFlow, Theano, Caffe, Keras), scikit-learn, Kaggle, big data (Spark, Hadoop MapReduce, HDFS), matplotlib, pandas, NumPy, SciPy, Python essentials, AWS, and various command lines.
- [Datasets from Huggingface](https://github.com/huggingface/datasets) - collection of Huggingface datasets
- [Huggingface - web API for visualizing and exploring of datasets](https://github.com/huggingface/datasets-server) - Lightweight web API for visualizing and exploring all types of datasets - computer vision, speech, text, and tabular - stored on the Hugging Face Hub
- [Huggingface - analyse datasets](https://github.com/TatjanaChernenko/data-measurements-tool) - EDA from Huggingface (Developing tools to automatically analyze datasets)
- [Jupyter Notebooks for Big Data - with Spark and Hadoop](https://medium.com/@techlatest.net/using-jupyter-notebook-with-big-data-a-guide-on-how-to-use-jupyter-notebook-with-big-data-f473af87185b) - A guide on how to use Jupyter Notebook with big data frameworks like Apache Spark and Hadoop, including recommended libraries and tools.
  
### NLP Resources

- [NLP state-of-the-art](https://github.com/sebastianruder/NLP-progress) - Tracking Progress in Natural Language Processing
- [NMT Tutorial](https://github.com/ymoslem/OpenNMT-Tutorial) - Neural Machine Translation (NMT) tutorial. Data preprocessing, model training, evaluation, and deployment.
- [NMT](https://github.com/Prompsit/mutnmt) - An educational tool to train, inspect, evaluate and translate using neural engines
- [FasterNMT](https://github.com/iC-RnD/FasterNMT) - NMT incl. data preprocessing, model training, evaluation, and deployment with great performance.
- [DeepLearningForNLPInPytorch](https://github.com/rguthrie3/DeepLearningForNLPInPytorch) - an IPython Notebook tutorial on deep learning for natural language processing, including structure prediction.
- [alennlp](https://github.com/allenai/allennlp) - An open-source NLP research library, built on PyTorch.
- [Natural Language Processing Tutorial for Deep Learning Researchers](https://github.com/graykode/nlp-tutorial)
- [Oxford Deep NLP 2017 course](https://github.com/oxford-cs-deepnlp-2017/lectures)
- [awasome-nlp](https://github.com/keon/awesome-nlp) - A curated list of resources dedicated to Natural Language Processing (NLP)
- [German-NLP Datasets](https://github.com/TatjanaChernenko/German-NLP)
- [Scrapy](https://github.com/smellslikeml/scrapy) - a fast high-level web crawling & scraping framework for Python.
- [Ressources for redacting personally identifiable information](https://github.com/smellslikeml/awesome-data-redaction) - resources for programmatically redacting personally identifiable information
- [Simple ML baselines - Jupyter Notebooks](https://github.com/TatjanaChernenko/simple_ML_baselines) - simple ML baselines
- [Huggingface - Transformer](https://github.com/TatjanaChernenko/transformers) - Huggingface - Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX; various tasks
- [Name Entity Recognition with Electra](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english) - Name Entity Recognition with Electra, Huggingface
- [Text Generation with GPT-2](https://huggingface.co/gpt2?text=A+long+time+ago%2C+) - Text Generation with GPT-2, Huggingface
- [Natural Language Inference with RoBERTa](https://huggingface.co/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal) - Natural Language Inference with RoBERTa, Huggingface
- [Summarization with BART](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct) - Text Summarization with BART, Huggingface
- [Data processing pipelines ](https://github.com/huggingface/datatrove) - data processing pipelines from Huggingface
- [Tokenizers from Huggingface](https://github.com/huggingface/tokenizers) - Fast State-of-the-Art Tokenizers optimized for Research and Production
- [text-generation-inference from Huggingface](https://github.com/huggingface/text-generation-inference#text-generation-inference) - Large Language Model Text Generation Inference. Text Generation Inference (TGI) is a toolkit for deploying and serving Large Language Models (LLMs). TGI enables high-performance text generation for the most popular open-source LLMs, including Llama, Falcon, StarCoder, BLOOM, GPT-NeoX, and more.
- [Open-source AI cookbook](https://github.com/huggingface/cookbook/tree/main) - Huggingface - Open-source AI cookbook(Fine_tuning_Code_LLM_on_single_GPU.ipynb, etc.)
- [Styleformer](https://github.com/PrithivirajDamodaran/Styleformer)  - A neural language style transfer framework to transfer text smoothly between styles.
- [conv-emotion](https://github.com/declare-lab/conv-emotion) - Implementation of different architectures for emotion recognition in conversations.

### Evaluation Tasks
- [Evaluate from Huggingface](https://github.com/huggingface/evaluate) - Evaluate is a library that makes evaluating and comparing models and reporting their performance easier and more standardized. Implementations of dozens of popular metrics: the existing metrics cover a variety of tasks spanning from NLP to Computer Vision
- [NMT Evaluation framework](https://github.com/Optum/nmt) - A useful framework to evaluate and compare different Machine Translation engines between each other on variety datasets.
- [FastChat - LLM chatbots evaluation platform](https://github.com/huggingface/FastChat) - FastChat is an open platform for training, serving, and evaluating large language model based chatbots.
- [ParlAI](https://github.com/facebookresearch/ParlAI) - a framework for training and evaluating AI models on a variety of openly available dialogue datasets.
- [AutoGluon](https://github.com/autogluon/autogluon) - if you prefer more control over the forecasting model exploration, training, and evaluation processes.
- [tune from Huggingface](https://github.com/huggingface/tune)  - A benchmark for comparing Transformer-based models.
 
### Image / Video Technologies
- [Activity detection](https://github.com/TatjanaChernenko/ActionAI) - Real-Time Spatio-Temporally Localized Activity Detection by Tracking Body Keypoints
- [Dance transfer](https://github.com/TatjanaChernenko/everybody_dance_faster) - acquire pose estimates from a participant, train a pix2pix model, transfer source dance video, and generate a dance gif; Motion transfer booth for a 1 hour everybody dance now video generation using EdgeTPU and Tensorflow 2.0
- [Video embeddings and similarity](https://github.com/smellslikeml/image-similarity-metric-learning) - Training CNN model to generate image embeddings
- [Deep Fakes Detection](https://github.com/smellslikeml/Deepfake-detection) - (2019) Repository to detect deepfakes, an opensource project as part of AI Geeks effort.
- [Diffusers from Huggingface](https://github.com/huggingface/diffusers) - Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch

### Voice Technologies
- [Speech Cognitive Service](https://github.com/LaloCo/SpeechCognitiveService_Translate) - A Jupyter Notebook that details how to use Azure's Speech Cognitive Service to Translate speech
- [Audio-Speech Tutorial, 2022](https://github.com/TatjanaChernenko/Audio-Speech-Tutorial) - an introduction on the topic of audio and speech processing - from basics to applications (approx. 2022)
- [espnet](https://github.com/espnet/espnet) - End-to-End Speech Processing Toolkit
- [TTS](https://github.com/coqui-ai/TTS) - a deep learning toolkit for Text-to-Speech, battle-tested in research and production
- [Speech-to-text benchmark](https://github.com/TatjanaChernenko/speech-to-text-benchmark) - speech-to-text benchmarking framework
- [Speech-to-text](https://techtldr.com/transcribing-speech-to-text-with-python-and-openai-api-whisper/) - with Whisper and Python, March 2023
- [Multilingual Text-to-Speech](https://github.com/TatjanaChernenko/Multilingual_Text_to_Speech) - Tomáš Nekvinda and Ondřej Dušek, One Model, Many Languages: Meta-Learning for Multilingual Text-to-Speech, 2020, Proc. Interspeech 2020
- [Unified Speech Tokenizer for Speech Language Models](https://github.com/TatjanaChernenko/SpeechTokenizer) - SpeechTokenizer; SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models, Xin Zhang and Dong Zhang and Shimin Li and Yaqian Zhou and Xipeng Qiu, 2023
- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - a Fundamental End-to-End Speech Recognition Toolkit and Open Source SOTA Pretrained Models; hopes to build a bridge between academic research and industrial applications on speech recognition. By supporting the training & finetuning of the industrial-grade speech recognition model, researchers and developers can conduct research and production of speech recognition models more conveniently, and promote the development of speech recognition ecology
- [Whisper model](https://github.com/openai/whisper) - OpenAI Whisper
- [Wenet](https://github.com/wenet-e2e/wenet) - Production First and Production Ready End-to-End Speech Recognition Toolkit
- [Distilled variant of Whisper ](https://github.com/TatjanaChernenko/distil-whisper) - Distilled variant of Whisper for speech recognition. 6x faster, 50% smaller, within 1% word error rate.
- [Fine-tune Whisper](https://huggingface.co/blog/fine-tune-whisper) -Fine-Tune Whisper For Multilingual ASR with Transformers

### Different ML Resources
- [Applied ML](https://github.com/eugeneyan/applied-ml) - (not really up-to-date, but good) Papers & tech blogs by companies sharing their work on data science & machine learning in production.
- [500 AI projects](https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code) - 500 AI Machine learning, Deep learning, Computer vision, NLP Projects with code
- [Parameter-Efficient Fine-Tuning from Huggingface](https://github.com/huggingface/peft) -  PEFT: State-of-the-art Parameter-Efficient Fine-Tuning.
- [Huggingface notebooks for various(!) tasks](https://github.com/huggingface/notebooks/tree/main) - Notebooks using the Hugging Face libraries
- [Huggingface educational resources](https://github.com/huggingface/education-toolkit) - Educational materials
- [Huggingface: notifications](https://github.com/huggingface/knockknock) - Knock Knock: Get notified when your training ends with only two additional lines of code
- [Huggingface: No-code raining and deployments of state-of-the-art machine learning models](https://github.com/TatjanaChernenko/autotrain-advanced) - AutoTrain is a no-code tool for training state-of-the-art models for Natural Language Processing (NLP) tasks, for Computer Vision (CV) tasks, and for Speech tasks and even for Tabular tasks.
- [Huggingface: Transformer Tutorials](https://github.com/nielsrogge/transformers-tutorials) - transformers-tutorials (by @nielsrogge) - Tutorials for applying multiple models on real-world datasets.

# Industrial research

## [OpenAI](https://github.com/openai)
- [OpenAI - simple app](https://github.com/openai/openai-quickstart-python) - My note: a model used and several functions are already deprecated; my version above has things updated.
- [Retrieval-Augmented Generation in Azure using Azure AI search](https://github.com/Azure-Samples/azure-search-openai-demo) - A sample app for the Retrieval-Augmented Generation pattern running in Azure, using Azure AI Search for retrieval and Azure OpenAI large language models to power ChatGPT-style and Q&A experiences.
- [A collection of custom OpenAI WebApps](https://github.com/MaxineXiong/OpenAI-API-Web-Apps)
- [Real time speech2text](https://github.com/saharmor/whisper-playground) - Build real time speech2text web apps using OpenAI's Whisper
- [OpenAI cookbook](https://github.com/openai/openai-cookbook)
- [OpenAI WhatsApp Chatbot](https://github.com/simonsanvil/openai-whatsapp-chatbot)
- [GPT-engineer](https://github.com/gpt-engineer-org/gpt-engineer) - Specify what you want it to build, the AI asks for clarification, and then builds it.
- [Prompt-engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
- [PDF search app with OpenAI](https://github.com/alejandro-ao/langchain-ask-pdf) - an AI-app that allows you to upload a PDF and ask questions about it. It uses OpenAI's LLMs to generate a response.
- [OpenAI Code Automation](https://github.com/nanowell/OpenAI-GPT-Code-Automation) - Fully coded Apps by GPT-4 and ChatGPT. Power of AI coding automation and new way of developing.
- [Semantic Search](https://github.com/nomic-ai/semantic-search-app-template) - Tutorial and template for a semantic search app powered by the Atlas Embedding Database, Langchain, OpenAI and FastAPI

## [Microsoft](https://github.com/microsoft)
- [OptiGuide](https://github.com/microsoft/OptiGuide) - Large Language Models for Supply Chain Optimization
- [Generative AI lessons](https://github.com/microsoft/generative-ai-for-beginners) - 12 Lessons, Get Started Building with Generative AI
- [LLMOps Workshop](https://github.com/microsoft/llmops-workshop) - Learn how to build solutions with Large Language Models.
- [Data Science Lessons](https://github.com/microsoft/Data-Science-For-Beginners)
- [AI Lessons](https://github.com/microsoft/AI-For-Beginners)
- [unilm](https://github.com/microsoft/unilm) - Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities. An open source AutoML toolkit for automate machine learning lifecycle, including feature engineering, neural architecture search, model compression and hyper-parameter tuning.
- [Old Photo Restoration via Deep Latent Space Translation](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life) - Bringing Old Photo Back to Life (CVPR 2020 oral)
- [NNI](https://github.com/microsoft/nni) - An open source AutoML toolkit for automate machine learning lifecycle, including feature engineering, neural architecture search, model compression and hyper-parameter tuning.
  
## [Meta Research](https://github.com/facebookresearch)

- [From Audio to Photoreal Embodiment](https://github.com/facebookresearch/audio2photoreal): Synthesizing Humans in Conversations
- [Seamless](https://github.com/facebookresearch/seamless_communication): Speech-to-speech translation (S2ST), Speech-to-text translation (S2TT), Text-to-speech translation (T2ST), Text-to-text translation (T2TT), Automatic speech recognition (ASR)
- [Fairseq(-py)](https://github.com/facebookresearch/fairseq) is a sequence modeling toolkit that allows researchers and developers to train custom models for translation, summarization, language modeling and other text generation tasks.
- [Faiss](https://github.com/facebookresearch/faiss) is a library for efficient similarity search and clustering of dense vectors.
- [PyTorch-BigGraph](https://github.com/facebookresearch/PyTorch-BigGraph) (PBG) is a distributed system for learning graph embeddings for large graphs, particularly big web interaction graphs with up to billions of entities and trillions of edges.
- [Llama 2 Fine-tuning](https://github.com/facebookresearch/llama-recipes) - examples to quickly get started with fine-tuning for domain adaptation and how to run inference for the fine-tuned models. For ease of use, the examples use Hugging Face converted versions of the models.
- [Pearl](https://github.com/facebookresearch/Pearl) - A Production-ready Reinforcement Learning AI Agent Library
- [TorchRecipes](https://github.com/facebookresearch/recipes) - Recipes are a standard, well supported set of blueprints for machine learning engineers to rapidly train models using the latest research techniques without significant engineering overhead.
- [fastText](https://github.com/facebookresearch/fastText) is a library for efficient learning of word representations and sentence classification.
- [ParlAI](https://github.com/facebookresearch/ParlAI) - a framework for training and evaluating AI models on a variety of openly available dialogue datasets.

## [AWS samples](https://github.com/orgs/aws-samples/repositories?q=&type=all&language=python&sort=)

- [Image Generator with Stable Diffusion on Amazon Bedrock using Streamlit](https://github.com/aws-samples/image-generator-with-stable-diffusion-on-amazon-bedrock-using-streamlit) - A quick demostration to deploy a Stable Diffusion Web application with containers running on Amazon ECS. The model is provided by Amazon Bedrock in this example
- [Transactional Data Lake using Apache Iceberg with AWS Glue Streaming and MSK Connect (Debezium)](https://github.com/aws-samples/transactional-datalake-using-amazon-msk-serverless-and-apache-iceberg-on-aws-glue) - Stream CDC into an Amazon S3 data lake in Apache Iceberg format with AWS Glue Streaming using Amazon MSK Serverless and MSK Connect (Debezium)
- [MLOps using Amazon SageMaker and GitHub Actions](https://github.com/aws-samples/mlops-sagemaker-github-actions) - MLOps example using Amazon SageMaker Pipeline and GitHub Actions
- [Near-Real Time Usage Anomaly Detection using OpenSearch](https://github.com/aws-samples/near-realtime-aws-usage-anomaly-detection) - Detect AWS usage anomalies in near-real time using OpenSearch Anomaly Detection and CloudTrail for improved cost management and security
- [Amazon DocumentDB (with MongoDB compatibility) samples](https://github.com/aws-samples/amazon-documentdb-samples) - Code samples that demonstrate how to use Amazon DocumentDB
- [Marketing Content Generator](https://github.com/aws-samples/generative-ai-marketing-portal) - CDK Deployment for a sample marketing portal using generative AI for content generation and distribution; Marketing Content Generation and Distribution powered by Generative AI
- [Amazon SageMaker and AWS Trainium Examples](https://github.com/aws-samples/sagemaker-trainium-examples) - Text classification using Transformers, Pretrain BERT using Wiki Data, Pretrain/Fine tune Llama using Wiki Data.
- [AWS SageMaker Local Mode](https://github.com/aws-samples/amazon-sagemaker-local-mode) - Amazon SageMaker Local Mode Examples
- [End-to-end AIoT w/ SageMaker and Greengrass 2.0 on NVIDIA Jetson Nano](https://github.com/aws-samples/aiot-e2e-sagemaker-greengrass-v2-nvidia-jetson) - Hands-on lab from ML model training to model compilation to edge device model deployment on the AWS Cloud. It covers the detailed method of compiling SageMaker Neo for the target device, including cloud instance and edge device, and how to write and deploy Greengrass-v2 components from scratch.
- [InsuranceLake ETL with CDK Pipeline](https://github.com/aws-samples/aws-insurancelake-etl) - This solution helps you deploy ETL processes and data storage resources to create an Insurance Lake using Amazon S3 buckets for storage, AWS Glue for data transformation, and AWS CDK Pipelines. It is originally based on the AWS blog Deploy data lake ETL jobs using CDK Pipelines, and complements the InsuranceLake Infrastructure project.
- [Amazon Forecast](https://aws.amazon.com/forecast/) - for a low-code/no-code fully managed time series AI/ML forecasting service.
- [AutoGluon](https://github.com/autogluon/autogluon) - if you prefer more control over the forecasting model exploration, training, and evaluation processes.
- [Retrieval Augmented Generation with Streaming LLM](https://github.com/aws-samples/smartsearch-ai-knowledge-workshop) - leverage LLMs for RAG(Retrieval Augmented Generation).
- [Build generative AI agents with Amazon Bedrock, Amazon DynamoDB, Amazon Kendra, Amazon Lex, and LangChain](https://github.com/aws-samples/generative-ai-amazon-bedrock-langchain-agent-example)
  

## [NVIDIA](https://github.com/NVIDIA)

- [Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples) - State-of-the-Art Deep Learning scripts organized by models - easy to train and deploy with reproducible accuracy and performance on enterprise-grade infrastructure.
- [NeMo](https://github.com/NVIDIA/NeMo): a toolkit for conversational AI



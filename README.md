*Applied AI Scientist and AI Architect working across enterprise speech AI, multilingual AI, LLM systems, agentic AI, knowledge graphs, retrieval-based AI, evaluation systems, and AI architecture.*

# Introduction

- 👋 Hi, I'm Tatjana Chernenko
- 📫 Contact: tatjana.chernenko.work@gmail.com
- Applied AI Scientist and AI Architect with work spanning enterprise speech AI, multilingual AI, applied NLP, LLM systems, agentic AI, RAG, knowledge graphs, evaluation and benchmarking systems, terminology intelligence, and AI-ready data architecture.
- My public GitHub contains a selective set of personal, academic, and research-oriented technical artifacts. Most recent enterprise work is not public due to confidentiality and employer constraints.

## Focus Areas

- Speech AI (ASR, Speech Translation, TTS), multilingual AI, and language technologies
- Evaluation, benchmarking, and reliability-oriented AI quality systems
- Agentic AI
- Applied NLP, terminology intelligence, and specialised-vocabulary handling
- LLM systems, GenAI
- Retrieval-augmented generation (RAG)
- Knowledge graphs, knowledge-enhanced AI, and workflow automation
- Enterprise AI architecture, data foundations, and governance-aware AI execution

## Selected Publications and Patents

- **LREC 2026:** [A Dataset for Evaluating ASR on Specialized Vocabulary](https://lrec.elra.info/lrec2026-main-032)
Emily Haubert Klering, Eduardo Gabriel Cortes, Tatjana Chernenko, Mariana Vargas Trarbach, Gabriel de Oliveira Ramos, Sandro José Rigo, Maitê Dupont, Ana Luiza Treichel Vianna, Gabriela Krause dos Santos, Vinicius Meirelles Pereira, Denis Andrei de Araujo, Rafael Kunst
  SAP–UNISINOS research collaboration. Focused on evaluation methodology and benchmark design for specialised-vocabulary robustness in enterprise ASR. Evaluating the ability of Automatic Speech Recognition (ASR) models to transcribe specialized vocabulary remains a persistent challenge, as standard datasets predominantly feature common words and thus obscure weaknesses on rare or out-of-vocabulary (OOV) terms. To address this limitation, we introduce a linguistically curated bilingual dataset (English and Portuguese) comprising 13,846 utterances (18.7 hours) distributed across synthetic and literature-derived subsets, with OOV rates reaching up to 100%. We further propose a diagnostic evaluation framework that partitions recognition performance into Biased Word Error Rate (B-WER), targeting domain-specific jargon, and Unbiased Word Error Rate (U-WER), focusing on general vocabulary. Baseline evaluations using Whisper models (medium, large-v3, and large-v3-turbo) confirm the necessity of this framework. On the most challenging datasets, B-WER reaches 0.88–0.90, whereas U-WER remains as low as 0.06–0.19, demonstrating that conventional WER masks critical failure modes in jargon recognition. Additionally, an oracle upper bound experiment shows that providing correct jargon via prompting reduces B-WER by 0.50–0.70 absolute, quantifying the considerable potential for contextual biasing. We release the datasets and evaluation scripts as a reproducible benchmark to foster research on domain-aware contextual biasing and OOV handling in ASR systems.

- **US Patent:** *[Semantic Domain Assignment Referencing Governance Domains and Term Databases](https://patentsgazette.uspto.gov/week01/OG/html/1542-1/US12518105-20260106.html)*
  T. Chernenko, B. Schork, M. DANEI — US Patent 12,518,105 (2026)

- **US Patent Application:** *Adaptive Fidelity Pipeline for Minimizing Hallucinations and Skipped Content in Speech-to-Text Systems*
  US Patent App. 250089US01 (2025) — [link pending]

- **US Patent:** *[System and Method Performing Terminology Disambiguation](https://patents.google.com/patent/US20240176778A1/en)*
  T. Chernenko, B. Schork, M. DANEI — US Patent 12,386,820 (2025)

- **US Patent:** *[Detection of Abbreviation and Mapping to Full Original Term](https://patents.justia.com/patent/12067370)*
  T. Chernenko, A. Snitko, J. Scharnbacher, M. Vasiltschenko — US Patent 12,067,370 (2024)

## Articles

Practitioner writing on AI evaluation, benchmarking, and enterprise speech AI — published on Hugging Face.

- **[Representativeness Before Metrics: Rethinking AI Evaluation for Deployment](https://github.com/TatjanaChernenko/TatjanaChernenko.github.io/blob/main/representativeness_before_metrics.md) [HuggingFace link pending]**
  What enterprise speech AI evaluation reveals about benchmark reliability — and why the lessons reach further than speech. Argues that weak benchmark representativeness, not metric design, is the primary bottleneck between benchmark success and deployment confidence.

- **[When Benchmarks Saturate: Ecological Validity in AI Evaluation] [HuggingFace link pending]**
  Why discriminative power, behavioral realism, and decision relevance matter more as systems improve. Examines how saturation and weak ecological validity compound each other — and what evaluation surfaces need to do to remain informative.
  
## Selected Earlier Research Repositories (~2017)

### Research / Academic Work

1.  **CHERTOY: Word Sense Induction for Web Search Result Clustering**
   Academic NLP research project at the Institute for Computational Linguistics, Heidelberg University, based on the SemEval-2013 WSI task. Built an unsupervised word sense induction pipeline for clustering ambiguous web-search snippets into semantically coherent subtopic groups using sense2vec word representations, vector-mixture bag-of-words snippet embeddings, and MeanShift clustering; evaluated 40 controlled experimental variants across preprocessing, embedding models, compositional representations, and clustering algorithms, improving pairwise clustering quality over baseline.
   GitHub: [CHERTOY System](https://github.com/TatjanaChernenko/word_sense_induction_CHERTOY_system)

2. **Natural Language Generation from Structured Inputs for Image Description Generation**
   Academic research project at the Institute for Computational Linguistics, Heidelberg University, on structured-to-text generation for image description. Built an encoder-decoder architecture with a feed-forward encoder over normalized attribute vectors and an LSTM decoder for sequence generation, using MS COCO, V-COCO, and COCO-a to model objects, actions, semantic roles, spatial relations, and descriptive attributes under automatic and human evaluation.
   GitHub: [Data-to-text Generation](https://github.com/TatjanaChernenko/image_description_generation)

3. **LexRank-based Text Summarization with Semantic Similarity Enhancements**
   Research project on extractive summarization extending LexRank with semantic-similarity features to improve sentence ranking and summary quality in longer documents.
   GitHub: [Text Summarization with LexRank](https://github.com/TatjanaChernenko/text_summarization_LexRank_modified_ecnu)

4.  **Dialogue Management: Improving Task-Oriented Conversational Agents with Deep Learning** Thesis in Computational Linguistics at Heidelberg University,Institute for Computational Linguistics supervised by Prof. Dr. Riezler and Dr. Vivi Nastase. Conversational AI-powered interfaces have become a top priority for many companies. Automated solutions save money, improve the quality of services and support the workers in their regular tasks. Technical software support is one of the industry fields that can benefit from usage of intelligent agents. AI systems with very specific knowledge in a narrow domain can automate regular tasks and support the workers by resolving technical issues. This thesis presents a development of such an AI-powered interface SmartSupport that combines strengths of the retrieval and neural models. SmartSupport can be used as an independent system or be integrated into a conversational agent as an
underlying Question Answering System, improving its performance in a semantically restricted conversational space of technical software issues.

## Archived Additional Technical Repositories

Selected older repositories in areas such as predictive maintenance, anomaly detection, reinforcement learning, speech adaptation, and data augmentation remain available in the profile history as secondary technical artifacts.

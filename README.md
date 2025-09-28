# BioLexis: A Hybrid Intelligence Engine for Genomic Novelty Detection

A high-performance bioinformatics pipeline by *Team Geek Velocity that combines deep learning, supervised learning and Unspervised learning's anomaly detection to classify environmental DNA (eDNA) and uncover novel biodiversity.*

![GitHub Banner](https://user-images.githubusercontent.com/109479893/206894274-a62a962a-b8a7-4927-968a-63795d2c8846.png)


## 📖 Introduction

In fields like environmental monitoring, epidemiology, and metagenomics, quickly identifying organisms from DNA sequences is critical. While existing tools can identify known organisms, they often fail to characterize novel or divergent sequences, which may represent hidden biodiversity or emerging pathogens. 

BioLexis addresses this challenge by synthesizing evidence from two distinct machine learning arms to provide a confident, context-aware status for every sequence. It moves beyond simple classification to actively discover and flag novel biological entities.

---
## 🧬 Workflow & Demo

Our pipeline employs a two-pronged approach. The supervised arm provides taxonomic predictions based on a trusted reference database, while the unsupervised arm builds a "map of known biodiversity" to identify sequences that are statistical outliers or divergent members of known groups. A final Decision Engine synthesizes this evidence to deliver a definitive classification.

### 📹 Demo Video
Watch a full walkthrough of the BioLexis pipeline and its interactive report:

---

##📂 Folder Structure:


BioLexis/
├── configs/                  # Configuration files (YAML)
│   └── default.yaml
│
├── data/
│   ├── raw/                  # Raw input files (CSV/FASTA)
│      ├── labels.csv        # Reference data for training
│      └── new_sequences.fasta # Scientist's experimental sequences
│   
│
├── results/                  # All pipeline outputs
│   ├── per_sequence_results.csv
│   ├── clusters_with_novelty.csv
│   ├── abundance_by_assignment.csv
│   └── report.html
│
├── src/                      # Source code
│   ├── pipeline.py           # Main pipeline orchestrator
│   ├── preprocess.py         # Cleaning and deduplication
│   ├── kmers.py              # K-mer feature generation
│   ├── embed.py              # Autoencoder embeddings
│   ├── cluster.py            # UMAP + HDBSCAN clustering
│   ├── label_transfer.py     # Per-rank classifier training/prediction
│   ├── abundance.py          # Abundance calculation
│   ├── diversity.py          # Diversity metrics (Shannon, Simpson)
│   ├── visualize.py          # Visualization utilities
│   └── evaluate.py           # Performance evaluation
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation


---
## 🚀 Features

* Hybrid ML Approach: Combines Logistic Regression classifiers with HDBSCAN anomaly detection for robust analysis.
* Novelty Detection: Explicitly identifies two types of novelty:
    * Emergent Taxa: Divergent members within known groups (e.g., new strains or species).
    * High-Confidence Novelty: Sequences with no close relatives in the reference map.
* Interactive Reporting: Generates a single, self-contained HTML report with interactive Plotly visualizations for intuitive data exploration.
* Efficient & Scalable: Employs a "train once, predict many" model, where the computationally expensive reference map is built once and reused for rapid analysis of many input samples.
* Multi-k-mer Featurization: Uses a range of k-mer sizes to create a rich, multi-resolution embedding space for improved model accuracy.

---

## 🛠 Tech Stack

* Backend: Python
* ML/Data Science: Scikit-learn, PyTorch, Pandas, NumPy
* Dimensionality Reduction & Clustering: UMAP-learn, HDBSCAN
* Visualization: Plotly, Matplotlib,Seaborn
* Bioinformatics: BioPython
* Reporting: HTML
* Cloud/GPU: Google Colab (T4/L4/A100), AWS

---
## ⚙ Installation (for Users)

To run the BioLexis pipeline on your own data, follow these steps:

1.  Clone the repository:

 
    bash
    git clone [https://github.com/KushalJain07/BioLexis.git](https://github.com/KushalJain07/BioLexis.git)
    cd BioLexis
    

3.  Create a virtual environment (recommended):
    bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
    

4.  Install the required dependencies:
    bash
    pip install -r requirements.txt
    

5.  Run the pipeline:
    bash
    
    python src/pipeline.py  --input path/to/your.fasta 
    

---
## 👨‍💻 Installation (for Developers)

To contribute to the development of BioLexis:

1.  Fork and clone the repository.

2.  Set up the development environment using the steps above.

3.  Install development dependencies (includes testing tools like pytest):
    bash
    pip install -r requirements-dev.txt
    
    4.  Run the test suite:
    bash
    pytest
    



---
## 🐞 Known Issues

* The Decision Engine currently iterates through sequences to apply rules. For extremely large datasets (millions of unique sequences), this loop could be vectorized for improved performance.
* The Known Islands Library check requires the user to manually curate a list of rare organism embeddings. Future work could involve automatically populating this library.

---
## 👥 Team

This project was developed for the Smart India Hackathon by Team Geek Velocity.



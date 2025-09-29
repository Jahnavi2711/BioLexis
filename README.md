# BioLexis: A Hybrid Intelligence Engine for Genomic Novelty Detection

A high-performance bioinformatics pipeline by *Team Geek Velocity that combines deep learning, supervised learning and Unspervised learning's anomaly detection to classify environmental DNA (eDNA) and uncover novel biodiversity.*

<img width="591" height="645" alt="image" src="https://github.com/user-attachments/assets/44672010-add1-43d3-a5a5-0b2f4fb0fc7a" />

<img width="591" height="645" alt="image" src="https://github.com/user-attachments/assets/7d812f9d-6680-4a09-a900-aebcc3cc1316" />

<img width="592" height="624" alt="image" src="https://github.com/user-attachments/assets/c7b3823b-fcd7-4a36-924d-8ae793515564" />

<img width="1206" height="533" alt="image" src="https://github.com/user-attachments/assets/6eb21dfc-556d-4bea-88f8-2af8dd5717a0" />




## 📖 Introduction

In fields like environmental monitoring, epidemiology, and metagenomics, quickly identifying organisms from DNA sequences is critical. While existing tools can identify known organisms, they often fail to characterize novel or divergent sequences, which may represent hidden biodiversity or emerging pathogens. 

BioLexis addresses this challenge by synthesizing evidence from two distinct machine learning arms to provide a confident, context-aware status for every sequence. It moves beyond simple classification to actively discover and flag novel biological entities.

---
## 🧬 Workflow & Demo

Our pipeline employs a two-pronged approach. The supervised arm provides taxonomic predictions based on a trusted reference database, while the unsupervised arm builds a "map of known biodiversity" to identify sequences that are statistical outliers or divergent members of known groups. A final Decision Engine synthesizes this evidence to deliver a definitive classification.

### 📹 Demo Video
Watch a full walkthrough of the BioLexis pipeline and its interactive report:

---

## 📂 Folder Structure:
```
BioLexis/
├── configs/ # Configuration files (YAML)
│ └── default.yaml
│
├── data/
│ ├── raw/ # Raw input files (CSV/FASTA)
│ │ ├── labels.csv # Reference data for training
│ │ └── new_sequences.fasta # Scientist's experimental sequences
│ └── processed/ # Cleaned/processed intermediate files
│
├── results/ # All pipeline outputs
│ ├── per_sequence_results.csv
│ ├── clusters_with_novelty.csv
│ ├── abundance_by_assignment.csv
│ └── report.html
│
├── src/ # Source code
│ ├── pipeline.py # Main pipeline orchestrator
│ ├── preprocess.py # Cleaning and deduplication
│ ├── kmers.py # K-mer feature generation
│ ├── embed.py # Autoencoder embeddings
│ ├── cluster.py # UMAP + HDBSCAN clustering
│ ├── label_transfer.py # Per-rank classifier training/prediction
│ ├── abundance.py # Abundance calculation
│ ├── diversity.py # Diversity metrics (Shannon, Simpson)
│ ├── visualize.py # Visualization utilities
│ └── evaluate.py # Performance evaluation
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── LICENSE

```
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

* Backend:   Python
* ML/Data Science:   Scikit-learn, PyTorch, Pandas, NumPy
* Dimensionality Reduction & Clustering:   UMAP-learn, HDBSCAN
* Visualization:   Plotly, Matplotlib,Seaborn
* Bioinformatics:   BioPython
* Reporting:   HTML
* Cloud/GPU:   Google Colab (T4/L4/A100), AWS

---
## ⚙ Installation (for Users)

To run the BioLexis pipeline on your own data, follow these steps:

1.  Clone the repository:
```
    bash
    git clone -[
    cd BioLexis
  ```  

2.  Create a virtual environment (recommended):
 ```
    bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
  ```  

3.  Install the required dependencies:
   
   ```
   bash
    pip install -r requirements.txt```
   ```
4.  Run the pipeline:

```
   bash
      python -m src.pipeline --config configs/fast.yaml --input data/raw/input.fasta --reference data/raw/labels.csv --out results/run1
    
  ```  

---

## 👨‍💻 Installation (for Developers)

To contribute to the development of BioLexis:

1.  Fork and clone the repository.

2.  Set up the development environment using the steps above.

3.  Install development dependencies (includes testing tools like pytest):

    ``` bash
        pip install -r requirements-dev.txt
    ```

4.  Run the test suite:

   
```
 bash
    pytest
 ```   



---
## 🐞 Known Issues

* The Decision Engine currently iterates through sequences to apply rules. For extremely large datasets (millions of unique sequences), this loop could be vectorized for improved performance.

---
## 👥 Team

This project was developed for the Smart India Hackathon by Team Geek Velocity.



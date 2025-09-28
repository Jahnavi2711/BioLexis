### **Project Overview**

The BioLexis project is a sophisticated bioinformatics pipeline designed to analyze environmental DNA (eDNA) from sources like water or soil. Its primary goal is to identify which organisms are present in a sample by classifying their 18S rRNA gene sequences—a common genetic marker.

What makes this project powerful is its **dual-analysis approach**. It doesn't just identify known organisms; it is also specifically designed to discover potentially new, unclassified, or anomalous sequences, making it a tool for both routine biodiversity monitoring and novel discovery. The project was developed in the context of the Smart India Hackathon 2025.

Here is a step-by-step breakdown of how the entire system works, from raw DNA sequence to final report.

---

### **Step 1: Feature Engineering (Turning DNA into Data)**

The pipeline's first job is to convert raw genetic sequences (strings of A, T, C, G) into a numerical format that a machine learning model can understand.

* **Multi k-mer Counting:** The system slides a window of size 'k' across each DNA sequence to count the occurrences of all possible DNA substrings of that length. For example, if k=4, it counts "ATGC", "TGCA", etc.
* **Multi-Scale Approach:** BioLexis cleverly uses multiple 'k' sizes simultaneously (k=4, 8, and 12). This is a key feature.
    * **k=4:** Short k-mers capture broad, fundamental compositional signatures of the DNA.
    * **k=8:** Medium-length k-mers provide more specific signals, often corresponding to genus-level biological features.
    * **k=12:** Long k-mers capture highly specific motifs that can be unique to a particular species.

By combining these scales, the system creates a rich, high-dimensional numerical vector for each sequence that represents its biological information at different levels of detail.

### **Step 2: Feature Weighting (Identifying Important Signals)**

The raw k-mer counts can be noisy. Some k-mers are very common across all life forms and don't provide much useful information for telling species apart.

* **TF-IDF (Term Frequency-Inverse Document Frequency):** To solve this, the pipeline applies TF-IDF, a technique borrowed from text analysis. It re-weights the k-mer counts.
    * It **amplifies the importance** of rare k-mers that are discriminative (i.e., specific to only a few organisms).
    * It **suppresses the importance** of universally common k-mers that are essentially biological noise.

This step refines the numerical vector into a more potent "genomic signature" that is optimized for classification.

### **Step 3: Dimensionality Reduction (Making Data Efficient)**

The TF-IDF vectors are still very large and computationally intensive to work with. The next step is to intelligently compress them without losing the essential information.

* **Deep Autoencoder:** The system uses an unsupervised neural network called an Autoencoder. It's trained to take the high-dimensional vector, squeeze it through a narrow "bottleneck," and then try to reconstruct the original vector on the other side.
* **Dense Embedding:** The compressed data in the bottleneck is a **dense latent representation**, or an "embedding." This low-dimensional embedding optimally captures the primary patterns (variance) in the data while filtering out noise. This final, efficient embedding is used for all subsequent analysis.

### **Step 4: The Dual-Analysis Engine**

This is the core of the project where the actual analysis happens. The embedding for each sequence is sent to two parallel arms simultaneously.

* **Arm 1: Supervised Classification**
    * **Goal:** To identify known organisms.
    * **Method:** This arm uses a suite of **hierarchical, per-rank Logistic Regression models**. Based on your saved information, this aligns with the methods you are using. The hierarchical structure is crucial because it mirrors biological taxonomy (Kingdom > Phylum > Class > ... > Species). The models assign a precise probability that the sequence belongs to a specific known organism based on the decision boundaries they learned during training on labeled data.

* **Arm 2: Unsupervised Discovery**
    * **Goal:** To detect novel or anomalous sequences.
    * **Method:** This arm uses a two-tiered statistical approach for novelty detection.
        1.  **Density-Based Clustering (HDBSCAN):** First, the HDBSCAN algorithm maps the overall structure of all the sequence embeddings. Any embedding that falls in a low-density region—far away from any established cluster of known organisms—is immediately flagged as a major outlier.
        2.  **Intra-Cluster Anomaly Detection:** For embeddings that *do* fall within a known cluster, a second check is performed. The system calculates the distance of the embedding from the center of its assigned cluster. If this distance exceeds a statistical threshold (e.g., it's further away than 98% of the other members), it is flagged as an anomaly—an atypical member of that group.

### **Step 5: Rule-Based Synthesis (The Final Verdict)**

The final step is to integrate the results from both the Classification Arm and the Discovery Arm to make a single, coherent decision for each sequence. A simple rule-based engine assigns one of four flags:

1.  **Known Organism:** The classifier confidently identifies the sequence, and it sits comfortably inside its expected genetic cluster. (Both arms agree).
2.  **High-Confidence Novelty:** The sequence is a major outlier according to the clustering model and cannot be classified. (Clear discovery).
3.  **Novel Candidate (or Unusual Find):** The sequence is flagged as an anomaly within a known group. It's too atypical to be confidently classified to the species level but clearly belongs to a known family or genus. (Potential new species/strain).
4.  **Flagged for Review (or Conflicting Result):** The classifier confidently identifies the sequence, but the discovery arm flags it as a structural outlier for that very same group. (The arms disagree, requiring manual expert review).

### **Final Output**

The entire analysis is compiled into an interactive HTML report. This user-friendly dashboard includes:
* **A UMAP visualization:** A 2D map that visually plots the clusters of sequences, allowing for an intuitive understanding of the sample's biodiversity.
* **Sample Richness Metrics:** Quantitative data about the diversity found in the sample.
* **Clear Classification:** Each sequence is clearly labeled with one of the four outcomes, allowing for immediate visual analysis and interpretation.

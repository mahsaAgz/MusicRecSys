# Addressing the Blurry Collaborative Embedding Problem for Cold-Start Music Recommendation
## Overview
This repository contains the **Music Recommendation** project for the **Recommender System and Graph Machine Learning (DS535)** course at **KAIST**.  You can find the detail in final_report.pdf

Our model, **Contrastive Collaborative Filtering for Cold-Start Recommendation (CCFCRec)**, enhances music recommendation by:
- Addressing the **cold-start problem** in new music and users.
- Combining **co-occurrence collaborative filtering** with **content-based filtering**.
- Using **contrastive learning** to refine recommendations and improve diversity.

## Features
- **Handles Cold-Start Users & Items** effectively.
- **Improves recommendation diversity** by leveraging artist & album data.
- **Contrastive learning framework** for better embeddings.
- **Outperforms KNN-based models** in recommendation accuracy.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/MusicRecommendation.git
cd MusicRecommendation
pip install -r requirements.txt
```

## Dataset
We use the **Yahoo! Music R2 Dataset**, which includes:
- **1.8 million users**
- **136,000 songs**
- **717 million user-item interactions**

Download the dataset:  
[Yahoo Dataset](https://drive.google.com/file/d/1NFe0jWa-RdP9wcSGRe6KXMbR0ENXlr_B/view)  

To process fresh data, remove old processed files and generate negatives:

```bash
python generate_neg.py
```

## Usage
Run the model to generate music recommendations:

```bash
python model.py
```

## Methodology
Our model **CCFCRec** consists of:
1. **Content-Based Collaborative Filtering (CBCE):** Generates embeddings from song metadata.
2. **Co-occurrence Collaborative Filtering (COCE):** Learns user-item interaction patterns.
3. **Contrastive Learning Mechanism:** Transfers knowledge between CBCE and COCE for **better cold-start recommendations**.
![Picture1](https://github.com/user-attachments/assets/5570c95b-de12-4a8e-8f97-1d04bd72f0b9)

The **loss function** is defined as:

$$
L = L_{inter} + \lambda L_{contrast} + L_{reg}
$$

where:
- $L_{inter}$: Interaction loss for user-item predictions.
- $L_{contrast}$: Contrastive loss to align content & co-occurrence embeddings.
- $L_{reg}$: Regularization term to prevent overfitting.

## Performance
Our model significantly outperforms **K-Nearest Neighbors (KNN)** in recommendation accuracy.

| Model   | HR@5   | HR@10  | HR@20  | NDCG@5  | NDCG@10  | NDCG@20  |
|---------|--------|--------|--------|---------|----------|----------|
| KNN     | 0.0005 | 0.0010 | 0.0011 | 0.0012  | 0.0037   | 0.0065   |
| **CCFCRec** | **0.0043** | **0.0037** | **0.0033** | **0.0003** | **0.0004** | **0.0005** |

## Future Work
- **Enhancing ranking optimization** to improve NDCG scores.
- **Adding real-time adaptation** using reinforcement learning.
- **Incorporating more song attributes** (e.g., tempo, mood).

## Contributors
- **Mahsa Aghazadeh** (KAIST, GSDS)
- **Karin Rizky Irminanda** (KAIST, ISysE)
- **Dewei Zhu** (KAIST, ISysE)
- **Anurag Yadav** (KAIST, GSDS)

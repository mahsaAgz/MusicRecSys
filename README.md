# MusicRecommendation

## Overview
This repository contains the MusicRecommendation project developed for DS535 at KAIST. The project aims to provide music recommendations by leveraging machine learning models.

## Getting Started
### Dataset

The project uses the Yahoo dataset for training and evaluation. You can use pre-processed data in the data folder or  You can download the dataset from the following link 
[Yahoo dataset](https://drive.google.com/file/d/1NFe0jWa-RdP9wcSGRe6KXMbR0ENXlr_B/view)
#### Note
If you want to use other data than our pre-processed data or change the data, You need to delete pkl files and train_rating_with_neg_1.csv in the data folder and run generate_neg.py to have negative users

### Usage
To start generating music recommendations, execute the `model.py` script:

```bash
python model.py


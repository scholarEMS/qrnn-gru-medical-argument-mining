# qrnn-gru-medical-argument-mining
Implementing a QRNN–GRU based neural architecture for automatic argument annotation in medical drug reviews. The framework combines word embeddings, sequential modeling, and structured drug metadata to analyze argumentative and sentiment patterns with preprocessing, embedding generation, training, evaluation, and optimization components 
# QRNN-GRU Argument Mining (Medical Reviews)

This repository reproduces the experiments in:
"QRNN–GRU Framework for Automatic Argument Annotation in Medical Drug Reviews"
The full pipeline implemented in this repository:
Data Preprocessing → Word Embedding Generation → Model Training → Evaluation → Optimization
## Installation
pip install -r requirements.txt

## Dataset
Download Drug Review dataset from UCI ML Repository.
https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+(Drugs.com)
Place the downloaded files in data/raw/
data/raw/drugLibTrain_raw.tsv
data/raw/drugLibTest_raw.tsv

src/
 ├── preprocessing.py
 ├── embeddings.py
 ├── data_loader.py
 ├── qrnn.py
 ├── qrnn_gru.py
 ├── train.py
 ├── evaluate.py
 ├── firefly.py
 └── utils.py

data/
 ├── raw/          # Place downloaded dataset here
 └── processed/    # Generated automatically

results/           # Generated after training/evaluation
requirements.txt
README.md

## Preprocessing. Step 1 — Preprocess Text Data
python src/preprocessing.py

## Embedding. Step 2- Train Word Embeddings
python src/embeddings.py

## Training. Step 3- Train the QRNN-GRU Model
python src/train.py

## Evaluation.Step 4-Evaluate the Model
python src/evaluate.py

## Optimization
Firefly optimization is implemented in src/firefly.py
Model Architecture Overview

The proposed framework integrates:

QRNN (Quasi-Recurrent Neural Network) for fast sequential feature extraction

GRU (Gated Recurrent Unit) for contextual temporal modeling

Structured feature encoding for metadata such as effectiveness, side effects, and condition

Feature fusion layer combining text and structured features

Final classification layer for argument category prediction

An optional Firefly Optimization module is included for hyperparameter tuning.

# SubsequenceExtraction_SR

## Introduction
This is the implementation of the paper Improving Sequential Recommendation via Subsequence Extraction.

## Requirements
+ Python = 3.6
+ numpy = 1.19.2
+ pandas = 1.2.2
+ pytorch = 3.7
+ tqdm = 4.57.0
+ matplotlib = 3.3.4

## Usage (ML-1M example)

Preprocess the data: python preprocess.py

Train&Evaluate: python train_transformer_np_5.py --dataset=SASRecM1M --n_interests=8 --n_layer=2 --n_neg=30 --lam=0 --l2=0.0001 --r_loss=0.0 --dropout=0.2 --max_len=200 --gpu=1 &

See the visualization: run visualize.ipynb in jupyter notebook

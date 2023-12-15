# RI Project 2 - Neural Reranker

## Description

This project required us to implement, train and evaluate a neural reranking model capable of
boosting the ranking order produced by our previously developed searching system. It takes the top results of the documents from BM25 runs and reranks them, having the goal to push more relevant results to the top of the list.

We performed a pairwise training of the model, feeding it positive (relevant) and negative (non-relevant) query-document pairs, so that it could learn to assign a higher score for positive pairs and a lower score for negative pairs. After the training, the model was able to assign a relevancy score to every individual query-document pair, and was able to improve the order of the BM25 runs, providing more relevant documents towards the top of the list.

**Course:** Information Retrieval (2023/2024).

## Walkthrough

- Refer to the content inside `ri_neural_reranker.ipynb` for a complete explaination of each step taken in this work.

## Authors

- Diogo Paiva, 103183

- João Fonseca, 103154

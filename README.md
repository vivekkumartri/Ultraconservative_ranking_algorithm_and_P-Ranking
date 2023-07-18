# Ordinal Regression Algorithms for Ranking Learning

## Introduction

Ordinal regression is a type of learning that involves ranking instances with a rating between 1 and k, with the objective of accurately predicting the rank of each instance. Traditional classification and regression techniques are not well-suited for ranking tasks due to the structured labels with a total order relation between them. Therefore, specialized algorithms for ranking learning have been developed.

In this project, we explore a group of closely related online algorithms for ordinal regression and evaluate their performance in terms of ranking loss. We present an alternative approach to ranking learning that directly maintains a totally ordered set via projections. Our algorithm uses projections to associate each ranking with a distinct sub-interval of the real numbers and adapts the support of each sub-interval while learning.

To evaluate the performance of our algorithm, we compare it with seven pre-existing algorithms on four different data sets. The data sets used include synthetic data, Each-Movie data set for collaborative filtering, IMDB movie data set for collaborative filtering, and TMDB movie data set. Our results highlight the strengths of the P-Ranking algorithm on the first two data sets, while the multi-class perceptron algorithm outperforms others on the last two data sets. We also propose a neural network approach for ranking and discuss some natural language processing algorithms used in our project.

## Project Structure

The project repository is organized as follows:

- `/data`
  - `/synthetic_data`
    - `synthetic_dataset.csv`
  - `/each_movie`
    - `each_movie_dataset.csv`
  - `/imdb_movie`
    - `imdb_movie_dataset.csv`
  - `/tmdb_movie`
    - `tmdb_movie_dataset.csv`
- `/models`
  - `/pranking_algorithm`
    - `pranking_model.py`
  - `/multiclass_perceptron`
    - `multiclass_perceptron_model.py`
- `/utils`
  - `data_preprocessing.py`
  - `evaluation_metrics.py`
- `README.md`
- `requirements.txt`

The `/data` directory contains the datasets used in the experiments. Each dataset is organized in a separate directory and contains a CSV file with the relevant data. The `/models` directory contains the implementations of the algorithms used in the project. Each algorithm has its own directory, which contains the corresponding model implementation. The `/utils` directory includes utility functions used for data preprocessing and evaluation metrics calculation. The `README.md` file provides an overview of the project, and the `requirements.txt` file lists the dependencies required to run the project.

## Getting Started

To run the project, follow these steps:

1. Clone the repository:

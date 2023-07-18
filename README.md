# Ordinal Regression Algorithms for Ranking Learning

## Introduction

Ordinal regression is a type of learning that involves ranking instances with a rating between 1 and k, with the objective of accurately predicting the rank of each instance. Traditional classification and regression techniques are not well-suited for ranking tasks due to the structured labels with a total order relation between them. Therefore, specialized algorithms for ranking learning have been developed.

In this project, we explore a group of closely related online algorithms for ordinal regression and evaluate their performance in terms of ranking loss. We present an alternative approach to ranking learning that directly maintains a totally ordered set via projections. Our algorithm uses projections to associate each ranking with a distinct sub-interval of the real numbers and adapts the support of each sub-interval while learning.

To evaluate the performance of our algorithm, we compare it with seven pre-existing algorithms on four different data sets. The data sets used include synthetic data, Each-Movie data set for collaborative filtering, IMDB movie data set for collaborative filtering, and TMDB movie data set. Our results highlight the strengths of the P-Ranking algorithm on the first two data sets, while the multi-class perceptron algorithm outperforms others on the last two data sets. We also propose a neural network approach for ranking and discuss some natural language processing algorithms used in our project.

## Project Structure

The project repository is organized as follows:

- `algorithm.py`
- `/synthetic_data_implementation.py`
- `/tmdb_implementation.py`
- `/each_movie_implementation.py`
- `imdb_implementation.py`
  - ``
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


## Results and Discussion

The results obtained from the experiments are presented in detail in the project report. We compare the performance of our algorithm with seven pre-existing algorithms on four different datasets. The evaluation metrics used include ranking loss, accuracy, and precision-recall measures. The findings highlight the strengths and weaknesses of each algorithm on different datasets.

We observed that the P-Ranking algorithm outperformed other algorithms on the synthetic data and Each-Movie data sets. However, the multi-class perceptron algorithm showed superior performance on the IMDB and TMDB movie data sets. We also discuss the limitations of P-Ranking and the importance of considering the distribution of data in high-dimensional space when selecting an appropriate algorithm for ordinal regression.

## Conclusion

In conclusion, this project explored various online algorithms for ordinal regression and evaluated their performance on different datasets. We introduced an alternative approach to ranking learning that directly maintains a totally ordered set via projections. Our findings demonstrated the strengths and weaknesses of different algorithms and emphasized the importance of algorithm selection based on the characteristics of the data.

The project also provided implementations of the P-Ranking algorithm and the multi-class perceptron algorithm, along with utility functions for data preprocessing and evaluation metrics calculation. Researchers and practitioners can use this project as a reference for understanding and implementing ordinal regression algorithms for ranking learning.

Please refer to the [project report](https://github.com/vivekkumartri/Ultraconservative_ranking_algorithm_and_P-Ranking/blob/d535e64e9e4a496d524467f9e3b97ec072a82c2c/Vasusena_IE506_CourseProject_EndtermReview_Report.pdf) for more detailed information on the experiments, results, and future directions.

## References

[List the references used in the project]

1. [Reference 1]
2. [Reference 2]
3. [Reference 3]
4. [Reference 4]

[Include any other relevant sections or information as necessary]


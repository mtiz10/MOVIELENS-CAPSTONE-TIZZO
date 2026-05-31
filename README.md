# MovieLens Rating Prediction Project 🎬📊
**HarvardX: PH125.9x Data Science Capstone**

##  Overview
This repository contains the Capstone Project for the HarvardX Professional Certificate in Data Science. The objective of this project is to build a robust recommendation system capable of predicting movie ratings using the **MovieLens 10M dataset**. 

The algorithm was developed using a progressive, evidence-based approach, advancing from a baseline global average to a **Regularized Multilevel Model** to account for inherent human biases and structural data sparsity.

##  Methodology: The "Staircase of Complexity"
Instead of deploying a black-box algorithm, the predictive architecture was engineered incrementally. Every mathematical expansion was justified by empirical discoveries from the Exploratory Data Analysis (EDA):

1. **Naive Baseline Model**: A fundamental statistical anchor using only the global mean.
2. **Movie Effect Model**: Adjusted the baseline by incorporating specific movie quality and popularity ($b_i$).
3. **Movie + User Effects Model**: Refined predictions by factoring in individual evaluator temperaments and subjective strictness ($b_u$).
4. **Regularized Multilevel Model (Final)**: Implemented an optimized penalty parameter ($\lambda$) to shrink high-variance estimates from sparse, low-sample outliers toward the global mean, actively preventing overfitting.

##  Results
The performance of the models was evaluated using the **Root Mean Squared Error (RMSE)**. The final regularized model successfully outperformed the strict target benchmark of **0.86490**.

| Model Architecture | RMSE |
| :--- | :--- |
| Naive Average | 1.06120 |
| Movie Effect | 0.94391 |
| Movie + User Effects | 0.86535 |
| **Regularized (Final)** | **0.86482** 🏆 |

##  Repository Structure
* `MOVIELENS-CAPSTONE-TIZZO.pdf`: The comprehensive project report detailing the theoretical framework, EDA, mathematical justifications, and conclusions.
* `MOVIELENS-CAPSTONE-TIZZO.Rmd`: The R Markdown source file used to dynamically generate the report and visualizations.
* `MOVIELENS-CAPSTONE-TIZZO.R`: The standalone R script containing the complete workflow (data ingestion, wrangling, modeling, and evaluation) for reproducibility.

##  Technologies & Libraries
* **Language**: R
* **Data Wrangling**: `tidyverse`, `dplyr`, `data.table`
* **Visualization**: `ggplot2`, `patchwork`, `scales`
* **Machine Learning & Stats**: `caret`

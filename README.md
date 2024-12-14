# Bayesian Matrix Factorization for Disease Risk Prediction

This repository implements a Bayesian matrix factorization approach to predict disease status using gene expression and demographic data, focusing on inflammatory bowel diseases (IBD) such as Crohn's disease and ulcerative colitis.

## Project Overview

Accurate disease risk prediction is crucial for early diagnosis and personalized healthcare. Traditional deep learning models often face challenges with high-dimensional, noisy gene expression data and lack biological interpretability. To address these issues, we propose a Bayesian matrix factorization method that reduces noise and dimensionality, uncovers biologically meaningful latent factors, and effectively predicts disease status by integrating patient demographic information.

## Methodology

### Data Acquisition and Preprocessing

We utilized whole blood gene expression data from a study by Jonas Halfvarson et al. The dataset includes gene-level expression measurements and patient-level covariates such as age and sex. Samples were categorized into three groups: healthy controls, Crohn's disease patients, and ulcerative colitis patients. Gene expression data was filtered to retain genes with significant variance, and demographic variables were encoded appropriately for analysis.

### Model Specification

Our supervised hierarchical Poisson factorization model decomposes the gene expression matrix into two low-dimensional matrices:

- **Factor Matrix (θ)**: Captures latent biological processes for each sample.

- **Loading Matrix (β)**: Represents each gene's contribution to these latent factors.

Disease status is modeled using softmax regression on the latent factors and auxiliary demographic variables, allowing for the integration of gene expression profiles and patient information in disease prediction.

### Parameter Inference

We employed a Bayesian inference strategy, combining Gibbs sampling and Laplace approximations, to estimate the posterior distributions of latent factors and model parameters. This approach facilitates the extraction of meaningful biological insights and enhances the interpretability of the model.

## Results

The model's performance was evaluated using classification metrics such as accuracy, confusion matrices, and classification reports. Experiments were conducted with varying numbers of latent dimensions and iterations to assess convergence and predictive capabilities. The results indicate that the Bayesian matrix factorization approach effectively captures underlying biological processes and provides competitive accuracy in disease status prediction compared to baseline methods.

## Discussion

Comparative analysis with baseline models, including logistic regression and principal component analysis, demonstrates the advantages of the Bayesian approach in handling high-dimensional gene expression data and enhancing interpretability. The integration of demographic information further improves predictive performance, underscoring the importance of a holistic approach in disease risk prediction.

## References

- Halfvarson, J., et al. (2022). *Title of the Study*. *Journal Name*, *Volume(Issue)*, Page numbers.

- Furey, T. S., et al. (2000). Support vector machine classification and validation of cancer tissue samples using microarray expression data. *Bioinformatics*, 16(10), 906-914.

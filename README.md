# Improvements to Collaborative Filtering Algorithm

## Introduction
Recommendation system has been widely used in different kinds of websites,, in order to provide more accurate products or services to them. Collaborative Filtering algorithm (CF) is considered to be the most common algorithm to construct the system. However, constrained by the sparsity of dataset and consider different real-world problems, CF algorithms still suffers from some shortcomings that may induce to giving a wrong recommendation.

In some cases,users or movies that may have exact same rating patterns can not be judged as neighbors in the CF algorithm. Thus, the low similarity of neighbors will have a bad effect on the performance of the recommendation, raised a method to help the algorithm find a better neighbors, thus contribute to the reduce of RMSE, we also tried some other statistical methods for the same goals.

## Methods Summary
**- Method tend to Improve**  
K-Nearest Neighbor Collaborate Filtering Algorithm (use RMSE as metrics)

**- Improvements**  
Remove User and Movie Effects (Normalization), SVD

**- Methods used for cross validation and hypothesis test**  
K-fold cross validation, validation set, t-test

NOTES

## 1. TASK: Do Accent ID only using text features
1/22
- for 7 langs, # of 20sec files maintained (out of total files): 
	- 922/1045 (88%) in train: `/traintestsplit/7lang.trainlist.sorted.20sec`
	- 304/348 (87%) in test: `/traintestsplit/7lang.testlist.sorted.20sec`
	- 20sec train/test split = 75/25!
- chance = *26%*
OUTPUT:
Stored predictions in `multiview/knn.{k}.predictions` for test points
| k | Accuracy |
----|----------|
| 2 | 37.17% |
| 3 | 39.14% |
| 4 | 39.47% |
| 5 | 41.12% |
| 6 | 39.47% |
| 7 | 40.13% |
| 8 | 41.45% |
| 9 | 41.78% |
| 10 | 41.12% |
| 18 | 41.12% |
| 19 | 41.12% |
| 20 | *43.75%* |
| 21 | 42.11% |
| 25 | 40.79% |
| 30 | 39.47% |
| 40 | 40.79% |

For future:
- logistic regression - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- k neighbors regressor (?) - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
- csr\_matrix - https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.csr_matrix.html
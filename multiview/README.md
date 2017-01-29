## 1. TASK: Do Accent ID only using text features
1/29
- used [logistic regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) from sklearn 
Overall Accuracy: *_61.18%_*
| lang | acc | acc/total\_acc / rel # test files
----|-----|------------|
|AR | 32.00% | 1.07 |
|CZ | 21.74% | _0.79_ |
|FR | 75.81% | 1.03 |
|HI | 77.22% | 0.82 |
|IN | 38.10% | _1.52_ |
|KO | 50.00% | 1.17 |
|MA | 67.24% | 0.97 |

- general analysis of test set: # of test files and % out of whole test set
	AR: 25 (8.2%)
	CZ: 23 (7.6%)
	FR: 62 (20.4%)
	HI: 79 (26.0%)
	IN: 21 (6.9%)
	KO: 36 (11.8%)
	MA: 58 (19.1%)

1/22
- for 7 langs, # of 20sec files maintained (out of total files): 
	- 922/1045 (88%) in train: `/traintestsplit/7lang.trainlist.sorted.20sec`
	- 304/348 (87%) in test: `/traintestsplit/7lang.testlist.sorted.20sec`
	- 20sec train/test split = 75/25!
- chance = *26%* (percentage of Hindi test files--79/304)

OUTPUT:
Stored predictions in `multiview/knn_predictions_170123/knn.{k}.predictions` for test points

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
| 20 | ***43.75%*** |
| 21 | 42.11% |
| 25 | 40.79% |
| 30 | 39.47% |
| 40 | 40.79% |

-----------------------
MISC:
- k neighbors regressor (?) - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
- csr\_matrix - https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.csr_matrix.html
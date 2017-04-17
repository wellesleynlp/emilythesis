## 2. TASK: Try initial run of CCA on speech + text
4/16


## 1. TASK: Do Accent ID only using text features
1/29
- used [logistic regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) from sklearn 

Overall Accuracy: *_61.18%_*

| lang | acc | acc/total\_acc / rel # test files
----|-----|------------|
|AR | 32.00% | 1.07 |
|CZ | 21.74% | _**0.79**_ |
|FR | 75.81% | 1.03 |
|HI | 77.22% | 0.82 |
|IN | 38.10% | _**1.52**_ |
|KO | 50.00% | 1.17 |
|MA | 67.24% | 0.97 |

- general analysis of test set: # of test files and % out of whole test set

AR: 25 (8.2%)\n
CZ: 23 (7.6%)\n
FR: 62 (20.4%)\n
HI: 79 (26.0%)\n
IN: 21 (6.9%)\n
KO: 36 (11.8%)\n
MA: 58 (19.1%)\n

- removing stop words: overall accuracy was the exact same (61.18%), but notably Korean did worse while Czech and Indonesian did better. Results in logreg.nostop*


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
| 5 | 41.12% |
| 10 | 41.12% |
| 19 | 41.12% |
| 20 | ***43.75%*** |
| 21 | 42.11% |
| 25 | 40.79% |
| 30 | 39.47% |

-----------------------
MISC:
- k neighbors regressor (?) - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
- csr\_matrix - https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.csr_matrix.html
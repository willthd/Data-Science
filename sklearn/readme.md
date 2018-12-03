# SKLearn



### GridSearch

```python
### SVM classifier
SVMC = SVC(probability=True)
svc_param_grid = {"kernel": ['rbf'], 
                  "gamma": [ 0.001, 0.01],
                  "C": [1, 10, 50, 100, 200, 300, 1000],
                 "random_state" : [2]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=10, scoring="accuracy", n_jobs= -1, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_()

# Best score
gsSVMC.best_score_
```


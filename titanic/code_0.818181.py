# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# load dataSet

train = pd.read_csv("/Users/PJS/Desktop/dsSchool/day1/dataSet/train.csv", index_col="PassengerId")
test = pd.read_csv("/Users/PJS/Desktop/dsSchool/day1/dataSet/test.csv", index_col="PassengerId")

print(train.shape)
print(test.shape)

# preprocess

## Age, 꼭 필요 하지 않다. 어차피 age는 feature로 안들어가고 child만 들어가니까
train.loc[train["Age"].isnull() & (train["Name"].str.contains("Master") | train["Name"].str.contains("Miss")), "Age"] = 5
train.loc[train["Age"].isnull() & (train["Name"].str.contains("Mr") | train["Name"].str.contains("Mrs") | train["Name"].str.contains("Dr.")), "Age"] = 30

test.loc[test["Age"].isnull() & (test["Name"].str.contains("Master") | test["Name"].str.contains("Miss")), "Age"] = 5
test.loc[test["Age"].isnull() & (test["Name"].str.contains("Mr") | test["Name"].str.contains("Mrs") | test["Name"].str.contains("Ms.")), "Age"] = 30

##Child
train["Child"] = train["Age"] < 15
test["Child"] = test["Age"] < 15

## Master
train["Master"] = train["Name"].str.contains("Master")
test["Master"] = test["Name"].str.contains("Master")

## Fare
train["Fare_fillin"] = train["Fare"]
test["Fare_fillin"] = test["Fare"]

test.loc[test["Fare"].isnull(), "Fare_fillin"] = 0

## Embarked
train["Embarked_C"] = (train["Embarked"] == "C")
train["Embarked_S"] = (train["Embarked"] == "S")
train["Embarked_Q"] = (train["Embarked"] == "Q")

test["Embarked_C"] = (test["Embarked"] == "C")
test["Embarked_S"] = (test["Embarked"] == "S")
test["Embarked_Q"] = (test["Embarked"] == "Q")

## Sex
train.loc[train["Sex"] == "male", "Sex_encode"] = 1
train.loc[train["Sex"] == "female", "Sex_encode"] = 0
test.loc[test["Sex"] == "male", "Sex_encode"] = 1
test.loc[test["Sex"] == "female", "Sex_encode"] = 0

print(train.shape)
print(test.shape)

## FamilySize

train["Family_size"] = train["SibSp"] + train["Parch"] + 1
test["Family_size"] = test["SibSp"] + test["Parch"] + 1

train["Single"] = train["Family_size"] == 1
train["Nuclear"] = (train["Family_size"] < 5) & (train["Family_size"] > 1)
train["Big"] = train["Family_size"] >= 5

test["Single"] = test["Family_size"] == 1
test["Nuclear"] = (test["Family_size"] < 5) & (test["Family_size"] > 1)
test["Big"] = test["Family_size"] >= 5



# features 순서도 영향을 끼치네?
features = ["Pclass", "Sex_encode", "Fare_fillin",
                 "Embarked_C", "Embarked_S", "Embarked_Q",
                 "Child", "Single", "Nuclear", "Big", "Master"]

label = "Survived"

X_train = train[features]
print(X_train.shape)

Y_train = train[label]
print(Y_train.shape)

X_test = test[features]

model = SVC(C=1, kernel='rbf', coef0=1)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

submit = pd.read_csv("/Users/PJS/Desktop/dsSchool/day1/dataSet/gender_submission.csv", index_col="PassengerId")
submit[label] = predictions
print(submit.head(10))
submit.to_csv("/Users/PJS/Desktop/dsSchool/day1/dataSet/myself.csv")



import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
import sys
from sklearn.metrics import classification_report



def convert_numbers_to_labels(dataset):
    dataset['target_class'].replace([0.0, 1.0], ['ham', 'spam'], inplace=True)
    return dataset

def convert_labels_to_numbers(dataset):
    dataset['target_class'].replace(['ham', 'spam'],[0.0, 1.0], inplace=True)
    return dataset

def clean_data(data_train,data_test):
    if 'file_names' in data_train.columns:
        del data_train["file_names"]
    if 'Unnamed: 0' in data_train.columns:
        del data_train["Unnamed: 0"]
    if 'file_names' in data_test.columns:
        del data_test["file_names"]
    if 'Unnamed: 0' in data_test.columns:
        del data_test["Unnamed: 0"]
    data_train = convert_labels_to_numbers(data_train)
    data_test = convert_labels_to_numbers(data_test)
    to_drop = [x for x in data_test.columns.tolist() if x not in data_train.columns.tolist()]
    to_add = [x for x in data_train.columns.tolist() if x not in data_test.columns.tolist()]
    data_test.drop(to_drop, axis=1, inplace=True)
    for col in to_add:
        data_test[col] = 0.0

    data_test = data_test.sort_index(axis=1)
    data_test = data_test[[col for col in data_test.columns if col != 'target_class'] + ['target_class']]
    return data_train, data_test


use_vectors = sys.argv[1]

if(use_vectors == "bag_of_words"):
    data_train = pd.read_csv("../programs/bag_of_words_matrix_train.csv")
    data_test = pd.read_csv("../programs/bag_of_words_matrix_test.csv")
elif(use_vectors == "bernolli"):
    data_train = pd.read_csv("../programs/bernolli_matrix_train.csv")
    data_test = pd.read_csv("../programs/bernolli_matrix_test.csv")




train_data,test_data = clean_data(data_train,data_test)

X = train_data.iloc[:,:-1]
Y = train_data.iloc[:,-1]
"""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
"""
classifier = SGDClassifier(random_state=50)
print("Training........")
alpha = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
n_iter = [50]
loss = ['log']
penalty = ['l2']
n_jobs = [-1]

param = {'alpha': alpha, 'max_iter': n_iter, 'loss': loss,'penalty': penalty, 'n_jobs': n_jobs}
grid_search = GridSearchCV(estimator=classifier, param_grid=param, cv=5)
grid_search.fit(X, Y)
best_grid = grid_search.best_estimator_

print("Predicting....")
main_test_data_train = test_data.iloc[:,:-1]
main_test_data_test = test_data.iloc[:,-1]
pred = best_grid.predict(main_test_data_train)
print('Tuned accuracy after grid search_trained data:',accuracy_score(np.asarray(main_test_data_test), np.asarray(pred)))
c_report = classification_report(np.array(main_test_data_test), np.asarray(pred))
print(" ")
print(c_report)


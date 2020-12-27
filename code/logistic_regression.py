from math import exp
from numpy import *
import pandas as pd
#from programs.naive_bayes import *
import time
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import sys
import warnings
warnings.filterwarnings("ignore")



def sigmoid(x):
    return 1.0/(1+exp(-x))

def convert_numbers_to_labels(dataset):
    dataset['target_class'].replace([0.0, 1.0], ['ham', 'spam'], inplace=True)
    return dataset

def convert_labels_to_numbers(dataset):
    dataset['target_class'].replace(['ham', 'spam'],[0.0, 1.0], inplace=True)
    return dataset


def train_logistic_regression(dataset,target,lambdaN):
    labelSet = mat(target).transpose()
    dataSet = mat(dataset)
    dataSet = dataSet.astype(float)
    m,n = shape(dataSet)
    alpha = 0.1
    #alpha = alpha
    numIteration = 50
    weights = mat(zeros((n,1)))
    for k in range(numIteration):
        h = sigmoid(dataSet*weights)
        error = (labelSet - h)
        weights = weights + alpha * dataSet.transpose() * error - alpha*lambdaN*weights
    return weights

def classify(weight,data):
    list_of_list = data.values.tolist()
    dataMatrix = mat(list_of_list)
    dataMatrix = dataMatrix.astype(float)
    wx = dataMatrix * weight
    results_target = []
    for i in range(len(wx)):
        if wx[i][0] < 0.0:
            results_target.append(0.0)
        else:
            results_target.append(1.0)
    return results_target

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



def hypeparameter_tune(X_train,y_train):
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.30)
    lamda = [0.1,0.001,0.01,0.0001]
    current_max_accuracy = -1
    best_lambda = -1
    for i in lamda:
        weights = train_logistic_regression(X_train,y_train,i)
        prediction_t = classify(weights, X_test)
        score = accuracy_score(asarray(y_test), asarray(prediction_t))
        if (score>=current_max_accuracy):
            current_max_accuracy = score
            best_lambda = i

    return best_lambda




use_vectors = sys.argv[1]

if(use_vectors == "bag_of_words"):
    data_train = pd.read_csv("../programs/bag_of_words_matrix_train.csv")
    data_test = pd.read_csv("../programs/bag_of_words_matrix_test.csv")
elif(use_vectors == "bernolli"):
    data_train = pd.read_csv("../programs/bernolli_matrix_train.csv")
    data_test = pd.read_csv("../programs/bernolli_matrix_test.csv")
else:
    print("path issue....")


print("Training........")
train, test = clean_data(data_train,data_test)
X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]
main_test_data = test.iloc[:,:-1]
label_main_test_data = test.iloc[:,-1]
best_lamda = hypeparameter_tune(X_train,y_train)
weights = train_logistic_regression(X_train,y_train,best_lamda)
print("Predicting....")
prediction_main_list = classify(weights,main_test_data)
cm = accuracy_score(array(prediction_main_list), asarray(label_main_test_data))
#print("f1_score",f1_score(array(prediction_main_list), asarray(label_main_test_data)))
#print("precision",precision_score(array(prediction_main_list), asarray(label_main_test_data)))
#print("recall",precision_score(array(prediction_main_list), asarray(label_main_test_data)))
c_report = classification_report(array(prediction_main_list), asarray(label_main_test_data))
print(cm)
print(c_report)





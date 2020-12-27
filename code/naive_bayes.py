import os
import sys
import numpy as np
import pandas as pd
import math
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import regex as re


def train_multinomial_naive_bayes(dataset):
    print("Training........")
    classes = sorted(list(set(dataset['target_class'])))
    vocabulary = list(dataset.iloc[0:0, 1:-1])
    number_data_points = len(dataset)
    prior = []
    conditional_probabilities = []
    for val in classes:
        dataset_split = dataset[dataset['target_class'] == val]
        prior.append(np.float(len(dataset_split))/np.float(number_data_points))
        col_sums = list(dataset_split.iloc[:, 1:-1].astype(float).sum(axis=0))
        laplace_smooth = ((np.array(col_sums) + 1)) / (np.sum(col_sums) + len(vocabulary))
        conditional_probabilities.append(laplace_smooth)
    conditional_probabilities = np.transpose(conditional_probabilities)
    return classes,vocabulary,prior,conditional_probabilities

def train_bernoulli_naive_bayes(dataset):
    print("Training........")
    classes = sorted(list(set(dataset['target_class'])))
    vocabulary = list(dataset.iloc[0:0, 1:-1])
    number_data_points = len(dataset)
    prior = []
    conditional_probabilities = []
    for val in classes:
        dataset_split = dataset[dataset['target_class'] == val]
        prior.append(np.float(len(dataset_split)) / np.float(number_data_points))
        col_sums = list(dataset_split.iloc[:, 1:-1].astype(float).sum(axis=0))
        laplace_smooth = ((np.array(col_sums) + 1)) / (len(dataset_split) + 2)
        conditional_probabilities.append(laplace_smooth)
    conditional_probabilities = np.transpose(conditional_probabilities)
    return classes, vocabulary, prior, conditional_probabilities




def predict_multinomial(list,classes,vocabulary,prior,conditional_probabilities):
    word_list = list
    word_list = [i for i in word_list if i in vocabulary]
    score = []
    for index_of_c in range(len(classes)):
        c = classes[index_of_c]
        score.append(np.log(prior[index_of_c]))
        for index_of_t in range(len(word_list)):
            t = word_list[index_of_t]
            index_of_t_in_V = vocabulary.index(t)
            score[index_of_c] = score[index_of_c] + np.log(conditional_probabilities[index_of_t_in_V][index_of_c])
        #print(c,score[index_of_c])
    index_prediction = np.argmax(score)
    prediction = classes[index_prediction]
    return prediction


def predict_bernoulli(list,classes,vocabulary,prior,conditional_probabilities):
    word_list = list
    word_list = [i for i in word_list if i in vocabulary]
    score = []
    for index_of_c in range(len(classes)):
        c = classes[index_of_c]
        score.append(np.log(prior[index_of_c]))
        for index_of_t in range(len(vocabulary)):
            t = vocabulary[index_of_t]
            if(t in word_list):
                score[index_of_c] = score[index_of_c] + np.log(conditional_probabilities[index_of_t][index_of_c])
            else:
                score[index_of_c] = score[index_of_c] + np.log((1-conditional_probabilities[index_of_t][index_of_c]))
    index_prediction = np.argmax(score)
    prediction = classes[index_prediction]
    return prediction

def tokenize_words(sentence):
    words = re.sub('[^A-Za-z]+', ' ', sentence)
    re.sub(r'\b\w{1,2}\b', ' ', sentence)
    words = re.sub("[^\w]", " ", words).split()
    cleaned_text = [w.lower() for w in words]#if w not in stop_words]
    return cleaned_text

def get_accuracy(y_test,y_pred):
    cm = classification_report(y_test, y_pred)
    return cm




naive_bayes_type = sys.argv[1]
if(naive_bayes_type == "bag_of_words"):
    data_train = pd.read_csv("../programs/bag_of_words_matrix_train.csv")
    data_test = pd.read_csv("../programs/original_test_data.csv")
    if 'file_names' in data_train.columns:
        del data_train["file_names"]
    if 'Unnamed: 0' in data_train.columns:
        del data_train["Unnamed: 0"]
    if 'predicted' in data_test.columns:
        del data_test["predicted"]

    classes, vocabulary, prior, conditional_probabilities = train_multinomial_naive_bayes(data_train)
    data_test['email_sentences'] = data_test['email_sentences'].map(
        lambda sentence: sorted(list(tokenize_words(sentence))))
    print("Predicting....")
    print(" ")
    data_test['predicted'] = data_test["email_sentences"].map(
        lambda sentence: predict_multinomial(sentence, classes, vocabulary, prior, conditional_probabilities))
    cm = accuracy_score(np.array(data_test['target_class']), np.asarray(data_test['predicted']))
    print("accuracy_score-->", cm)
    print(" ")
    print("Detailed Classificatiion Report")
    print(get_accuracy(data_test['target_class'], data_test['predicted']))

elif(naive_bayes_type == "bernolli"):

    data_train = pd.read_csv("../programs/bernolli_matrix_train.csv")
    data_test = pd.read_csv("../programs/original_test_data.csv")

    if 'file_names' in data_train.columns:
        del data_train["file_names"]
    if 'Unnamed: 0' in data_train.columns:
        del data_train["Unnamed: 0"]
    if 'predicted' in data_test.columns:
        del data_test["predicted"]


    classes, vocabulary, prior, conditional_probabilities = train_bernoulli_naive_bayes(data_train)
    data_test['email_sentences'] = data_test['email_sentences'].map(
        lambda sentence: sorted(list(tokenize_words(sentence))))
    print("Predicting....")
    print(" ")
    data_test['predicted'] = data_test["email_sentences"].map(
        lambda sentence: predict_bernoulli(sentence, classes, vocabulary, prior, conditional_probabilities))
    cm = accuracy_score(np.array(data_test['target_class']), np.asarray(data_test['predicted']))
    print("accuracy_score-->", cm)
    print(" ")
    print("Detailed Classificatiion Report")
    print(get_accuracy(data_test['target_class'], data_test['predicted']))
else:
    print("Enter correct naive_bayes_type")


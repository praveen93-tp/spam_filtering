import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from programs.convert_data_to_dataframe import read_convert_data,concat_vector_dataframe
from programs.vectorizers import bag_of_words_vectorizer,bernoulli_vectorizer
#from programs.naive_bayes import *
#from programs.logistic_regression import *
import time
import numpy as np
import pandas as pd



trainPath = sys.argv[1]
testPath = sys.argv[2]

print("Generating Bernuolli Vector........")
data_train = read_convert_data(trainPath)
data_train.to_csv("original_train_data.csv")
ber_vectors_for_train = bernoulli_vectorizer(data_train)
transformed_data_train = concat_vector_dataframe(data_train,ber_vectors_for_train)
transformed_data_train.to_csv("bernolli_matrix_train.csv")

data_test = read_convert_data(testPath)
data_test.to_csv("original_test_data.csv")
ber_vectors_for_test = bernoulli_vectorizer(data_test)
transformed_data_test = concat_vector_dataframe(data_test,ber_vectors_for_test)
transformed_data_test.to_csv("bernolli_matrix_test.csv")

print("Generating Bag of Word Vector........")
train_data = read_convert_data(trainPath)
bag_vectors_for_train = bag_of_words_vectorizer(train_data)
transformed_data_train_2 = concat_vector_dataframe(train_data,bag_vectors_for_train)
transformed_data_train_2.to_csv("bag_of_words_matrix_train.csv")

test_data = read_convert_data(testPath)
bag_vectors_for_test = bag_of_words_vectorizer(test_data)
transformed_data_test_2 = concat_vector_dataframe(test_data,bag_vectors_for_test)
transformed_data_test_2.to_csv("bag_of_words_matrix_test.csv")




















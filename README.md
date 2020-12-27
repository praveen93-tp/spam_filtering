# spam_filtering

Datasets: 
 https://www.kaggle.com/wcukierski/enron-email-dataset
 

The input folder structure should be as in the screen shots below:
   input
   
    -test
    
       -spam
       
       -ham
       
    -train
    
       -spam
       
       -ham
       
       
Need to give path till “\input\train” or “\input\test”

generate_vectors.py for generating both Bernoulli and Bag_of words vectors. Once this executed excel files will get generated in the current folder. These are train and test vector files for each of both Bernoulli and bag_of_words.

Every time you execute a new dataset. Do re-execute generate_vectors.py before executing other files as it generates new vector file specific to the dataset which is supposed to be used by other algorithms.

Execute as following below

python generate_vectors.py <train_location> <test_location>

python generate_vectors.py "D:\python generate_vectors.py "D:\spam_filtering\input\train" "D:\spam_filtering\input\test"

python naive_bayes.py <bag_of_words or bernolli> please give exact spellings. 

“python naive_bayes.py bag_of_words “ generates results for multinomial naive bayes with bag_of_words 

“python naive_bayes.py bernolli “ generates results for Bernoulli vectors with discrete naive bayes.


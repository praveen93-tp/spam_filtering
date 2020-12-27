import os
import pandas as pd
import numpy as np



def read_convert_data(filepath):
    list_of_classes = os.listdir(filepath)
    dataframe = pd.DataFrame(columns=['file_names', 'email_sentences', 'target_class'])
    for each_class in list_of_classes:
        files = os.listdir(filepath + "//" + each_class)
        #sentences = []
        #list_of_files = []
        #classes_list = []
        for each_file in files:
            current_file = filepath + "//" + each_class + "//" + each_file
            dataframe = dataframe.append({'file_names': each_file, 'email_sentences': convert_to_list(current_file), 'target_class': each_class}, ignore_index=True)
            #sentences.append(convert_to_list(current_file))
            #list_of_files.append(each_file)
            #classes_list.append(each_class)
    return dataframe

def concat_vector_dataframe(dataframe,vectors):
    col = vectors[0]
    vectors = np.delete(vectors, (0), axis=0)
    vectors = pd.DataFrame(vectors, columns=col)
    main_df = pd.concat([dataframe["file_names"], vectors, dataframe["target_class"]], axis=1, ignore_index=True)
    column = []
    column.append("file_names")
    [column.append(x) for x in col]
    column.append("target_class")
    main_df.columns = column
    return main_df

def convert_to_list(current_file_url):
    open_f = open(current_file_url, 'r')
    word_list = str(open_f.read())
    return word_list

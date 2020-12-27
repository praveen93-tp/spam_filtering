import numpy as np
import regex as re

def bag_of_words_vectorizer(dataframe):
     complete_sentences = list(dataframe['email_sentences'])
     vocabulary = sorted(list(set(tokenize_sentences(complete_sentences))))
     vector_matrix = []
     vector_matrix.append(vocabulary)
     for sentence in complete_sentences:
         words = tokenize_words(sentence)
         bag_vector = np.zeros(len(vocabulary))
         for w in words:
             for index, word in enumerate(vocabulary):
                 if word == w:
                     bag_vector[index] += 1
         vector_matrix.append(bag_vector)

     return np.asarray(vector_matrix)


def bernoulli_vectorizer(dataframe_2):
    complete_sentences_2 = list(dataframe_2['email_sentences'])
    vocabulary_2 = sorted(list(set(tokenize_sentences(complete_sentences_2))))
    vector_matrix_2 = []
    vector_matrix_2.append(vocabulary_2)
    for sentence in complete_sentences_2:
        words = tokenize_words(sentence)
        bernoulli_vector = np.zeros(len(vocabulary_2))
        for w in words:
            for index, word in enumerate(vocabulary_2):
                if word == w:
                    bernoulli_vector[index] = 1
        vector_matrix_2.append(bernoulli_vector)
    return np.asarray(vector_matrix_2)



def tokenize_sentences(list_of_sentences):
    words = []
    for sentence in list_of_sentences:
        w = tokenize_words(sentence)
        words.extend(w)
    words = sorted(list(words))
    return words


def tokenize_words(sentence):
    """
    stop_words = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other",
    "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", " and ", "been", "have", " in ", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", " not ", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", " if ", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]
    #words = re.findall(r"[A-Za-z@#]+|\S", sentence)
    """
    words = re.sub('[^A-Za-z]+', ' ', sentence)
    re.sub(r'\b\w{1,2}\b', ' ', sentence)
    words = re.sub("[^\w]", " ", words).split()
    cleaned_text = [w.lower() for w in words]#if w not in stop_words]
    return cleaned_text

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from DataLoader import DataLoader

# evaluating model performance
def computer_metrics(x_train, y_train, x, y):

    TP, TN, FN, FP = 0, 0, 0, 0 #placeholder

    dummyTrain_x, dummyTrain_y, dummyTest_x, true_y = DataLoader.split_data(x, y)

    i = 0
    for i in range(len(y_train)):
        if i in range(len(true_y)):
            if y_train[i] == 1:  # Spam
                if true_y[i] == 1:
                    TP += 1
                else:
                    FP += 1
            else:  # Ham
                if true_y[i] == 0:
                    TN += 1
                else:
                    FN += 1
        else:
            break

    if (TP + TN + FP + FN) == 0:
        Accuracy = 0
    else:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)

    if (TP + FP) == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)

    if (TP + FN) == 0:
        Recall = 0
    else:
        Recall = TP / (TP + FN)

    if (Precision + Recall) == 0:
        F1 = 0
    else:
        F1 = (2 * Precision * Recall) / (Precision + Recall)
    # compute the true and false positives and negatives
    # compute accuracy, precision, reval, and F1 score using the equations
    #   on the handout
    return Accuracy, Precision, Recall, F1

# logging and analysis
    # main function
    # predict training and test samples
    # compute and log performance metrics

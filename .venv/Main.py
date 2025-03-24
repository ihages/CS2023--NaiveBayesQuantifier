# Posterior Probability:
# P(c|x) = (P(x|c)P(c))/P(x)
#        = P(x1|c) * P(x2|c) * ... * P(xi|c) * P(c)
# if P(y=spam|x) > P(y=ham|x):
#   return spam
# else:
#   return ham

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

from EvaluationMetrics import computer_metrics
from NaiveBayes import NBLearn, prediction
from DataLoader import DataLoader

if __name__ == '__main__':
    #constructors
    nbTrain = NBLearn()

    #fileName = (input(str("Enter file name: "))) #absolute path
    fileName = "SMSSpamCollection.txt"
    absolute_path = os.path.abspath(fileName)
    # fileName = r"data/SMSSpamCollection.txt"  # Use raw string or forward slashes

    #funtion calls
    x, y = DataLoader.load_data(absolute_path)
    train_x, train_y, test_x, test_y = DataLoader.split_data(x, y)
    nbTrain.train(train_x, train_y)
    prediction(test_x, test_y)
    accuracy, precision, recall, f1 = computer_metrics(train_x, train_y, x, y)
    print(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}')

# C:\Users\ihage\PycharmProjects\CS2023--NaiveBayesQuantifier\data\SMSSpamCollection.txt
# data\SMSSpamCollection.txt
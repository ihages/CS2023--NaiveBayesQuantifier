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

class NBLearn:
    def __init__(self):
        self.word = ""
        self.num_spam = 0
        self.num_ham = 0
        self.vocabHam = []
        self.vocabSpam = []

        self.spam_count = 0
        self.ham_count = 0
        self.total_words = 0

        self.word_dict = {}

    def update_word_dict(self, word, is_spam):
        if word not in self.word_dict:
            self.word_dict[word]={
                "total_count": 0,
                "ham count": 0,
                "spam count": 0,
                "P(word)": 0,
                "P(word|spam)": 0,
                "P(word|ham)": 0
            }
        if is_spam:
           self.word_dict[word]["spam count"] += 1
           self.spam_count += 1
        else:
            self.word_dict[word]["ham count"] += 1
            self.ham_count += 1

        self.word_dict[word]["total_count"] = self.word_dict[word]["spam count"] + self.word_dict[word]["ham count"]
        self.total_words = self.spam_count + self.ham_count

        if self.total_words > 0:
            self.word_dict[word]["P(word)"] = self.word_dict[word]["total_count"] / self.total_words
        if self.spam_count > 0:
            self.word_dict[word]["P(word|spam)"] = self.word_dict[word]["spam count"] / self.spam_count
        if self.ham_count > 0:
            self.word_dict[word]["P(word|ham)"] = self.word_dict[word]["ham count"] / self.ham_count

    def train(self, x, y): # compute class priors: p(ham) and p(spam)

        message = []
        i=0
        for i in range(len(y)):
            label = y[i]
            if label == 0:
                self.num_ham += 1
                self.vocabHam.append(x[i])
            else:
                self.num_spam += 1
                self.vocabSpam.append(x[i])
            is_spam = label

            message = x[i]
            #categorizing words
            for word in message:
                j = str(word)
                self.update_word_dict(j, is_spam)

            i+=1

        #summing up lists and sizes -- moved to init

# we need to make a table for every word in the email
# import words
# data preprocessing

# implementing naive bayes classifier


def prediction(x, y):
    # Load data for true messages and labels
    predicted_labels = []

    nbTrain = NBLearn()

    # Loop through each message
    for message in x:
        likelihood_spam = 0
        likelihood_ham = 0

        # Loop through each word in the message
        for word in message:
            if word in nbTrain.word_dict:
                # Add log probabilities to avoid underflow
                likelihood_spam += nbTrain.word_dict[word]["P(word|spam)"]
                likelihood_ham += nbTrain.word_dict[word]["P(word|ham)"]

        # Predict based on which likelihood is greater
        if likelihood_spam > likelihood_ham:
            predicted_labels.append(1)  # Spam
        else:
            predicted_labels.append(0)  # Ham

    return predicted_labels

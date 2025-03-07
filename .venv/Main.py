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

class DataLoader: # used dataLoader_demo.py from Canvas to start this
    @staticmethod # what does this mean?
    def preprocess(line):
        text = line
        # convert text to lowercase
        text = text.lower()  # Fix this line

        # remove special chars and punctuation
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
            #https://www.geeksforgeeks.org/python-removing-unwanted-characters-from-string/

        #splits words into list
        words = word_tokenize(text)

        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        #objects
        wnl = WordNetLemmatizer()
        ps = PorterStemmer()
        lemstemwords = []
        for word in words:
            lemstemwords.append(ps.stem(wnl.lemmatize(word)))

        return lemstemwords

    @staticmethod
    def load_data(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                x, y = [], []
                line = file.readline()
                while line:
                    split_xy = line.split('\t')
                    if len(split_xy) >= 2:
                        y.append(split_xy[0])
                        message = split_xy[1]
                        processed_text = DataLoader.preprocess(message)
                        x.append(processed_text)
                    line = file.readline()
            return x, y
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return [], []
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return [], []

    @staticmethod
    def split_data(x, y, test_ratio=0.2):
        # shuffle and split the dataset into training and testing sets
        pass

# we need to make a table for every word in the email
# import words
# data preprocessing

# implementing naive bayes classifier
def train(file_path): # compute class priors: p(ham) and p(spam)
    hamCount, spamCount = 0, 0

    hamWords, spamWords = [], []

    x, y = DataLoader.load_data(file_path)

    # preprocess all messages
    x = [DataLoader.preprocess(message) for message in x]

    #categorizing words
    for i, label in enumerate(y):
        if label == 1:
            spamCount += 1
            spamWords.extend(x[i])  # add words to the spam words list
        elif label == 0:
            hamCount += 1
            hamWords.extend(x[i])  # add words to the ham words list

    #summing up lists and sizes
    fullWordList = [] + spamWords + hamWords

    # removing duplicate words
    word_counts = {}
    for word in fullWordList:
        if word not in word_counts:
            word_counts[word] = {'total': 0, 'ham': 0, 'spam': 0}
        word_counts[word]['total'] += 1
        if word in hamWords:
            word_counts[word]['ham'] += 1
        if word in spamWords:
            word_counts[word]['spam'] += 1

    wordsDict = {}
    for word, counts in word_counts.items():
        probHam = float(100 * counts['ham'] / counts['total'])
        probSpam = float(100 * counts['spam'] / counts['total'])
        likelihood = -1
        if probHam > probSpam:
            likelihood = 0
        elif probHam < probSpam:
            likelihood = 1
        wordsDict[word] = {
            "Occurrences": counts['total'],
            "P(Ham)": probHam,
            "P(Spam)": probSpam,
            "Likeliness": likelihood,
        }
    return wordsDict

def prediction():
    # compute the log probabilities of each message being ham or spam
    # assign the label with the higher probability
    return 1

# evaluating model performance
def computer_metrics():
    TP = 0 #placeholder
    TN = 0 #placeholder
    FP = 0 #placeholder
    FN = 0 #placeholder

    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    Precision = (TP)/(TP+FP)
    Recall = (TP)/(TP+FN)
    F1 = (2*Precision*Recall)/(Precision+Recall)
    # compute the true and false positives and negatives
    # compute accuracty, precision, reval, and F1 score using the equations
    #   on the handout
    return 1

# logging and analysis
    # main funtion
    # predict training and test samples
    # compute and log performance metrics

if __name__ == '__main__':
    fileName = (input(str("Enter file name: "))) #absolute path
    absolute_path = os.path.abspath(fileName)
    # fileName = r"data/SMSSpamCollection.txt"  # Use raw string or forward slashes
    print(absolute_path)
    word_dict = train(absolute_path)
    for i in word_dict:
        print(word_dict[i])
# C:\Users\ihage\PycharmProjects\CS2023--NaiveBayesQuantifier\data\SMSSpamCollection.txt
# data\SMSSpamCollection.txt
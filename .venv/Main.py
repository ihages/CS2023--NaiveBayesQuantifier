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

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class DataLoader: # used dataLoader_demo.py from Canvas to start this
    @staticmethod # what does this mean?
    def preprocess(line):
        text = line
        # convert text to lowercase
        text = text.lower()

        # remove special chars and punctuation
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
            #https://www.geeksforgeeks.org/python-removing-unwanted-characters-from-string/

        #splits words into list
        words = word_tokenize(text)

        stop_words = set(stopwords.words('english'))
        i=0
        for words[i] in words:
            if words[i] in stop_words:
                words.pop(i)
            i+=1
        i=0
        #objects
        wnl = WordNetLemmatizer()
        ps = PorterStemmer()
        lemstemwords = []
        for word in words:
            lemstemwords.append(ps.stem(wnl.lemmatize(word)))
            i+=1

        return lemstemwords

    @staticmethod
    def load_data(file_path):
        # opens a file text and load data
        with open(file_path, 'r') as file:#'r' is read
            x, y = [],[]
            #while loop over lines in text file
                # split into label and text
            line = file.readline()
            while line:
                split_xy = line.split('\t')
                y.append(split_xy[0])
                message = split_xy[1]
                processed_text = DataLoader.preprocess(message)
                x.append(processed_text)
        return x, y

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
    word=0 #instantiate word
    wordCountList=[] #instantiate list
    noDupeList = [] #this is a temp, and we will overwrite fullWordList with the noDupeList
    for fullWordList[word] in fullWordList:
        if word not in fullWordList:
            noDupeList.append(fullWordList[word])
        if word in fullWordList:
            wordCountList[wordCountList.index(word, fullWordList[word])] +=1

#counting the number of occurrences of a word in a text string
    wordD, wordF, wordH, wordS, totalOccurrences, hamOccurrences, spamOccurrences = 0
    for noDupeList[wordD] in noDupeList:
        for fullWordList[wordF] in fullWordList:
            if wordF == wordD:
                totalOccurrences+=1 #this should count how many times each unique word shows up in a line
        for hamWords[wordH] in hamWords:
            if wordH in hamWords:
                hamOccurrences+=1
        for spamWords[wordS] in spamWords:
            if wordS in spamWords:
                spamOccurrences+=1

        probHam = float(100 * hamOccurrences / totalOccurrences)
        probSpam = float(100 * spamOccurrences / totalOccurrences)

        # determines if a word is likely to be spam, ham, or neither
        likelihood = -1 #-1 is for neither
        if probHam > probSpam:
            likelihood = 0
        elif probHam < probSpam:
            likelihood = 1

        # dict holding values for each word
        wordsDict = {
            "Word": wordD,
            "Occurrences": totalOccurrences,
            "P(Ham)": probHam,
            "P(Spam)": probSpam,
            "Likeliness": likelihood,
        }
        return wordsDict

    # count word occurrences separately for ham and spam messages
    # compute probabilities of words given a class p(word|ham), p(word|spam) with Laplace smoothing

    return

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
    #fileName = (input(str("Enter file name: ")))
    fileName = "SMSSpamCollection.txt"
    print(fileName)
    wordsDict = train(fileName)
    print(wordsDict)

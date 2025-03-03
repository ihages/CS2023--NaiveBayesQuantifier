# Posterior Probability:
# P(c|x) = (P(x|c)P(c))/P(x)
#        = P(x1|c) * P(x2|c) * ... * P(xi|c) * P(c)
# if P(y=spam|x) > P(y=ham|x):
#   return spam
# else:
#   return ham

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re


class DataLoader: # used dataLoader_demo.py from Canvas to start this
    @staticmethod # what does this mean?
    def preprocess(text):
        # convert text to lowercase
        text.lower()
        # remove special chars and puntuation
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
            #https://www.geeksforgeeks.org/python-removing-unwanted-characters-from-string/
        # split text into list of words
        word = text.split()
            #word is a list of words
        # tokenize words using word tokenize
        # apply lemmatization
        wordNetLemmatizer.lemmatize(word)
            # group words
        wordNetLemmatizer.stem(word)
            # apply stem to each word
        # remove stopwords
        return word

    @staticmethod
    def load_data(file_path):
        # opens a file text and load data
        file = open(file_path, 'r')   #'r' is read
        x, y = [],[]
        #while loop over lines in text file
            #split into label and text
            #replace label with 0 or 1--append label to y
            #process x
        for line in file:
            y.append(line[0]) #adds first word of line into y[]
            line[0].pop() #removes label from the list
            x.append(DataLoader.preprocess(line))
        return x,y

    @staticmethod
    def split_data(x, y, test_ratio=0.2):
        # shuffle and split the dataset into training and testing sets
        pass

# we need to make a table for every word in the email
# import words
# data preprocessing

# implementing naive bayes classifier
def train(file_path): # compute class priors: p(ham) and p(spam)
    hamCount = 0
    spamCount = 0
    hamWords = []
    spamWords = []

    DataLoader.load_data(file_path)
    x, y = DataLoader.load_data(file_path)

    i = 0
    for x[i] in x:
        DataLoader.preprocess(x)

    #categorizing words
    label = 0
    for y[label] in DataLoader.load_data(file_path):
        if y[label] == 1:
            spamCount += 1
            word = 0
            for x[word] in x: #counts through the words in x
                spamWords.append(x[word]) #add words to the spamwords list
        elif y[label] == 0:
            hamCount += 1
            word = 0
            for x[word] in x:  # counts through the words in x
                hamWords.append(x[word])  # add words to the hamwords list

    #summing up lists and sizes
    fullWordList = [] + spamWords + hamWords

    # removing duplicate words
    word=0 #instantiate word
    wordCountList=[] #instantiate list
    noDupeList = [] #this is a temp and we will overwrite fullWordList with the noDupeList
    for fullWordList[word] in fullWordList:
        if word not in fullWordList:
            noDupeList.append(fullWordList[word])
        if word in fullWordList:
            wordCountList[wordCountList.index(word, fullWordList[word])] +=1

#counting the number of occurrences of a word in a text string
    for dupeList[wordD] in dupeList:
        for fullWordList[wordF] in fullWordList:
            if wordF == wordD:
                totalOccurrences+=1 #this should count how many times each unique word shows up in a line
        for hamWordList[wordH] in hamWords:
            if wordH == hamWord:
                hamOccurrences+=1
        for spamWordList[wordS] in spamWords:
            if wordS == spamWord:
                spamOccurrences+=1

        probHam = float(100 * hamOccurrences / totalOccurrences)
        probSpam = float(100 * spamOccurrences / totalOccurrences)

        # determines if a word is likely to be spam, ham, or neither
        likelyhood = -1 #-1 is for neither
        if probHam > probSpam:
            likelyhood = 0
        elif probHam < probSpam:
            likelyhood = 1

        # dict holding values for each word
        wordsDict[wordD] = {
            "Word": wordD,
            "Occurrences": totalOccurrences,
            "P(Ham)": probHam,
            "P(Spam)": probSpam,
            "Likeliness": likelyhood,
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
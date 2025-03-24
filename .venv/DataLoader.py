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
        #text = line
        # convert text to lowercase
        text = line.lower()

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
            for i in range(len(y)):
                if y[i] == 'ham':
                    y[i] = 0
                elif y[i] == "spam":
                    y[i] = 1
        return x, y

    @staticmethod
    def split_data(x, y, test_ratio=0.8):
        # shuffle and split the dataset into training and testing sets
        split_x = int(len(x)*test_ratio)
        split_y = int(len(y)*test_ratio)
        train_x = x[:split_x]
        train_y = y[:split_y]
        test_x = x[split_x:]
        test_y = y[split_y:]

        return train_x, train_y, test_x, test_y

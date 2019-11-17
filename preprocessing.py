from csv import DictReader
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
from nltk.corpus import stopwords
import string

nltk.download('wordnet')
nltk.download('stopwords')


class DataSet():
    def __init__(self, name="train", path="./FNC-1"):
        self.lemmatizer = WordNetLemmatizer()
        self.stopWords = set(stopwords.words('english'))
        self.data = {'BodyID': [], 'Headline': [], 'Body': [], 'Stance': []}
        self.path = path

        print("Reading dataset")
        bodies = name + "_bodies.csv"
        stances = name + "_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        #make the body ID an integer value
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))

    def read(self, filename):
        rows = []
        with open(self.path + "/" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows

    def preprocess_text(self, text, lemmatize=True, remove_stop=True, remove_punc=False):
        text = text.lower()
        if remove_punc:
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator)
        # Now do magic stuff
        text_tokens = nltk.word_tokenize(text)
        if lemmatize:
            text_tokens = [self.lemmatizer.lemmatize(w) for w in text_tokens]
        if remove_stop:
            text_tokens = [word for word in text_tokens if word not in self.stopWords]
        return text_tokens

    def preprocess(self, lemmatize=True, remove_stop=True, remove_punc=False):

        for bodyID, body in self.articles.items():
            self.articles[bodyID] = self.preprocess_text(body, lemmatize, remove_stop, remove_punc)

        for i, stance in enumerate(self.stances):
            bodyID = int(stance['Body ID'])
            stance_label = stance['Stance'].lower()
            body = self.articles[bodyID]
            headline = self.preprocess_text(stance['Headline'], lemmatize, remove_stop, remove_punc)
            self.data['BodyID'].append(bodyID)
            self.data['Headline'].append(headline)
            self.data['Body'].append(body)
            self.data['Stance'].append(stance_label)
        return pd.DataFrame(self.data)

    def get_labels(self):
        return [stance['Stance'].lower() for stance in self.stances]

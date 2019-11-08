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
        self.path = path

        print("Reading dataset")
        bodies = name + "_bodies.csv"
        stances = name + "_stances.csv"

        self.old_stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()
        self.headlines = dict()
        self.stances = dict()

        # make the body ID an integer value
        for s in self.old_stances:
            s['Body ID'] = int(s['Body ID'])

        # copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        # get the actual stance and stuff you dummy
        for s in self.old_stances:
            self.headlines[int(s['Body ID'])] = s['Headline']
            self.stances[int(s['Body ID'])] = s['Stance']

        print("Total stances: " + str(len(self.old_stances)))
        print("Total bodies: " + str(len(self.articles)))

    def read(self, filename):
        rows = []
        with open(self.path + "/" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows

    def preprocess(self, lemmatize=True, removeStop=True, removePunc=False):
        lemmatizer = WordNetLemmatizer()
        data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
        stopWords = set(stopwords.words('english'))

        data = {'ID': [], 'Headline': [], 'Body': [], 'Stance': []}

        # self.headlines_lemmatized = pd.
        for bodyID in self.articles.keys():
            stance = self.stances[bodyID].lower()
            body = self.articles[bodyID].lower()
            headline = self.headlines[bodyID].lower()

            if removePunc:
                body = body.translate(None, string.punctuation)
                headline = headline.translate(None, string.punctuation)

            # Now do magic stuff
            headline_tokens = nltk.word_tokenize(headline)
            body_tokens = nltk.word_tokenize(body)
            if lemmatize:
                headline_tokens = [lemmatizer.lemmatize(w) for w in headline_tokens]
                body_tokens = [lemmatizer.lemmatize(w) for w in body_tokens]

            if removeStop:
                data['Headline'].append([word for word in headline_tokens if word not in stopWords])
                data['Body'].append([word for word in body_tokens if word not in stopWords])
            else:
                data['Headline'].append(headline_tokens)
                data['Body'].append(body_tokens)

            data['Stance'].append(stance)

            data['ID'].append(bodyID)

        return pd.DataFrame(data)

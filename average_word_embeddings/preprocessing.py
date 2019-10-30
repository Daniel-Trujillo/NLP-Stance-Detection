from csv import DictReader
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd

nltk.download('wordnet')
  
class DataSet():
    def __init__(self, name="train", path="./FNC-1"):
        self.path = path

        print("Reading dataset")
        bodies = name+"_bodies.csv"
        stances = name+"_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()
        self.headlines = dict()

        #make the body ID an integer value
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        for s in self.stances:
            self.headlines[int(s['Body ID'])] = s['Headline']

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))


    def read(self,filename):
        rows = []
        with open(self.path + "/" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows

    
    def preprocess(self):
        lemmatizer = WordNetLemmatizer()

        data = {'ID': [], 'Headline': [], 'Body': [], 'Stance': []}

        #self.headlines_lemmatized = pd.
        for bodyID in self.articles.keys():
            stance = self.stances[bodyID]
            body = self.articles[bodyID]
            headline = self.headlines[bodyID]

            # Now do magic stuff
            headline_tokens = nltk.word_tokenize(headline)
            headline_lemmatized = [lemmatizer.lemmatize(w) for w in headline_tokens]
            data['Headline'].append(headline_lemmatized)

            body_tokens = nltk.word_tokenize(body)
            body_lemmatized = [lemmatizer.lemmatize(w) for w in body_tokens]
            data['Body'].append(body_lemmatized)

            data['Stance'].append(stance)

            data['ID'].append(bodyID)

        df = pd.DataFrame(data)

ds = DataSet()

import numpy as np
from preprocessing import DataSet
from sklearn.feature_extraction.text import TfidfVectorizer
ds = DataSet(path="../FNC-1/")
data = ds.preprocess(True, True, True)

def get_ngram(n, text):
    ngrams = {}
    for i in range(n, len(text)+1):
        ngram_words = tuple(text[i-n:i])
        if ngram_words in ngrams.keys():
            ngrams[ngram_words] += 1
        else:
            ngrams[ngram_words] = 1
    return ngrams


def getIDF(tokens):
    vectorizer = TfidfVectorizer(
        use_idf=True,
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        ngram_range=(1, 5)
    )
    vectorizer.fit_transform(tokens)
    idf = vectorizer.idf_
    return dict(zip(vectorizer.get_feature_names(), idf))


def nGramMathing():
    idf = getIDF(data["Body"].to_numpy())
    features = []
    for index, row in data.iterrows():
        H = []
        A = []
        for n in range(1, 6):
            H_ngram = get_ngram(n, row['Headline'])
            A_ngram = get_ngram(n, row["Body"])
            H.extend(list(H_ngram.keys()))
            A.extend(list(A_ngram.keys()))
        sum = 0
        for i, h in enumerate(H):
            TF_hi = (H.count(h) + A.count(h)) * len(h)
            idf_hi = idf.get(" ".join(h), 0)
            sum += (TF_hi * idf_hi)
        sc = sum / (len(H) + len(A))
        features.append(sc)
    return np.array(features).reshape(-1, 1)

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def preprocess_data(text_series):
    try:
        nltk.data.find("stopwords")
    except LookupError:
        nltk.download("stopwords")

    STOPWORDS = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    text_series = text_series.str.replace(r"\W", " ", regex=True)
    text_series = text_series.str.lower()
    text_series = text_series.apply(
        lambda x: " ".join([word for word in x.split() if word not in STOPWORDS])
    )
    text_series = text_series.apply(
        lambda x: " ".join([stemmer.stem(word) for word in x.split()])
    )
    text_series = text_series.str.strip()
    return text_series

#print(preprocess_data(["This is a test. I wonder what text will be returned .... ..", "Does this count as a series?"]))
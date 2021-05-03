import numpy as np
import pandas as pd
import json
import string
import unidecode
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim.corpora as corpora
import gensim
from gensim import models
import spacy
from nltk.stem.snowball import SnowballStemmer
import progressbar
import dask.dataframe as dd


class Extract:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.stemmer = SnowballStemmer(language='english')
        self.tc = nltk.classify.textcat.TextCat()
    def decontracted(self, phrase):
        """Decontract english form
        """
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)

        # Other
        phrase = re.sub(r" gf ", " girlfriend ", phrase)

        return phrase

    def clean_text(self, data):
        """Clean text
        """

        # Lower case
        data = data.lower()

        # Remove accents
        data = unidecode.unidecode(data)

        # Expand english contractions
        data = self.decontracted(data)

        # Replace punctuation by white space
        translator = str.maketrans(
            string.punctuation, ' '*len(string.punctuation))
        data = data.translate(translator)

        # Remove caracters with digits
        table_ = str.maketrans('', '', string.digits)
        data = data.translate(table_)

        # Lemmatizer
        allowed_tags = ["NOUN", "ADJ", "VERB", "ADV"]

        doc = self.nlp(data)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_tags:
                #lem = self.stemmer.stem(token.lemma_)
                new_text.append(token.lemma_)


        # Stop words
        stop = stopwords.words('english')
        data = [x for x in new_text if x not in stop]

        # join all
        data = " ".join(data)
        return data

    def clean_yelp(self, data_dir, output_file):
        input_business = data_dir + 'yelp_academic_dataset_business.json'
        output_business = data_dir + 'temp_business.csv'
        keep_cols = ['business_id', 'categories']
        header = True
        chunksize=500
        max_value = int(sum(1 for row in open(input_business,'r')) / chunksize)
        bar = progressbar.ProgressBar(max_value=max_value)
        i = 0
        for chunk in pd.read_json(input_business, chunksize=chunksize, orient='records', lines=True):
            bar.update(i)
            chunk = chunk[keep_cols]
            # Save to file
            if header:
                chunk.to_csv(output_business, index=False)
            else:
                chunk.to_csv(output_business, mode='a', header=False, index=False)
            header = False
            i += 1

        input_review = data_dir + 'yelp_academic_dataset_review.json'
        output_review = data_dir + 'temp_review.csv'
        keep_cols = ['text', 'stars', 'business_id']
        header = True
        chunksize=500
        max_value = int(sum(1 for row in open(input_review,'r')) / chunksize)
        bar = progressbar.ProgressBar(max_value=max_value)
        for chunk in pd.read_json(input_review, chunksize=chunksize, orient='records', lines=True):
            chunk = chunk[keep_cols]
            chunk = chunk.replace('\n', ' ', regex=True)
            chunk = chunk.replace('\r', ' ', regex=True)
            bar.update(i)
            # Save to file
            if header:
                chunk.to_csv(output_review, index=False)
            else:
                chunk.to_csv(output_review, mode='a', header=False, index=False)
            header = False
            i += 1
        self.concat(data_dir, output_file)

    def concat(self, data_dir, output_file):

        # Companies
        df1 = dd.read_csv(data_dir + 'temp_business.csv')
        df1 = df1.dropna(subset=['categories'])
        # Only restaurants
        df1 = df1[df1['categories'].str.contains(r'Restaurants')]

        # Review
        df2 = dd.read_csv(data_dir + 'temp_review.csv')
        # Only bad review
        df2 = df2[df2['stars'] < 3]

        # Merge dataframes
        df = df1.merge(df2, how='inner', on='business_id')

        # Keep only useful columns
        df = df[['text', 'stars']]
        df.to_csv(data_dir + output_file, index=False, single_file=True)

    def detect_lang(self, text):
        return self.tc.guess_language(text)

    def clean_file(self, input_file, output_file, chunksize=500):
        header = True
        max_value = int(sum(1 for row in open(input_file, 'r')) / chunksize)
        bar = progressbar.ProgressBar(max_value=max_value)
        i = 0
        for chunk in pd.read_csv(input_file, chunksize=chunksize):
            bar.update(i)
            # Clean text
            chunk["clean_text"] = chunk["text"].apply(
                lambda x: self.clean_text(x))

            chunk["bad_review"] = chunk["stars"].apply(
                lambda x: 0 if x > 2 else 1)

            # Save to file
            if header:
                chunk.to_csv(output_file, index=False)
            else:
                chunk.to_csv(output_file, mode='a', header=False, index=False)

            header = False
            i += 1

    def load_dataset(self, filename, size=0):
        df = pd.read_csv(filename)
        df = df.dropna()
        if size:
            return df.sample(size)
        return df


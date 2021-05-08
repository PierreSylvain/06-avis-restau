import pandas as pd
import string
import unidecode
import re
import nltk
from nltk.corpus import stopwords
import spacy
from nltk.stem.snowball import SnowballStemmer
import progressbar
import dask.dataframe as dd


class Extract:
    """Extract data from Yelp dataset and clean text
    """

    def __init__(self):
        """Init class
        :param self:
        """
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.stemmer = SnowballStemmer(language='english')
        self.tc = nltk.classify.textcat.TextCat()

    def decontracted(self, phrase):
        """Decontract english form

        :param self:
        :param phrase: phrase to be decontracted

        :return: string
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
        """Clean text by applying these filters
         - Lower case
         - Remove accents
         - Expand english contractions
         - Replace punctuation by white space
         - Remove characters with digits
         - Stemmization
         - Stop words

         :parameter self:
         :parameter data: English text to clean

         :return: String
        """

        # Lower case
        data = data.lower()

        # Remove accents
        data = unidecode.unidecode(data)

        # Expand english contractions
        data = self.decontracted(data)

        # Replace punctuation by white space
        translator = str.maketrans(
            string.punctuation, ' ' * len(string.punctuation))
        data = data.translate(translator)

        # Remove characters with digits
        table_ = str.maketrans('', '', string.digits)
        data = data.translate(table_)

        words = data.split(' ')
        new_text = []
        for token in words:
            stem = self.stemmer.stem(token)
            if token != "":
                new_text.append(stem)

        # Stop words
        stop = stopwords.words('english')
        data = [x for x in new_text if x not in stop]

        # join all
        data = " ".join(data)
        return data

    def clean_yelp(self, data_dir, output_file):
        """Read from Yelp file and convert JSON dataset to CSV dataset

         :parameter self:
         :param data_dir: Directory for the input dataset
         :param output_file: filename for CSV file output

         :return: void
        """
        input_business = data_dir + 'yelp_academic_dataset_business.json'
        output_business = data_dir + 'temp_business.csv'
        keep_cols = ['business_id', 'categories']
        header = True
        chunk_size = 500
        max_value = int(sum(1 for row in open(input_business, 'r')) / chunk_size)
        bar = progressbar.ProgressBar(max_value=max_value)
        i = 0
        for chunk in pd.read_json(input_business, chunksize=chunk_size, orient='records', lines=True):
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
        chunk_size = 500
        max_value = int(sum(1 for row in open(input_review, 'r')) / chunk_size)
        bar = progressbar.ProgressBar(max_value=max_value)
        for chunk in pd.read_json(input_review, chunksize=chunk_size, orient='records', lines=True):
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
        """Concat the 2 datasets
        select stars < 3
        and keep only text and stars columns

        :param self:
        :param data_dir: Directory for the input dataset
        :param output_file: filename for CSV file output

        :return: void
        """
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

    def clean_file(self, input_file, output_file, chunk_size=500):
        """
        :param self:
        :param input_file: CSV file where to extract data
        :param output_file: output filename for new CSV
        :param chunk_size: Size of chunk to read all the datas

        :return: void
        """
        header = True
        max_value = int(sum(1 for row in open(input_file, 'r')) / chunk_size)
        bar = progressbar.ProgressBar(max_value=max_value)
        i = 0
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
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
        """
        Load a dataset, with the possibility to limit number of rows.

        :param filename: Data filename to load
        :param size: Number of row to return (if empty all rows)

        :return: dataframe
        """
        df = pd.read_csv(filename)
        df = df.dropna()
        if size:
            return df.sample(size)

        return df

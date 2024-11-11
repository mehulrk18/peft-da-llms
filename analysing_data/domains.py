# Imports
import re
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    TfidfTransformer,
    CountVectorizer,
    TfidfVectorizer,
)
from rank_bm25 import BM25Okapi
from nltk.stem import WordNetLemmatizer


# Definitions
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


class Domain:
    """
    This class represents a dataset of text documents in a single domain.
    @field name - a string describing the domain.
    @field documents - a list of lists of words corresponding to each text in the domain.
    @field client - OpenAI client
    @field vocab - a set of all words (containing no special characters, no stop words, and lemmatized) in this domain
    @field sentence_embeddings - list of sentence embeddings in each text
    @field word_count - total word count
    @field token_frequencies - dictionary of tokens to their relative frequency in domain
    @field vocab_size - size of unique vocabulary
    @field dataset_size - number of texts in this domain corpus
    @field tfidf, tfidf_X - TFIDF vectorizer and corresponding vectorized array of documents
    """

    def __init__(self, file_paths: list, file_names: str, client=None):
        """
        Create a Domain.
        @param file_name - a string describing the domain.
        @param file_paths - a list of string paths of the json files to be used to construct the domains.
        @param client - OpenAI client
        """
        self.name = file_names
        self.vocab = set()
        self.client = client
        self.documents, self.sentence_embeddings = self.download(file_paths)
        self.word_count = self.compute_total_word_count()
        self.token_frequencies = self.compute_token_frequencies()
        self.vocab_size = len(self.vocab)
        self.dataset_size = len(self.documents)
        self.tfidf, self.tfidf_X = self.compute_tfidf()

    def __str__(self):
        return f"""
                    ------- Domain Summary ------- \n
                    Dataset: {self.name}\n
                    Number of Documents: {self.dataset_size}\n
                    Vocabulary Size: {self.vocab_size}\n
                    Total Word Count: {self.word_count}\n"""

    def download(self, data) -> tuple:
        """ "
        Processes source and target json files to obtain processed source domain text lists and target domain text datasets.
        @param self - the current Domain datasets to constructs.
        @param file_paths - a list of string paths of the json files used to construct the domains.
        @returns - a list of lists of words corresponding to each text in the domain corpus and a list of the corresponding sentence embeddings.
        """

        domain_words = []
        sentence_embeddings = []
        print(f"Processing {self.name} data.")
        for i in range(len(data)):
            document = self.process_text(data[i])
            domain_words.append(document)
            article = data[i]
            if len(article.split()) > 4000:
                article =  ' '.join(article[:4000])
            if self.client:
                sentence_embeddings.append(self.compute_sentence_embeddings(article) )
        return domain_words, sentence_embeddings

    def compute_total_word_count(self) -> int:
        """
        Compute word count in entire corpus.
        @param self - this Domain class.
        """
        return sum([len(text) for text in self.documents])

    def compute_token_frequencies(self) -> dict:
        """
        Compute relative token frequencies of across entire corpus.
        @param self - this Domain class.
        @returns - a dictionary of tokens mapped to frequencies.
        """
        frequencies = {}
        for text in self.documents:
            for word in text:
                frequencies[word] = frequencies.get(word, 0) + 1 / self.word_count
        return frequencies

    def process_text(self, text: str) -> list:
        """
        Given string of text, remove special characters, capital letters, stopwords, lemmatize, and returns a list of words in order.
        Adds all processed words to vocabulary set.
        @param text - string containing any characters
        @return - list of filtered words
        """
        processed = [
            re.sub(r"[^a-zA-Z0-9\s]", "", string).lower() for string in text.split()
        ]
        processed = list(filter(lambda word: word not in STOP_WORDS, processed))
        processed = [lemmatizer.lemmatize(word) for word in processed]
        while "" in processed:
            processed.remove("")
        for word in processed:
            self.vocab.add(word)
        return processed

    def compute_tfidf(self):
        """
        Computes TFIDF for domain corpus.
        @param self - this Domain class.
        @returns vectorizer, array - corresponding TFIDF vectorizor and vectorized array of documents
        """
        documents = [" ".join(text) for text in self.documents]
        vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, stop_words=None)
        X = vectorizer.fit_transform(documents)
        return vectorizer, X

    def compute_sentence_embeddings(self, text):
        """
        Convert sentences to OpenAI embeddings vectors.
        @param self - this Domain class.
        @returns array - vector embedding
        """
        response = (
            self.client.embeddings.create(
                model="text-embedding-3-small", input=text, dimensions=32
            )
            .data[0]
            .embedding
        )
        return response
# Imports
import os
import json
import random
import argparse
import time
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import numpy as np
import math
from .domains import Domain

from decimal import *
from sklearn.metrics.pairwise import cosine_similarity
import sys


class Similarity:
    """
    A class representing metrics between two Domains.
    @field source - a Domain class describing source domain.
    @field target - a Domain class describing target domain.
    @field word_overlap - a float describing cross-domain overlap as found in Word Matters (Section 3.2).
    @field vocab_overlap - a float describing the domain-relevant vocabulary overlap.
    @field weighted_word_overlap - a float describing TF-IDF weighted word overlap.
    @field sentence_overlap - a float describing sentence embedding overlap
    @field renyi_divergence - a float describing the Renyi divergence between the source and target domain.
    @field kl_divergence - a float describing the KL divergence between the source and target domain.
    @field js_divergence - a float describing the JS divergence between the source and target domain.
    # TODO: Implement PAD, term familiarity clustering
    """

    def __init__(self, source: Domain, target: Domain, client=None):
        """
        @param source - a Domain class
        @param target - a Domain class
        """
        self.source = source
        self.target = target
        self.client = client
        self.sentence_overlap = self.compute_sentence_similarity()

        # self.renyi_divergence = compute_renyi_divergence(
        #     self.target.token_frequencies, self.source.token_frequencies
        # )
        self.kl_divergence = compute_kl_divergence(
            self.target.token_frequencies, self.source.token_frequencies
        )
        # self.js_divergence = compute_js_divergence(
        #     self.target.token_frequencies, self.source.token_frequencies
        # )
        self.vocab_overlap = self.compute_vocab_overlap()
        self.word_overlap = self.compute_word_overlap()
        self.relevance_overlap = self.compute_relevance_overlap()

    def __str__(self):
        return f"""
                    ------- Similarity Summary ------- \n
                    Source Domain: {self.source.name}\n
                    Target Domain: {self.target.name}\n 
                    Word Overlap: {self.word_overlap}\n
                    Relevant Vocab Overlap:  {self.vocab_overlap}\n
                    KL Divergence: {self.kl_divergence}\n
                """
                    # Renyi Divergence: {self.renyi_divergence}\n
                    # # JS Divergence: {self.js_divergence}\n

    def get_metrics(self):
        """
        Returns all task-agnostic metrics across source and target domains.
        @param self - this Similarity class
        @return - an array of task-agnostic metrics.
        """
        return [
            # self.js_divergence,
            self.kl_divergence,
            # self.renyi_divergence,
            self.word_overlap,
            self.vocab_overlap,
            self.relevance_overlap,
            self.sentence_overlap
        ]

    def compute_word_overlap(self) -> float:
        """
        Computes general word overlap of source and target domains as described in Word Matters (Section 3.2).
        @param self - a Similarity class
        @return - a float between [0.0, 1.0] describing word overlap
        """
        gamma = 0
        for i in range(self.source.dataset_size):
            Si = self.source.documents[i]
            for j in range(self.target.dataset_size):
                Tj_unique = set(self.target.documents[j])
                # Compute number of occurrences of each unique word in Si
                for word in Tj_unique:
                    Si_size = 1 if len(Si) == 0 else len(Si)
                    gamma += Si.count(word) / Si_size
        return gamma / self.source.dataset_size / self.target.dataset_size

    def compute_vocab_overlap(self, threshold=0.1) -> float:
        """
        Computes relevant (using TF-IDF) domain vocabulary overlap over source and target domain vocabularies by getting tokens with TF-IDF higher than a given threshold.
        @param self - a Similarity class
        @param threshold - float (minimum TF-IDF threshold to be considered in top vocabulary)
        @return - float of ratio of overlap between most relevant vocabulary over whole target vocabulary size
        """
        source_top, source_dict = get_tfidf_dict(
            self.source.tfidf.get_feature_names_out(), self.source.tfidf_X, threshold
        )
        target_top, target_dict = get_tfidf_dict(
            self.target.tfidf.get_feature_names_out(), self.target.tfidf_X, threshold
        )
        overlap = source_top.keys() & target_top.keys()
        return len(overlap) / len(target_dict)

    def compute_relevance_overlap(self) -> float:
        """
        Computes TF-IDF weighted word overlap of source and target domains (proposed, possible metric).
        @param self - Domain class
        @returns - a float between [0.0, 1.0] describing word overlap weighted by relevance.
        """

        gamma = 0
        tfidf_by_source = []
        vectorizer = self.source.tfidf
        for i in range(self.source.dataset_size):
            Si = self.source.documents[i]
            X = vectorizer.transform([" ".join(Si)])
            allWords = vectorizer.get_feature_names_out()
            values = np.array(X.todense().tolist()).T.sum(axis=1)  # .to_numpy()
            dictionary = dict()
            for k, v in zip(allWords, values):
                dictionary[k] = v
            tfidf_by_source.append(dictionary)
        for i in range(self.source.dataset_size):
            Si = self.source.documents[i]
            for j in range(self.target.dataset_size):
                Tj_unique = set(self.target.documents[j])
                for word in Tj_unique:
                    try:
                        weight = tfidf_by_source[i][word]
                    except:
                        weight = 0
                    if len(Si) == 0:
                        continue
                    gamma += (
                        Si.count(word)
                        * (1 / (1 - weight + sys.float_info.min) ** 2)
                        / len(Si)
                    )
        return gamma / self.source.dataset_size / self.target.dataset_size

    def compute_sentence_similarity(self) -> float:
        """
        Computes sentence overlap of source and target domains using OpenAI Embeddings and cosine similarity.
        @param self - a Similarity class
        @return - a float between [0.0, 1.0] describing sentence overlap
        """
        gamma = 0
        if self.client:
            for i in range(self.source.dataset_size):
                Si = self.source.sentence_embeddings[i]
                for j in range(self.target.dataset_size):
                    Tj = self.target.sentence_embeddings[j]
                    # Compute number if similar sentences
                    count = np.sum(is_similar(Tj, Si))
                    gamma += count / len(Si) / len(Tj)
        return gamma / self.source.dataset_size / self.target.dataset_size


"""
Helper Functions
"""


def compute_renyi_divergence(P: dict, Q: dict, alpha=0.01) -> float:
    """
    Computes Renyi divergence between two domains. Uses relative token frequency to create two distributions.
    @param P - a dictionary of tokens in domain P mapped to their relative frequencies.
    @param Q - a dictionary of tokens in domain Q mapped to their relative frequencies.
    @param alpha - 0 < alpha < ∞ and alpha != 1
    @return - a float describing the Renyi divergence of the target domain (P) from the source domain (Q)
    """
    summation = 0
    all_tokens = set.union(set(P.keys()), set(Q.keys()))
    for token in all_tokens:
        # Smoothing
        p = P.get(token, sys.float_info.min)
        q = Q.get(token, sys.float_info.min)
        summation += (p ** (1 - alpha)) * (q**alpha)
    divergence = math.log(summation) / (alpha - 1)
    return divergence


def compute_kl_divergence(P: dict, Q: dict) -> float:
    """
    Computes KL divergence between two domains (Renyi with alpha = 1). Uses relative token frequency to create two distributions.
    @param P - a dictionary of tokens in domain P mapped to their relative frequencies.
    @param Q - a dictionary of tokens in domain Q mapped to their relative frequencies.
    @return - a float describing the KL divergence of the target domain (P) from the source domain (Q)
    """
    all_tokens = set.union(set(P.keys()), set(Q.keys()))
    divergence = 0
    for token in all_tokens:
        # Smoothing
        p = P.get(token, sys.float_info.min)
        q = Q.get(token, sys.float_info.min)
        divergence += p * math.log(p / q)
    return divergence


def compute_js_divergence(P: dict, Q: dict) -> float:
    """
    Computes Jenson-Shannon divergence between two domains (Renyi with alpha = 1). Uses relative token frequency to create two distributions.
    @param P - a dictionary of tokens in domain P mapped to their relative frequencies.
    @param Q - a dictionary of tokens in domain Q mapped to their relative frequencies.
    @return - a float describing the KL divergence of the target domain (P) from the source domain (Q)
    """
    M = dict()
    for token in set.union(set(P.keys()), set(Q.keys())):
        M[token] = 0.5 * (
            P.get(token, sys.float_info.min) + Q.get(token, sys.float_info.min)
        )
    return 0.5 * compute_kl_divergence(P, M) + 0.5 * compute_kl_divergence(Q, M)


def get_tfidf_dict(tokens: list, X, threshold: float) -> tuple[dict, dict]:
    """
    Maps tokens (relevance > threshold) to their corresponding TF-IDF values.
    @param tokens - list of string tokens
    @param X - vectorized corpus from the corresponding domain TF-IDF vectorizer
    @param threshold - minimum TF-IDF threshold to be considered in top vocabulary
    @return - (dictionary of {tokens: TF-IDF values > threshold}, dictionary of all {tokens:TF-IDF values})
    """
    tfidf_X = np.array(X.todense().tolist()).T.sum(axis=1)  # .to_numpy()
    dictionary = dict()
    for k, v in zip(tokens, tfidf_X):
        if v > threshold:
            dictionary[k] = v
    top = dict(
        sorted(
            filter(lambda word: word[1] > threshold, dictionary.items()),
            key=lambda word: word[1],
            reverse=True,
        )
    )
    return top, dictionary


def is_similar(A: list, B: list, threshold=0.25) -> bool:
    """
    Filters sentence between two documents by cosine similarity > threshold.
    @param A - list of sentence strings from a text.
    @param B - list of sentence strings from a text.
    @param threshold - float between [0, 1.0] representing filtering threshold.
    """
    return cosine_similarity([A],[B]) > threshold
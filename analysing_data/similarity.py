# Imports
import os
import json
import random
import argparse
import time
from functools import lru_cache

from tqdm import tqdm
from rank_bm25 import BM25Okapi
import numpy as np
import scipy
import math
from .domains import DomainsDataset
from scipy.spatial.distance import jensenshannon

from decimal import *
from sklearn.metrics.pairwise import cosine_similarity
import sys


class Similarity:
    # """
    # A class representing metrics between two Domains.
    # @field source - a Domain class describing source domain.
    # @field target - a Domain class describing target domain.
    # @field word_overlap - a float describing cross-domain overlap as found in Word Matters (Section 3.2).
    # @field vocab_overlap - a float describing the domain-relevant vocabulary overlap.
    # @field weighted_word_overlap - a float describing TF-IDF weighted word overlap.
    # @field sentence_overlap - a float describing sentence embedding overlap
    # @field renyi_divergence - a float describing the Renyi divergence between the source and target domain.
    # @field kl_divergence - a float describing the KL divergence between the source and target domain.
    # @field js_divergence - a float describing the JS divergence between the source and target domain.
    # # TODO: Implement PAD, term familiarity clustering
    # """

    """
    A class representing metrics between two Domains.
    @field source - a Domain class describing source domain.
    @field target - a Domain class describing target domain.
    @field vocab_overlap - a float describing the domain-relevant vocabulary overlap.
    @field weighted_word_overlap - a float describing TF-IDF weighted word overlap.
    @field contextual_overlap - a float describing sentence embedding overlap
    @field shannon_entrophy- a float describing the Renyi divergence between the source and target domain.
    @field kl_divergence - a float describing the KL divergence between the source and target domain.
    @field js_divergence - a float describing the JS divergence between the source and target domain.
    # TODO: Implement PAD, term familiarity clustering
    """

    # def __init__(self, source: Domain, target: Domain, client=None):
    #     """
    #     @param source - a Domain class
    #     @param target - a Domain class
    #     """
    #     self.source = source
    #     self.target = target
    #     self.client = client
    #     self.sentence_overlap = self.compute_sentence_similarity()
    #
    #     # self.renyi_divergence = compute_renyi_divergence(
    #     #     self.target.token_frequencies, self.source.token_frequencies
    #     # )
    #     self.kl_divergence = compute_kl_divergence(
    #         self.target.token_frequencies, self.source.token_frequencies
    #     )
    #     # self.js_divergence = compute_js_divergence(
    #     #     self.target.token_frequencies, self.source.token_frequencies
    #     # )
    #     self.vocab_overlap = self.compute_vocab_overlap()
    #     self.word_overlap = self.compute_word_overlap()
    #     self.relevance_overlap = self.compute_relevance_overlap()

    @lru_cache(maxsize=128)
    def __init__(self, source: DomainsDataset, target: DomainsDataset, client=None, ttl_hash=None):
        """
        @param source - a Domain class
        @param target - a Domain class
        """
        self.source = source
        self.target = target
        self.client = client
        self.contextual_overlap = self.compute_contextual_similarity()
        self.s_updated_prob_dist, self.t_updated_prob_dist = self.get_global_prob_distribution()
        self.kl_divergence = self.compute_kl_divergence()
        # self.js_divergence = self.compute_js_divergence()
        self.vocab_overlap = self.compute_vocab_overlap()
        self.tf_idf_overlap = self.compute_relevance_overlap()

    def __str__(self):
        # return f"""
        #             ------- Similarity Summary ------- \n
        #             Source Domain: {self.source.name}\n
        #             Target Domain: {self.target.name}\n
        #             Word Overlap: {self.word_overlap}\n
        #             Relevant Vocab Overlap:  {self.vocab_overlap}\n
        #             KL Divergence: {self.kl_divergence}\n
        #         """
        return f"""
                ------- Similarity Summary ------- \n
                Source Domain: {self.source.name}\n
                Target Domain: {self.target.name}\n 
                TFIDF-weighted Overlap: {self.tf_idf_overlap}\n
                Relevant Vocab Overlap:  {self.vocab_overlap}\n
                KL Divergence: {self.kl_divergence}\n
                Source Entrophy: {self.source.shannon_entropy}\n
                Target Entrophy: {self.target.shannon_entropy}\n
                Contextual Overlap: {self.contextual_overlap}\n
    
            """
        # JS Divergence: {self.js_divergence}\n

    def get_metrics(self):
        """
        Returns all task-agnostic metrics across source and target domains.
        @param self - this Similarity class
        @return - an array of task-agnostic metrics.
        """
        return [
            self.tf_idf_overlap,
            self.vocab_overlap,
            self.kl_divergence,
            self.source.shannon_entropy,
            self.target.shannon_entropy,
            self.contextual_overlap,
        ]
        # self.js_divergence,

        # return [
        #     # self.js_divergence,
        #     self.kl_divergence,
        #     # self.renyi_divergence,
        #     self.word_overlap,
        #     self.vocab_overlap,
        #     self.relevance_overlap,
        #     self.sentence_overlap
        # ]

    # def compute_word_overlap(self) -> float:
    #     """
    #     Computes general word overlap of source and target domains as described in Word Matters (Section 3.2).
    #     @param self - a Similarity class
    #     @return - a float between [0.0, 1.0] describing word overlap
    #     """
    #     gamma = 0
    #     for i in range(self.source.dataset_size):
    #         Si = self.source.documents[i]
    #         for j in range(self.target.dataset_size):
    #             Tj_unique = set(self.target.documents[j])
    #             # Compute number of occurrences of each unique word in Si
    #             for word in Tj_unique:
    #                 Si_size = 1 if len(Si) == 0 else len(Si)
    #                 gamma += Si.count(word) / Si_size
    #     return gamma / self.source.dataset_size / self.target.dataset_size
    def compute_vocab_overlap(self, ) -> float:
        vocab1 = set(self.source.domain_words)
        vocab2 = set(self.target.domain_words)

        # Calculate precision
        overlap = len(vocab1.intersection(vocab2))
        # Calculate percentage overlap of source (high-resource) with respect to target (low-resource)
        if len(vocab2) > 0:
            overlap_percentage = (overlap / len(vocab2)) * 100
        else:
            overlap_percentage = 0

        return overlap_percentage

    # def compute_vocab_overlap(self, threshold=0.1) -> float:
    #     """
    #     Computes relevant (using TF-IDF) domain vocabulary overlap over source and target domain vocabularies by getting tokens with TF-IDF higher than a given threshold.
    #     @param self - a Similarity class
    #     @param threshold - float (minimum TF-IDF threshold to be considered in top vocabulary)
    #     @return - float of ratio of overlap between most relevant vocabulary over whole target vocabulary size
    #     """
    #     source_top, source_dict = get_tfidf_dict(
    #         self.source.tfidf.get_feature_names_out(), self.source.tfidf_X, threshold
    #     )
    #     target_top, target_dict = get_tfidf_dict(
    #         self.target.tfidf.get_feature_names_out(), self.target.tfidf_X, threshold
    #     )
    #     overlap = source_top.keys() & target_top.keys()
    #     return len(overlap) / len(target_dict)

    # def compute_relevance_overlap(self) -> float:
    #     """
    #     Computes TF-IDF weighted word overlap of source and target domains (proposed, possible metric).
    #     @param self - Domain class
    #     @returns - a float between [0.0, 1.0] describing word overlap weighted by relevance.
    #     """
    #
    #     gamma = 0
    #     tfidf_by_source = []
    #     vectorizer = self.source.tfidf
    #     for i in range(self.source.dataset_size):
    #         Si = self.source.documents[i]
    #         X = vectorizer.transform([" ".join(Si)])
    #         allWords = vectorizer.get_feature_names_out()
    #         values = np.array(X.todense().tolist()).T.sum(axis=1)  # .to_numpy()
    #         dictionary = dict()
    #         for k, v in zip(allWords, values):
    #             dictionary[k] = v
    #         tfidf_by_source.append(dictionary)
    #     for i in range(self.source.dataset_size):
    #         Si = self.source.documents[i]
    #         for j in range(self.target.dataset_size):
    #             Tj_unique = set(self.target.documents[j])
    #             for word in Tj_unique:
    #                 try:
    #                     weight = tfidf_by_source[i][word]
    #                 except:
    #                     weight = 0
    #                 if len(Si) == 0:
    #                     continue
    #                 gamma += (
    #                     Si.count(word)
    #                     * (1 / (1 - weight + sys.float_info.min) ** 2)
    #                     / len(Si)
    #                 )
    #     return gamma / self.source.dataset_size / self.target.dataset_size
    def compute_relevance_overlap(self) -> float:
        """
        Computes TF-IDF weighted word overlap of source and target domains (proposed, possible metric).
        @param self - Domain class
        @returns - a float between [0.0, 1.0] describing word overlap weighted by relevance.
        """
        vocab1 = set(self.source.tfidf_top_words)
        vocab2 = set(self.target.tfidf_top_words)

        # Calculate precision
        overlap = len(vocab1.intersection(vocab2))
        # Calculate percentage overlap of source (high-resource) with respect to target (low-resource)
        if len(vocab2) > 0:
            overlap_percentage = (overlap / len(vocab2)) * 100
        else:
            overlap_percentage = 0

        return overlap_percentage

    # def compute_sentence_similarity(self) -> float:
    #     """
    #     Computes sentence overlap of source and target domains using OpenAI Embeddings and cosine similarity.
    #     @param self - a Similarity class
    #     @return - a float between [0.0, 1.0] describing sentence overlap
    #     """
    #     gamma = 0
    #     if self.client:
    #         for i in range(self.source.dataset_size):
    #             Si = self.source.sentence_embeddings[i]
    #             for j in range(self.target.dataset_size):
    #                 Tj = self.target.sentence_embeddings[j]
    #                 # Compute number if similar sentences
    #                 count = np.sum(is_similar(Tj, Si))
    #                 gamma += count / len(Si) / len(Tj)
    #     return gamma / self.source.dataset_size / self.target.dataset_size
    def compute_contextual_similarity(self) -> float:
        """
        Computes sentence overlap of source and target domains using OpenAI Embeddings and cosine similarity.
        @param self - a Similarity class
        @return - a float between [-1.0, 1.0] describing sentence overlap
        """

        if self.target.dataset_size > self.source.dataset_size:
            source_embedding = np.array(self.source.sentence_embeddings)
            target_embedding = np.array(self.target.sentence_embeddings)[:self.source.dataset_size]
        elif self.target.dataset_size < self.source.dataset_size:
            source_embedding = np.array(self.source.sentence_embeddings)[:self.target.dataset_size]
            target_embedding = np.array(self.target.sentence_embeddings)
        else:
            source_embedding = np.array(self.source.sentence_embeddings)
            target_embedding = np.array(self.target.sentence_embeddings)

        if source_embedding.shape[0] == 1:
            source_embedding = source_embedding.reshape(1, -1)
            target_embedding = target_embedding.reshape(1, -1)

        scores = cosine_similarity(source_embedding, target_embedding)
        scores = np.mean(scores)
        return scores

    def get_global_prob_distribution(self):
        s_prob_dist_words = list(self.source.prob_dist.keys())
        t_prob_dist_words = list(self.target.prob_dist.keys())

        # add the source distribution words to target, and target distribution work to course, to a have a global representation which is needed to compute kl divergence.
        s_updated_prob_dist = self.source.prob_dist
        for t_word in t_prob_dist_words:
            if t_word not in s_updated_prob_dist:
                s_updated_prob_dist[t_word] = sys.float_info.epsilon

        t_updated_prob_dist = self.target.prob_dist
        for s_word in s_prob_dist_words:
            if s_word not in t_updated_prob_dist:
                t_updated_prob_dist[s_word] = sys.float_info.epsilon

        s_updated_prob_dist = {k: v for k, v in sorted(s_updated_prob_dist.items(), key=lambda item: item[0])}
        t_updated_prob_dist = {k: v for k, v in sorted(t_updated_prob_dist.items(), key=lambda item: item[0])}
        return s_updated_prob_dist, t_updated_prob_dist

    def compute_kl_divergence(self) -> float:

        s_pd = list(self.s_updated_prob_dist.values())
        t_pd = list(self.t_updated_prob_dist.values())
        kl_div = scipy.special.kl_div(s_pd, t_pd)
        kl_div = np.sum(kl_div)
        return kl_div

    def compute_js_divergence(self) -> float:

        s_pd = list(self.s_updated_prob_dist.values())
        t_pd = list(self.t_updated_prob_dist.values())
        js_div = jensenshannon(s_pd, t_pd)
        js_div = np.sum(js_div)
        return js_div

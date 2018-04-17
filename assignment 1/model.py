"""
Main module describing the IBM model 1 and 2 logic.
"""

# STD
import abc
from collections import defaultdict
import random
from itertools import product

# PROJECT
from corpus import ParallelCorpus


class Model:
    """
    Super class defining shared functions and variables for IBM models 1 and 2.
    """
    def __init__(self):
        self.cooc_counts = defaultdict(int)  # p(e_l, f_k): Co-occurrences between source and target language words
        self.source_counts = defaultdict(int)  # Count of words in the source language
        self.translation_probs = defaultdict(lambda: random.random())

    @abc.abstractmethod
    def train(self, data: ParallelCorpus, epochs=10):
        pass

    @staticmethod
    def add_null_token(sentence):
        return ["NULL"] + sentence


class Model1(Model):
    """
    Class for IBM model 1.
    """
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon  # Normalization constant

    def train(self, data: ParallelCorpus, epochs=10):
        for epoch in range(epochs):
            print("Starting epoch #{}...".format(epoch+1))
            self.cooc_counts = defaultdict(int)
            self.source_counts = defaultdict(int)

            for i, (source_sentence, target_sentence) in enumerate(data):
                print(
                    "\rProcessing sentence {}/{} ({:.2f} %)...".format(i+1, len(data), (i+1)/len(data)),
                    end="", flush=True
                )
                source_sentence = self.add_null_token(source_sentence)
                norm = 0

                # Compute normalization
                for (source_token, target_token) in product(source_sentence, target_sentence):
                    norm += self.translation_probs[(source_token, target_token)]

                # Collect counts
                for (source_token, target_token) in product(source_sentence, target_sentence):
                    pair = (source_token, target_token)
                    self.cooc_counts[pair] += self.translation_probs[pair] / norm
                    self.source_counts[source_token] += self.translation_probs[pair] / norm

                # Estimate probabilities
                for (source_token, target_token) in product(source_sentence, target_sentence):
                    pair = (source_token, target_token)
                    self.translation_probs[pair] = self.cooc_counts[pair] / self.source_counts[source_token]

        print(self.translation_probs)



class Model2(Model):
    """
    Class for IBM model 2.
    """
    def __init__(self):
        super().__init__()
        self.alignment_counts = defaultdict(int)  # c(j|i, m, l)
        self.alignment_probs = defaultdict(float)


if __name__ == "__main__":
    corpus = ParallelCorpus(
        source_path="./data/training/hansards.36.2.f", target_path="./data/training/hansards.36.2.e"
    )
    model1 = Model1(epsilon=0.1)
    model1.train(corpus)
